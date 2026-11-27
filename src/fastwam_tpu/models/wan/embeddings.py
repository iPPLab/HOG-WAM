"""Timestep / text / patch embeddings shared by both experts.

M1 implements the video-tower pieces; M2 reuses them for the action tower.

Ported from
    reference/maxdiffusion/src/maxdiffusion/models/embeddings_flax.py
        NNXFlaxTimesteps, NNXTimestepEmbedding, NNXPixArtAlphaTextProjection,
        get_sinusoidal_embeddings
    Semantics from reference/FastWAM/src/fastwam/models/wan22/wan_video_dit.py
        sinusoidal_embedding_1d, WanVideoDiT.{patch_embedding,text_embedding,
        time_embedding,time_projection}

Conventions that must not drift (spec §2.2, §2.5, §9.5)
    * sinusoidal_embedding_1d emits concat([cos, sin]) — COS FIRST. Getting the
      order wrong silently corrupts every adaLN modulation.
    * frequencies are 10000 ** (-arange(dim//2) / (dim//2)); torch computes in
      float64, we build fp32 (t <= 1000, so this is safe).
    * time_embedding = Linear(256, D) -> SiLU -> Linear(D, D).
    * time_projection = SiLU -> Linear(D, 6*D); the 6 chunks are
      (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).
    * text_embedding = Linear(4096, D) -> GELU(tanh) -> Linear(D, D).
    * patch_embedding is a Conv3d with kernel == stride == (1,2,2); implement it as
      reshape + dense, not a convolution (spec §9.5 item 22).

Module/parameter naming (M1 decision, see transformer.py for the full note)
    The attribute names here are maxdiffusion's, not FastWAM's, because the converted
    The published Wan2.2-5B weights are stored in exactly that tree
    (`condition_embedder.time_embedder.linear_1.kernel`, ...). Keeping the names makes
    the weight mapping an identity check instead of a rename table — one fewer place
    to get a silent mismatch. `ConditionEmbedder` is therefore the container that owns
    `time_embedder` / `time_proj` / `text_embedder` as siblings; the M0 stub sketched
    the 6-way projection as living *inside* TimestepEmbedding, which would have
    produced a tree the checkpoint cannot fill.

The `freqs_dtype` knob mirrors rope.py: float64 (default) follows torch, float32
reproduces maxdiffusion's table bit-for-bit for JAX-vs-JAX parity.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

__all__ = [
    "EmbeddingShapeError",
    "sinusoidal_embedding_1d",
    "TimestepEmbedding",
    "TextEmbedding",
    "ConditionEmbedder",
    "PatchEmbedding",
]

SINUSOID_THETA = 10000.0


class EmbeddingShapeError(ValueError):
    """A shape that cannot be a correct FastWAM embedding call."""


def _two_product(a: jax.Array, b: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Dekker two-product in fp32: hi = fl(a*b), and hi + lo == a*b to ~fp64 accuracy.

    fp32 has a 24-bit mantissa, so the splitter is 2**12 + 1. The `t - (t - x)`
    idiom is only meaningful under strict IEEE evaluation; XLA does not reassociate
    it, and `test_sinusoidal_embedding_beats_naive_fp32` fails loudly if that ever
    changes.
    """
    splitter = jnp.asarray(4097.0, jnp.float32)
    ta = a * splitter
    a_hi = ta - (ta - a)
    a_lo = a - a_hi
    tb = b * splitter
    b_hi = tb - (tb - b)
    b_lo = b - b_hi
    hi = a * b
    lo = ((a_hi * b_hi - hi) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo
    return hi, lo


def sinusoidal_embedding_1d(dim: int, position: jax.Array, *, freqs_dtype=np.float64) -> jax.Array:
    """Args: dim (256), position [...] float. Returns [..., dim] = concat([cos, sin]).

    FastWAM only ever calls this on a flat [N]; the extra leading dimensions are
    supported because the per-token path is naturally [B, S] and flattening then
    unflattening around it is pure noise.

    Precision (spec §9.1 item 2). torch evaluates the whole thing in float64 and casts
    the *result* to fp32, so the reference sinusoid is correctly rounded. Doing it
    naively in fp32 is not: at t≈1000 the angle is ≈1000 rad, one fp32 ulp there is
    6e-5 rad, and the fp32-rounded frequency table costs another 6e-5 — together
    ~3e-5 of error in every cos/sin, which then rides through the adaLN modulation
    into every block. That is avoidable for a handful of elementwise ops, so it is
    avoided: the frequency table is carried as an (hi, lo) fp32 pair, the product is
    done exactly with a Dekker two-product, and the residual phase is folded back in
    with the first-order angle-addition identity
        cos(h + l) ≈ cos h − l·sin h,   sin(h + l) ≈ sin h + l·cos h
    which is exact to O(l²) ≈ 1e-9. Measured against the float64 reference: 5.9e-8
    (naive fp32: 3.3e-5).
    """
    if dim % 2 != 0:
        raise EmbeddingShapeError(f"sinusoidal dim must be even, got {dim}")
    half = dim // 2
    fd = np.dtype(freqs_dtype)
    # torch: pow(10000, -arange(half)/half), computed in float64.
    exponent = -(np.arange(half, dtype=fd) / np.array(half, dtype=fd))
    inv_freqs = (np.array(SINUSOID_THETA, dtype=fd) ** exponent).astype(np.float64)
    inv_hi = inv_freqs.astype(np.float32)
    inv_lo = (inv_freqs - inv_hi.astype(np.float64)).astype(np.float32)

    pos = position[..., None].astype(jnp.float32)
    angle, residual = _two_product(pos, jnp.asarray(inv_hi))
    residual = residual + pos * jnp.asarray(inv_lo)
    cos_hi, sin_hi = jnp.cos(angle), jnp.sin(angle)
    return jnp.concatenate(
        [cos_hi - sin_hi * residual, sin_hi + cos_hi * residual], axis=-1
    )


class TimestepEmbedding(nnx.Module):
    """Linear(freq_dim, hidden) -> SiLU -> Linear(hidden, hidden).

    __call__(t_sin: [..., freq_dim]) -> [..., hidden]

    The 6-way `time_proj` head lives on :class:`ConditionEmbedder`, one level up, to
    match the checkpoint tree.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        in_features: int,
        hidden_dim: int,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
    ):
        # dtype is fp32 on purpose: maxdiffusion computes the whole time/text branch in
        # fp32 even under bf16 training, and the adaLN modulation inherits that.
        self.linear_1 = nnx.Linear(
            rngs=rngs,
            in_features=in_features,
            out_features=hidden_dim,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )
        self.linear_2 = nnx.Linear(
            rngs=rngs,
            in_features=hidden_dim,
            out_features=hidden_dim,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )

    def __call__(self, t_sin: jax.Array) -> jax.Array:
        return self.linear_2(jax.nn.silu(self.linear_1(t_sin)))


class TextEmbedding(nnx.Module):
    """Linear(4096, hidden) -> GELU(tanh) -> Linear(hidden, hidden).

    __call__(context: [B, 128, 4096]) -> [B, 128, hidden]
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        in_features: int,
        hidden_dim: int,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
    ):
        self.linear_1 = nnx.Linear(
            rngs=rngs,
            in_features=in_features,
            out_features=hidden_dim,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )
        self.linear_2 = nnx.Linear(
            rngs=rngs,
            in_features=hidden_dim,
            out_features=hidden_dim,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )

    def __call__(self, context: jax.Array) -> jax.Array:
        # jax.nn.gelu defaults to approximate=True, i.e. the tanh form FastWAM uses.
        return self.linear_2(jax.nn.gelu(self.linear_1(context)))


class ConditionEmbedder(nnx.Module):
    """`condition_embedder` in the checkpoint: time + 6-way projection + text.

    embed_time(timestep)
        timestep [B]     -> t [B, hidden],    t_mod [B, 6, hidden]        (global)
        timestep [B, S]  -> t [B, S, hidden], t_mod [B, S, 6, hidden]     (per-token)
    embed_text(context: [B, L, text_dim]) -> [B, L, hidden]

    The rank of `t_mod` is what WanDiTBlock switches on: rank 4 means every token
    carries its own modulation, which is how the first video latent frame gets
    timestep 0 while the rest get t (spec §2.2).
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        hidden_dim: int,
        freq_dim: int,
        text_dim: int,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        freqs_dtype=np.float64,
    ):
        self.hidden_dim = hidden_dim
        self.freq_dim = freq_dim
        self.freqs_dtype = freqs_dtype
        self.time_embedder = TimestepEmbedding(
            rngs=rngs,
            in_features=freq_dim,
            hidden_dim=hidden_dim,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.time_proj = nnx.Linear(
            rngs=rngs,
            in_features=hidden_dim,
            out_features=6 * hidden_dim,
            dtype=jnp.float32,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )
        self.text_embedder = TextEmbedding(
            rngs=rngs,
            in_features=text_dim,
            hidden_dim=hidden_dim,
            param_dtype=param_dtype,
            precision=precision,
        )

    def embed_time(self, timestep: jax.Array) -> tuple[jax.Array, jax.Array]:
        if timestep.ndim not in (1, 2):
            raise EmbeddingShapeError(
                f"timestep must be [B] (global) or [B, seq] (per-token), got {timestep.shape}"
            )
        t_sin = sinusoidal_embedding_1d(self.freq_dim, timestep, freqs_dtype=self.freqs_dtype)
        t = self.time_embedder(t_sin)
        t_mod = self.time_proj(jax.nn.silu(t))
        return t, t_mod.reshape(*t_mod.shape[:-1], 6, self.hidden_dim)

    def embed_text(self, context: jax.Array) -> jax.Array:
        return self.text_embedder(context)


class PatchEmbedding(nnx.Module):
    """Conv3d(48, 3072, kernel=stride=(1,2,2)) expressed as reshape + dense.

    __call__(latents: [B, 48, T_l, H_l, W_l]) -> [B, T_l*(H_l//2)*(W_l//2), 3072]
    RoboTwin: [B,48,3,24,20] -> [B, 360, 3072].

    The parameter keeps the Conv layout [p_t, p_h, p_w, in, out] so the converted
    Wan2.2 checkpoint (`patch_embedding.kernel`, shape (1,2,2,48,3072)) loads without
    a reshape. Because kernel == stride, flattening that kernel to
    [p_t*p_h*p_w*in, out] and matmul-ing against patches gathered in (t, h, w, c)
    order is exactly the convolution — and it is a matmul, not a conv, on TPU.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        in_channels: int,
        out_features: int,
        patch_size: tuple[int, int, int],
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
    ):
        self.in_channels = in_channels
        self.out_features = out_features
        self.patch_size = tuple(patch_size)
        self.dtype = dtype
        self.precision = precision
        kernel_shape = (*self.patch_size, in_channels, out_features)
        self.kernel = nnx.Param(nnx.initializers.xavier_uniform()(rngs.params(), kernel_shape, param_dtype))
        self.bias = nnx.Param(jnp.zeros((out_features,), param_dtype))

    def __call__(self, latents: jax.Array) -> jax.Array:
        if latents.ndim != 5:
            raise EmbeddingShapeError(f"latents must be [B, C, T, H, W], got {latents.shape}")
        batch, channels, frames, height, width = latents.shape
        if channels != self.in_channels:
            raise EmbeddingShapeError(f"expected {self.in_channels} latent channels, got {channels}")
        p_t, p_h, p_w = self.patch_size
        for name, size, patch in (("T", frames, p_t), ("H", height, p_h), ("W", width, p_w)):
            if size % patch != 0:
                raise EmbeddingShapeError(f"latent {name}={size} is not divisible by patch {patch}")
        grid = (frames // p_t, height // p_h, width // p_w)

        # [B,C,T,H,W] -> [B,T,H,W,C] -> patches ordered (p_t, p_h, p_w, C), matching
        # the flattened conv kernel.
        x = jnp.transpose(latents, (0, 2, 3, 4, 1))
        x = x.reshape(batch, grid[0], p_t, grid[1], p_h, grid[2], p_w, channels)
        x = jnp.transpose(x, (0, 1, 3, 5, 2, 4, 6, 7))
        x = x.reshape(batch, grid[0] * grid[1] * grid[2], p_t * p_h * p_w * channels)

        kernel = self.kernel[...].reshape(-1, self.out_features)
        x, kernel, bias = (v.astype(self.dtype) for v in (x, kernel, self.bias[...]))
        return jnp.einsum("bsi,io->bso", x, kernel, precision=self.precision) + bias

    def grid_size(self, latents_shape: tuple[int, ...]) -> tuple[int, int, int]:
        """(f, h, w) token grid for a latent shape, without running the layer."""
        _, _, frames, height, width = latents_shape
        p_t, p_h, p_w = self.patch_size
        return frames // p_t, height // p_h, width // p_w
