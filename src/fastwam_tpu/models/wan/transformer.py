"""The Wan2.2-TI2V-5B video expert: 30 DiT blocks, d=3072, ffn=14336.

M1 implements this.

Ported from
    reference/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py
        WanTransformerBlock / WanModel — strip I2V, image_embedder, magcache,
        skip_blocks; add a self-attention mask argument; and split the block into
        pre_attention / post_attention halves so mot.py can drive it.
    Semantics from reference/FastWAM/src/fastwam/models/wan22/wan_video_dit.py
        DiTBlock (L249-268), Head, WanVideoDiT.pre_dit (L509-620) / post_dit,
        build_video_to_video_mask (L473-507)

Block (spec §2.2), with every piece of the ordering load-bearing:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunk(modulation + t_mod, 6)
    x = x + gate_msa * SelfAttn(norm1(x) * (1 + scale_msa) + shift_msa, rope, self_mask)
    x = x + CrossAttn(norm_cross(x), context, context_mask)          # no gate
    x = x + gate_mlp * FFN(norm_mlp(x) * (1 + scale_mlp) + shift_mlp)

    norm1 / norm_mlp: LayerNorm(eps=1e-6, no affine); norm_cross: LayerNorm(eps=1e-6,
    affine). `modulation` is a learned [1, 6, hidden] parameter ADDED to t_mod before
    chunking. FFN = Linear(H, ffn) -> GELU(tanh) -> Linear(ffn, H).

NAMING (M1 decision, the one thing to read before touching weight conversion)
    FastWAM calls the pre-cross-attention norm `norm3` and the pre-MLP norm `norm2`;
    maxdiffusion swaps those two names. This port uses **maxdiffusion's** names
    throughout — `norm2` is the affine pre-cross-attention norm, `norm3` is the
    non-affine pre-MLP norm — and likewise `attn1`/`attn2`, `ffn.act_fn.proj`/
    `ffn.proj_out`, `adaln_scale_shift_table`, `condition_embedder.*`, `proj_out`,
    `scale_shift_table`.
    Reason: the converted Wan2.2-5B weights published by FastWAM-JAX (TinyWAM-wan22's
    Orbax tree, 825 keys) are stored under exactly those names, so the mapping from
    checkpoint to module is an identity check rather than a rename table. A rename
    table is precisely the kind of thing that mis-pairs norm2/norm3 silently and
    costs a closed-loop run to discover. The docstring comments below say which
    FastWAM name each one corresponds to.

FP32LayerNorm (spec §9.3 item 12, and a known deviation)
    Follows maxdiffusion: upcast to fp32, normalise, cast back. Under bf16 training
    the official model normalises in bf16 with fp32 accumulation, so the two differ
    at the sub-ulp level. Recorded here because it is one of the few remaining
    numerical deltas against the reference and therefore a candidate explanation if
    M3's closed-loop score comes out wrong.

Shapes (RoboTwin, spec §4.5)
    latents [B,48,3,24,20] -> tokens [B,360,3072]; context [B,129,3072];
    t [B,360,3072]; t_mod [B,360,6,3072]; output [B,48,3,24,20].
"""

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ...config import VideoDiTConfig
from .attention import CrossAttention, SelfAttention, dot_product_attention
from .embeddings import ConditionEmbedder, PatchEmbedding
from .rope import precompute_freqs_3d
from .tree_keys import restore_tree_key_types, stringify_tree_keys

__all__ = [
    "WanVideoDiTConfigError",
    "FP32LayerNorm",
    "WanFeedForward",
    "video_self_attention_mask_numpy",
    "build_video_self_attention_mask",
    "build_token_timesteps",
    "unpatchify",
    "VideoPreState",
    "WanDiTBlock",
    "WanVideoDiT",
    "stringify_tree_keys",
    "restore_tree_key_types",
    "load_video_dit",
]

VIDEO_MASK_MODES = ("first_frame_causal", "bidirectional")
REMAT_POLICIES = ("none", "full")


class WanVideoDiTConfigError(ValueError):
    """Raised for config combinations the port does not implement."""


class FP32LayerNorm(nnx.Module):
    """LayerNorm that always normalises in fp32 and casts back.

    Behaviour of maxdiffusion `models/normalization_flax.py:153`, written the way
    codebases/TinyWAM-wan22 `tinywam/models/wan/layers.py:23` writes it — with the
    affine parameters held directly rather than inside a wrapped `nnx.LayerNorm`.
    That is not cosmetic: the converted Wan2.2-5B tree stores `blocks.N.norm2.scale`
    and `.bias`, and wrapping would put them at `norm2.layer_norm.scale`, which the
    key audit in tools/convert_weights.py flags immediately.

    Holding the parameters also pins the variance formula to `mean((x-mean)^2)`;
    `nnx.LayerNorm` defaults to `use_fast_variance=True`, i.e. `E[x^2]-E[x]^2`, which
    is mathematically equal but not bit-equal in fp32.

    `elementwise_affine=False` means no parameters at all, which is why the 5B
    checkpoint has `norm2` keys but no `norm1` / `norm3` / `norm_out` keys.
    """

    def __init__(self, *, rngs: nnx.Rngs, dim: int, eps: float, elementwise_affine: bool):
        del rngs  # affine params are constant-initialised; kept for a uniform signature
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.scale = nnx.Param(jnp.ones((dim,), jnp.float32))
            self.bias = nnx.Param(jnp.zeros((dim,), jnp.float32))
        else:
            self.scale = None
            self.bias = None

    def __call__(self, inputs: jax.Array) -> jax.Array:
        orig_dtype = inputs.dtype
        x = inputs.astype(jnp.float32)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.elementwise_affine:
            x = x * self.scale[...].astype(jnp.float32) + self.bias[...].astype(jnp.float32)
        return x.astype(orig_dtype)


class _ApproximateGELU(nnx.Module):
    """`ffn.act_fn` in the checkpoint: Linear(dim, ffn_dim) then GELU(tanh)."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        dim_in: int,
        dim_out: int,
        dtype: jnp.dtype,
        param_dtype: jnp.dtype,
        precision: jax.lax.Precision | None,
    ):
        self.proj = nnx.Linear(
            rngs=rngs,
            in_features=dim_in,
            out_features=dim_out,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.nn.gelu(self.proj(x))


class WanFeedForward(nnx.Module):
    """Linear(H, ffn) -> GELU(tanh) -> Linear(ffn, H), both biased."""

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        dim: int,
        ffn_dim: int,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
    ):
        self.act_fn = _ApproximateGELU(
            rngs=rngs, dim_in=dim, dim_out=ffn_dim, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        self.proj_out = nnx.Linear(
            rngs=rngs,
            in_features=ffn_dim,
            out_features=dim,
            use_bias=True,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.proj_out(self.act_fn(x))


def video_self_attention_mask_numpy(tokens_per_frame: int, num_frames: int) -> np.ndarray:
    """The numpy half of :func:`build_video_self_attention_mask`.

    Split out in M2 so `mot.py` can embed this block inside the larger joint mask
    without a jnp -> numpy round trip. That round trip is not merely inelegant: the
    joint mask is built inside the jitted forward, where `np.asarray` on a traced array
    raises TracerArrayConversionError. Everything here is host-side constant folding,
    and `jnp.asarray` at the very end is what makes it a literal in the jaxpr.
    """
    if tokens_per_frame <= 0 or num_frames <= 0:
        raise WanVideoDiTConfigError(
            f"tokens_per_frame and num_frames must be positive, got {tokens_per_frame}, {num_frames}"
        )
    seq_len = tokens_per_frame * num_frames
    mask = np.ones((seq_len, seq_len), dtype=bool)
    mask[:tokens_per_frame, tokens_per_frame:] = False
    return mask


def build_video_self_attention_mask(tokens_per_frame: int, num_frames: int) -> jax.Array:
    """`first_frame_causal`: frame-0 queries see only frame-0 keys.

        mask = ones(Sv, Sv); mask[:tokens_per_frame, tokens_per_frame:] = False

    This is what makes the first frame's per-layer K/V independent of the future
    video, which is what makes the KV-cache inference path exact.

    Returns: [Sv, Sv] bool, Sv = tokens_per_frame * num_frames (360 for RoboTwin).
    """
    return jnp.asarray(video_self_attention_mask_numpy(tokens_per_frame, num_frames))


def unpatchify(x: jax.Array, grid: tuple[int, int, int], patch_size: tuple[int, int, int]) -> jax.Array:
    """'b (f h w) (x y z c) -> b c (f x) (h y) (w z)' (wan_video_dit.py:410-415).

    The channel index is LAST inside the packed dim (spec §9.5 item 23) — getting
    that backwards produces a plausible-looking tensor with scrambled channels.
    """
    batch = x.shape[0]
    p_t, p_h, p_w = patch_size
    f, h, w = grid
    x = x.reshape(batch, f, h, w, p_t, p_h, p_w, -1)
    x = jnp.transpose(x, (0, 7, 1, 4, 2, 5, 3, 6))
    return x.reshape(batch, -1, f * p_t, h * p_h, w * p_w)


def build_token_timesteps(timestep: jax.Array, num_frames: int, tokens_per_frame: int) -> jax.Array:
    """[B] -> [B, num_frames*tokens_per_frame], with latent frame 0 forced to 0.

    `token_timesteps[:, 0, :] = 0` (wan_video_dit.py:548) is the single line the whole
    KV-cache equivalence rests on: latent frame 0 is declared clean during training
    AND at inference (spec §9.5 item 24).
    """
    if timestep.ndim != 1:
        raise WanVideoDiTConfigError(f"timestep must be [B], got {timestep.shape}")
    per_frame = jnp.broadcast_to(
        timestep.astype(jnp.float32)[:, None, None], (timestep.shape[0], num_frames, tokens_per_frame)
    )
    return per_frame.at[:, 0, :].set(0.0).reshape(timestep.shape[0], -1)


class VideoPreState(NamedTuple):
    """Everything the blocks need, in the order the M0 stub documented."""

    tokens: jax.Array  # [B, Sv, hidden]
    t: jax.Array  # [B, Sv, hidden]        — Head consumes this, NOT t_mod
    t_mod: jax.Array  # [B, Sv, 6, hidden]
    context: jax.Array  # [B, L, hidden]
    context_mask: jax.Array | None  # [B, Sv, L]
    cos: jax.Array  # [Sv, head_dim//2]
    sin: jax.Array  # [Sv, head_dim//2]
    grid: tuple[int, int, int]  # (f, h, w) post-patch token grid
    tokens_per_frame: int


class WanDiTBlock(nnx.Module):
    """One of the 30 blocks. Split in two so the MoT can run the mixed attention
    across both experts between the halves.

    pre_attention(x, t_mod, cos, sin)
        -> (q, k, v) each [B, S, 24, 128], and the gates the second half needs.
    post_attention(x, attn_out, gates, context, context_mask)
        -> [B, S, hidden]
    __call__(...) runs both with a single-expert self-attention (used by the video
    tower alone, e.g. the prefill path).
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
    ):
        self.dim = dim
        attn_kwargs = dict(
            rngs=rngs,
            query_dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.norm1 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
        self.attn1 = SelfAttention(**attn_kwargs)
        # norm2 here == FastWAM's norm3: the affine norm before cross-attention.
        self.norm2 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=True)
        self.attn2 = CrossAttention(**attn_kwargs)
        # norm3 here == FastWAM's norm2: the non-affine norm before the FFN.
        self.norm3 = FP32LayerNorm(rngs=rngs, dim=dim, eps=eps, elementwise_affine=False)
        self.ffn = WanFeedForward(
            rngs=rngs, dim=dim, ffn_dim=ffn_dim, dtype=dtype, param_dtype=param_dtype, precision=precision
        )
        # FastWAM calls this `modulation`; init is randn/sqrt(dim) in both.
        self.adaln_scale_shift_table = nnx.Param(
            jax.random.normal(rngs.params(), (1, 6, dim), dtype=param_dtype) / dim**0.5
        )

    def modulation(self, t_mod: jax.Array) -> tuple[jax.Array, ...]:
        """(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp).

        t_mod [B, S, 6, dim] -> per-token modulation, each part [B, S, dim].
        t_mod [B, 6, dim]    -> shared modulation, each part [B, 1, dim].

        The fp32 upcast is maxdiffusion's; FastWAM adds `modulation` in t_mod's own
        dtype, i.e. bf16 under the official training recipe. Keeping the upcast means
        the modulation path stays fp32 all the way through, which is the safer
        numerics and, per experiments/005, also a measurable slice of step time — an
        M5 lever, not an M1 one.
        """
        table = self.adaln_scale_shift_table[...]
        if t_mod.ndim == 4:
            parts = jnp.split(table[None] + t_mod.astype(jnp.float32), 6, axis=2)
            return tuple(part.squeeze(2) for part in parts)
        if t_mod.ndim == 3:
            return tuple(jnp.split(table + t_mod.astype(jnp.float32), 6, axis=1))
        raise WanVideoDiTConfigError(f"t_mod must be [B,6,dim] or [B,S,6,dim], got {t_mod.shape}")

    def pre_attention(
        self,
        x: jax.Array,
        t_mod: jax.Array,
        cos: jax.Array | None = None,
        sin: jax.Array | None = None,
    ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], tuple[jax.Array, ...]]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(t_mod)
        normed = (self.norm1(x.astype(jnp.float32)) * (1 + scale_msa) + shift_msa).astype(x.dtype)
        qkv = self.attn1.project_qkv(normed, cos, sin)
        return qkv, (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(
        self,
        x: jax.Array,
        attn_out: jax.Array,
        gates: tuple[jax.Array, ...],
        context: jax.Array | None = None,
        context_mask: jax.Array | None = None,
        cached_kv: tuple[jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        gate_msa, shift_mlp, scale_mlp, gate_mlp = gates
        projected = self.attn1.project_out(attn_out)
        x = (x.astype(jnp.float32) + projected * gate_msa).astype(x.dtype)

        # Cross-attention is NOT gated (spec §9.2 item 9).
        cross_in = self.norm2(x.astype(jnp.float32)).astype(x.dtype)
        cross_context = None if cached_kv is not None else context
        x = x + self.attn2(cross_in, cross_context, context_mask, cached_kv=cached_kv)

        ffn_in = (self.norm3(x.astype(jnp.float32)) * (1 + scale_mlp) + shift_mlp).astype(x.dtype)
        return (x.astype(jnp.float32) + self.ffn(ffn_in).astype(jnp.float32) * gate_mlp).astype(x.dtype)

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array,
        t_mod: jax.Array,
        cos: jax.Array | None = None,
        sin: jax.Array | None = None,
        self_attn_mask: jax.Array | None = None,
        context_mask: jax.Array | None = None,
        cached_kv: tuple[jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        (query, key, value), gates = self.pre_attention(x, t_mod, cos, sin)
        attn_out = dot_product_attention(query, key, value, self_attn_mask, precision=self.attn1.precision)
        return self.post_attention(x, attn_out, gates, context, context_mask, cached_kv)

    def compute_context_kv(self, context: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Cross-attention K/V for a context that is constant across denoising steps."""
        return self.attn2.compute_kv(context)


class WanVideoDiT(nnx.Module):
    """The video expert.

    pre_dit(latents, timestep, context, context_mask)
        latents:      [B, 48, T_l, H_l, W_l]
        timestep:     [B] float, per sample
        context:      [B, 128, 4096] (text) -- proprio token appended by fastwam.py
        context_mask: [B, 128] bool
        ->
        VideoPreState(tokens [B,Sv,3072], t [B,Sv,3072], t_mod [B,Sv,6,3072],
                      context [B,L,3072], context_mask [B,Sv,L], cos/sin [Sv,64], ...)

        The per-token timestep is ones(B, T_l, tokens_per_frame) * timestep, then
        token_timesteps[:, 0, :] = 0 -- latent frame 0 is declared clean in training
        AND at inference (spec §9.5 item 24). The whole KV-cache equivalence rests
        on this line.

    post_dit(tokens, t) -> [B, 48, T_l, H_l, W_l]
        Head = LayerNorm(no affine) -> Linear(3072, 48*1*2*2=192), with its own
        [1,2,3072] modulation, then unpatchify
        rearrange('b (f h w) (x y z c) -> b c (f x) (h y) (w z)') -- the channel
        index is LAST inside the packed dim (spec §9.5 item 23).

    `per_token_timestep=False` selects maxdiffusion's global-timestep path (temb
    [B,6,dim] shared by every token). No FastWAM config uses it — config.py forbids
    `seperated_timestep: false` — but it is the path the action expert's rank-3 t_mod
    exercises in M2, and keeping it here is what lets M1 parity-test both branches
    against the same golden.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        in_channels: int,
        out_channels: int,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        freq_dim: int,
        text_dim: int,
        patch_size: tuple[int, int, int],
        eps: float = 1e-6,
        per_token_timestep: bool = True,
        video_attention_mask_mode: str = "first_frame_causal",
        scan_layers: bool = False,
        remat_policy: str = "none",
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        freqs_dtype=np.float64,
    ):
        if hidden_dim != num_heads * head_dim:
            raise WanVideoDiTConfigError(
                f"the video expert requires hidden_dim == num_heads*head_dim, "
                f"got {hidden_dim} vs {num_heads * head_dim}"
            )
        if video_attention_mask_mode not in VIDEO_MASK_MODES:
            raise WanVideoDiTConfigError(
                f"video_attention_mask_mode must be one of {VIDEO_MASK_MODES}, "
                f"got {video_attention_mask_mode!r} (per_frame_causal is unused upstream "
                "and is not ported)"
            )
        if remat_policy not in REMAT_POLICIES:
            raise WanVideoDiTConfigError(
                f"remat_policy must be one of {REMAT_POLICIES}, got {remat_policy!r}; "
                "the full six-level dial lands with training/remat.py in M5"
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.patch_size = tuple(patch_size)
        self.eps = eps
        self.per_token_timestep = per_token_timestep
        self.video_attention_mask_mode = video_attention_mask_mode
        self.scan_layers = scan_layers
        self.remat_policy = remat_policy
        self.dtype = dtype
        self.precision = precision
        self.freqs_dtype = freqs_dtype

        self.patch_embedding = PatchEmbedding(
            rngs=rngs,
            in_channels=in_channels,
            out_features=hidden_dim,
            patch_size=self.patch_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        self.condition_embedder = ConditionEmbedder(
            rngs=rngs,
            hidden_dim=hidden_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            param_dtype=param_dtype,
            precision=precision,
            freqs_dtype=freqs_dtype,
        )

        block_kwargs = dict(
            dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        if scan_layers:
            # maxdiffusion's pattern: vmap the constructor so every parameter carries a
            # leading [num_layers] axis that nnx.scan can slice.
            @nnx.split_rngs(splits=num_layers)
            @nnx.vmap(in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"})
            def init_block(block_rngs: nnx.Rngs) -> WanDiTBlock:
                return WanDiTBlock(rngs=block_rngs, **block_kwargs)

            self.blocks = init_block(rngs)
        else:
            self.blocks = nnx.data([WanDiTBlock(rngs=rngs, **block_kwargs) for _ in range(num_layers)])

        self.norm_out = FP32LayerNorm(rngs=rngs, dim=hidden_dim, eps=eps, elementwise_affine=False)
        self.proj_out = nnx.Linear(
            rngs=rngs,
            in_features=hidden_dim,
            out_features=out_channels * int(np.prod(self.patch_size)),
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )
        self.scale_shift_table = nnx.Param(
            jax.random.normal(rngs.params(), (1, 2, hidden_dim), dtype=param_dtype) / hidden_dim**0.5
        )

    # ------------------------------------------------------------------ #
    # construction from the project config
    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, config: VideoDiTConfig, *, rngs: nnx.Rngs, **overrides) -> "WanVideoDiT":
        """Build straight from `Config.model.video_dit_config` — no pyconfig, no yaml."""
        if config.has_image_input or config.require_clip_embedding or config.require_vae_embedding:
            raise WanVideoDiTConfigError("the I2V / CLIP / VAE-embedding branches are not ported")
        if config.action_conditioned:
            raise WanVideoDiTConfigError("the action-conditioned branch is dead upstream and not ported")
        per_token = config.seperated_timestep and config.fuse_vae_embedding_in_latents
        if not per_token:
            raise WanVideoDiTConfigError(
                "FastWAM requires seperated_timestep and fuse_vae_embedding_in_latents; "
                "the global-timestep path exists for parity tests only"
            )
        kwargs = dict(
            in_channels=config.in_dim,
            out_channels=config.out_dim,
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            num_heads=config.num_heads,
            head_dim=config.attn_head_dim,
            num_layers=config.num_layers,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            patch_size=config.patch_size,
            eps=config.eps,
            per_token_timestep=True,
            video_attention_mask_mode=config.video_attention_mask_mode,
        )
        kwargs.update(overrides)
        return cls(rngs=rngs, **kwargs)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def rope_tables(self, grid: tuple[int, int, int]) -> tuple[jax.Array, jax.Array]:
        return precompute_freqs_3d(self.head_dim, *grid, freqs_dtype=self.freqs_dtype)

    def self_attention_mask(self, grid: tuple[int, int, int]) -> jax.Array | None:
        if self.video_attention_mask_mode == "bidirectional":
            return None
        return build_video_self_attention_mask(grid[1] * grid[2], grid[0])

    def pre_dit(
        self,
        latents: jax.Array,
        timestep: jax.Array,
        context: jax.Array,
        context_mask: jax.Array | None = None,
    ) -> VideoPreState:
        grid = self.patch_embedding.grid_size(latents.shape)
        tokens_per_frame = grid[1] * grid[2]
        tokens = self.patch_embedding(latents)

        if timestep.ndim == 2:
            token_timesteps = timestep.astype(jnp.float32)
        elif self.per_token_timestep:
            token_timesteps = build_token_timesteps(timestep, grid[0], tokens_per_frame)
        else:
            token_timesteps = timestep
        if token_timesteps.ndim == 2 and token_timesteps.shape[1] != tokens.shape[1]:
            raise WanVideoDiTConfigError(
                f"per-token timestep length {token_timesteps.shape[1]} != {tokens.shape[1]} tokens"
            )
        t, t_mod = self.condition_embedder.embed_time(token_timesteps)

        context_emb = self.condition_embedder.embed_text(context)
        expanded_mask = None
        if context_mask is not None:
            if context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
                raise WanVideoDiTConfigError(
                    f"context_mask must be [B, L] matching context {context.shape[:2]}, "
                    f"got {context_mask.shape}"
                )
            # [B, L] -> [B, Sv, L]: one key mask per query token (wan_video_dit.py:598).
            expanded_mask = jnp.broadcast_to(
                context_mask[:, None, :], (context_mask.shape[0], tokens.shape[1], context_mask.shape[1])
            )

        cos, sin = self.rope_tables(grid)
        return VideoPreState(
            tokens=tokens,
            t=t,
            t_mod=t_mod,
            context=context_emb.astype(tokens.dtype),
            context_mask=expanded_mask,
            cos=cos,
            sin=sin,
            grid=grid,
            tokens_per_frame=tokens_per_frame,
        )

    def _block_body(self):
        def body(block: WanDiTBlock, x, context, t_mod, cos, sin, self_attn_mask, context_mask):
            return block(x, context, t_mod, cos, sin, self_attn_mask, context_mask)

        if self.remat_policy == "full":
            return nnx.remat(body, prevent_cse=not self.scan_layers, policy=None)
        return body

    def run_blocks(
        self,
        tokens: jax.Array,
        context: jax.Array,
        t_mod: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        self_attn_mask: jax.Array | None = None,
        context_mask: jax.Array | None = None,
    ) -> jax.Array:
        body = self._block_body()
        args = (context, t_mod, cos, sin, self_attn_mask, context_mask)
        if not self.scan_layers:
            for block in self.blocks:
                tokens = body(block, tokens, *args)
            return tokens

        def scan_fn(carry, block):
            return body(block, carry, *args), None

        carry, _ = nnx.scan(
            scan_fn, length=self.num_layers, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )(tokens, self.blocks)
        return carry

    def post_dit(self, tokens: jax.Array, t: jax.Array, grid: tuple[int, int, int]) -> jax.Array:
        table = self.scale_shift_table[...]
        if t.ndim == 3:  # per-token: [B, S, dim]
            combined = table[None] + t[:, :, None, :]
            shift, scale = (part.squeeze(2) for part in jnp.split(combined, 2, axis=2))
        elif t.ndim == 2:  # global: [B, dim]
            shift, scale = jnp.split(table + t[:, None, :], 2, axis=1)
        else:
            raise WanVideoDiTConfigError(f"t must be [B,dim] or [B,S,dim], got {t.shape}")

        x = (self.norm_out(tokens.astype(jnp.float32)) * (1 + scale) + shift).astype(tokens.dtype)
        return unpatchify(self.proj_out(x), grid, self.patch_size)

    def __call__(
        self,
        latents: jax.Array,
        timestep: jax.Array,
        context: jax.Array,
        context_mask: jax.Array | None = None,
        self_attn_mask: jax.Array | None = None,
    ) -> jax.Array:
        state = self.pre_dit(latents, timestep, context, context_mask)
        if self_attn_mask is None:
            self_attn_mask = self.self_attention_mask(state.grid)
        tokens = self.run_blocks(
            state.tokens,
            state.context,
            state.t_mod,
            state.cos,
            state.sin,
            self_attn_mask,
            state.context_mask,
        )
        return self.post_dit(tokens, state.t, state.grid)


# --------------------------------------------------------------------------- #
# checkpoint loading
#
# `stringify_tree_keys` / `restore_tree_key_types` used to be defined here and again in
# vae.py; M2 made the ActionDiT/MoT loaders a third consumer, so they now live in
# tree_keys.py and are re-exported from this module's namespace for existing importers.
# --------------------------------------------------------------------------- #
def _restore_unfinalized(path: str, target: dict) -> dict:
    """Restore a tree that predates orbax's finalization marker.

    The published base checkpoints (Wan2.2-5B transformer and VAE) were written
    by an older orbax and carry no ``commit_success.txt``. orbax 0.12's
    ``StandardCheckpointer.restore`` treats that as "not finalized" and refuses, even
    though every array is present and readable.

    The team's ruling is to NOT rewrite the shared trees — several projects load them —
    so the loader drops to the handler underneath, which reads arrays without consulting
    the finalization marker. Measured on the real 825-array tree: 19 s, every value
    finite. ``PyTreeCheckpointHandler`` only accepts an ``epath.Path``, never a ``str``.
    """
    import orbax.checkpoint as ocp
    from etils import epath

    return ocp.PyTreeCheckpointHandler().restore(
        epath.Path(path), args=ocp.args.PyTreeRestore(item=target)
    )


def _restore_pure_dict(path: str, template: dict) -> dict:
    """Restore ``template``'s structure from an Orbax dir, tolerating either key style.

    Tries the flax-0.10 string-keyed layout first because that is what every existing
    checkpoint uses, then the flax-0.12 int-keyed layout because that is what this repo
    will write from now on; each of those is tried through the normal checkpointer and
    then through :func:`_restore_unfinalized`. All four failing raises with every cause
    attached — never a partial restore.
    """
    import orbax.checkpoint as ocp

    checkpointer = ocp.StandardCheckpointer()
    errors = {}
    for style, target in (
        ("flax 0.10 string keys", stringify_tree_keys(template)),
        ("flax 0.12 int keys", template),
    ):
        for how, restore in (
            ("standard", lambda t=target: checkpointer.restore(path, target=t)),
            ("unfinalized", lambda t=target: _restore_unfinalized(path, t)),
        ):
            try:
                restored = restore()
            except Exception as exc:  # noqa: BLE001 - every cause is reported below
                errors[f"{style} / {how}"] = exc
                continue
            return restore_tree_key_types(restored, template)

    raise WanVideoDiTConfigError(
        f"restoring the video DiT from {path} failed under every key style and restore path.\n"
        + "\n".join(f"  {style}: {type(exc).__name__}: {exc}" for style, exc in errors.items())
        + "\nThe checkpoint tree must match this module exactly - no partial restores."
    ) from next(iter(errors.values()))


def load_video_dit(
    ckpt_dir: str,
    config: VideoDiTConfig,
    *,
    rngs: nnx.Rngs | None = None,
    **overrides,
) -> "WanVideoDiT":
    """Restore the video expert from an Orbax dir (local or ``gs://``). Any drift raises.

    The on-disk tree is always UNSCANNED (a plain list of 30 blocks). When the model is
    built with ``scan_layers=True`` the block parameters are stacked on a leading
    [num_layers] axis after the restore, which is what ``nnx.scan`` slices.

    MEMORY: this materialises the full parameter tree. At 5 B that is ~20 GB in fp32 --
    fine on a v4-32 host, not on a laptop. The metadata-only audit
    (``tools/convert_weights.py check-transformer``) is the version that runs anywhere.
    """
    rngs = rngs if rngs is not None else nnx.Rngs(0)
    scan_layers = bool(overrides.get("scan_layers", False))
    path = ckpt_dir if ckpt_dir.startswith("gs://") else os.path.abspath(os.path.expanduser(ckpt_dir))

    # The restore target is always the unscanned tree, because that is the on-disk shape.
    # The Rngs is built *inside* the traced function: reusing the caller's would bump its
    # counter at a different trace level, which nnx rejects, and the seed is irrelevant
    # to a tree that only carries shapes.
    flat_overrides = {**overrides, "scan_layers": False}
    unscanned = nnx.eval_shape(lambda: WanVideoDiT.from_config(config, rngs=nnx.Rngs(0), **flat_overrides))
    template = nnx.to_pure_dict(nnx.state(unscanned, nnx.Param))
    restored = _restore_pure_dict(path, template)

    if scan_layers:
        per_layer = [restored["blocks"][index] for index in range(config.num_layers)]
        restored = {
            **{key: value for key, value in restored.items() if key != "blocks"},
            "blocks": jax.tree.map(lambda *layers: jnp.stack(layers, axis=0), *per_layer),
        }

    model = WanVideoDiT.from_config(config, rngs=rngs, **overrides)
    graphdef, state, rest = nnx.split(model, nnx.Param, ...)
    nnx.replace_by_pure_dict(state, restored)
    return nnx.merge(graphdef, state, rest)
