"""Self / cross attention for both experts, with arbitrary boolean masks.

M1 implements the video path; M2 adds the non-square (query_dim != inner_dim) path
that the action expert needs.

Ported from
    reference/maxdiffusion/src/maxdiffusion/models/attention_flax.py
        FlaxWanAttention (L1337-L1802), NNXAttentionOp (L1178), _apply_attention_dot (L707)
    Semantics from reference/FastWAM/src/fastwam/models/wan22/wan_video_dit.py:171-220
        SelfAttention / CrossAttention

Two mandatory deviations from maxdiffusion (report_maxdiffusion.md §3.3 b/c)
    1. maxdiffusion hardcodes in_features == out_features == inner_dim on q/k/v/o.
       The action expert needs Linear(1024 -> 3072) on q/k/v and Linear(3072 -> 1024)
       on o, so query_dim and inner_dim are separate constructor args here.
    2. No splash attention. The masks FastWAM uses are arbitrary 2D/3D boolean
       patterns (first_frame_causal, the MoT block mask, the cached-action mask, and
       a [B,S,L] cross-attention mask) that splash cannot express, and the sequences
       are tiny (392 train / 152 inference), so plain einsum + additive mask is both
       correct and fast. Do not "optimize" this into splash.

Conventions that must not drift (spec §2.2, §9.2, §9.3)
    * norm_q / norm_k are RMSNorm over the FULL inner_dim (3072), applied before
      splitting heads, computed in fp32 then cast back, then multiplied by the
      weight in the original dtype.
    * RoPE is applied to q and k AFTER the RMSNorm, self-attention only. Cross
      attention has no RoPE.
    * Cross attention has NO output gate (spec §9.2 item 9).
    * q/k/v/o all carry a bias.

Layout note: the public q/k/v layout is [B, S, heads, head_dim] — heads on axis 2,
sequence on axis 1 — because M2's MoT concatenates the two experts along the
sequence axis. maxdiffusion transposes to [B, heads, S, head_dim] internally; the
einsum here does the same thing without materialising the transpose.

Precision note: `norm_q`/`norm_k` use `nnx.RMSNorm`, which normalises *and* applies
the scale in the compute dtype, whereas FastWAM normalises in fp32, casts back, then
multiplies by the weight. Identical in fp32; a sub-ulp difference under bf16. This is
the same class of deviation as FP32LayerNorm (see transformer.py) and is recorded
here so it is a candidate explanation if M3's closed-loop numbers look off.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from .rope import apply_rope

__all__ = ["AttentionShapeError", "dot_product_attention", "SelfAttention", "CrossAttention"]


class AttentionShapeError(ValueError):
    """A shape or mask rank that cannot be a correct FastWAM attention call."""


def _expand_mask(mask: jax.Array, q_len: int, kv_len: int) -> jax.Array:
    """[S_q,S_kv] | [B,S_q,S_kv] | [B,1|H,S_q,S_kv] -> broadcastable [*,*,S_q,S_kv]."""
    if mask.shape[-2:] != (q_len, kv_len):
        raise AttentionShapeError(
            f"mask {mask.shape} must end in (q_len, kv_len) = {(q_len, kv_len)}"
        )
    if mask.ndim == 2:
        return mask[None, None, :, :]
    if mask.ndim == 3:
        # [B, S_q, S_kv] — one key mask per query token, per sample (spec §9.2 item 7).
        return mask[:, None, :, :]
    if mask.ndim == 4:
        return mask
    raise AttentionShapeError(f"mask must be 2D, 3D or 4D, got {mask.shape}")


def dot_product_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    mask: jax.Array | None = None,
    *,
    scale: float | None = None,
    precision: jax.lax.Precision | None = None,
) -> jax.Array:
    """Plain scaled dot-product attention with an additive mask.

    Args:
        query: [B, S_q, num_heads, head_dim].
        key / value: [B, S_kv, num_heads, head_dim].
        mask: broadcastable to [B, num_heads, S_q, S_kv]; True = attend.
            A boolean mask is turned into a -inf additive mask internally so the
            caller never has to think about dtype-specific fill values. A float mask
            is added to the logits as-is (no FastWAM path needs this, but the
            interface is the "arbitrary additive mask" one M2/M3 were promised).
        scale: defaults to head_dim ** -0.5.

    Returns:
        [B, S_q, num_heads, head_dim].
    """
    for name, arr in (("query", query), ("key", key), ("value", value)):
        if arr.ndim != 4:
            raise AttentionShapeError(f"{name} must be [B, S, heads, head_dim], got {arr.shape}")
    if key.shape[1] != value.shape[1]:
        raise AttentionShapeError(f"key/value length mismatch: {key.shape[1]} vs {value.shape[1]}")
    if query.shape[2] != key.shape[2] or query.shape[2] != value.shape[2]:
        raise AttentionShapeError("query/key/value must share the head count")
    if query.shape[3] != key.shape[3]:
        raise AttentionShapeError("query/key must share head_dim")

    head_dim = query.shape[3]
    scale = head_dim**-0.5 if scale is None else scale

    # maxdiffusion multiplies the scale after the matmul; keep that order.
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, precision=precision) * scale
    if mask is not None:
        expanded = _expand_mask(mask, query.shape[1], key.shape[1])
        if expanded.dtype == jnp.bool_:
            logits = jnp.where(expanded, logits, jnp.finfo(logits.dtype).min)
        else:
            logits = logits + expanded.astype(logits.dtype)
    probs = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", probs, value, precision=precision)


class _AttentionBase(nnx.Module):
    """The four biased projections plus qk-RMSNorm shared by self and cross attention.

    Parameter names (`query`, `key`, `value`, `proj_attn`, `norm_q`, `norm_k`) are
    maxdiffusion's, which is what the converted Wan2.2-5B tree uses.
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        inner_dim: int | None = None,
        eps: float = 1e-6,
        qk_norm: bool = True,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
    ):
        inner_dim = num_heads * head_dim if inner_dim is None else inner_dim
        if inner_dim != num_heads * head_dim:
            raise AttentionShapeError(
                f"inner_dim {inner_dim} must equal num_heads*head_dim = {num_heads * head_dim}"
            )
        self.query_dim = query_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_norm = qk_norm
        self.dtype = dtype
        self.precision = precision

        def _linear(in_features: int, out_features: int) -> nnx.Linear:
            return nnx.Linear(
                rngs=rngs,
                in_features=in_features,
                out_features=out_features,
                use_bias=True,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                kernel_init=nnx.initializers.lecun_normal(),
            )

        # query_dim -> inner_dim on q/k/v and inner_dim -> query_dim on the output is
        # the whole point of the decoupling: the action expert is 1024 -> 3072 -> 1024.
        self.query = _linear(query_dim, inner_dim)
        self.key = _linear(query_dim, inner_dim)
        self.value = _linear(query_dim, inner_dim)
        self.proj_attn = _linear(inner_dim, query_dim)

        self.norm_q = nnx.data(None)
        self.norm_k = nnx.data(None)
        if qk_norm:
            self.norm_q = nnx.RMSNorm(
                num_features=inner_dim, rngs=rngs, epsilon=eps, dtype=dtype, param_dtype=param_dtype
            )
            self.norm_k = nnx.RMSNorm(
                num_features=inner_dim, rngs=rngs, epsilon=eps, dtype=dtype, param_dtype=param_dtype
            )

    def _split_heads(self, x: jax.Array) -> jax.Array:
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

    def _merge_heads(self, x: jax.Array) -> jax.Array:
        return x.reshape(*x.shape[:-2], self.inner_dim)

    def project_out(self, attn_out: jax.Array) -> jax.Array:
        """[B, S, num_heads, head_dim] -> [B, S, query_dim]."""
        if attn_out.ndim != 4:
            raise AttentionShapeError(f"attn_out must be [B, S, heads, head_dim], got {attn_out.shape}")
        return self.proj_attn(self._merge_heads(attn_out))


class SelfAttention(_AttentionBase):
    """q/k/v: Linear(query_dim, inner_dim); o: Linear(inner_dim, query_dim).

    __call__(x: [B, S, query_dim], cos, sin, mask) -> [B, S, query_dim]

    Also exposes the two halves the MoT needs to interleave experts:
        project_qkv(x, cos, sin) -> (q, k, v) each [B, S, num_heads, head_dim]
        project_out(attn_out: [B, S, num_heads, head_dim]) -> [B, S, query_dim]
    """

    def project_qkv(
        self,
        x: jax.Array,
        cos: jax.Array | None = None,
        sin: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        if self.qk_norm:
            # RMSNorm is over the full inner_dim, BEFORE the head split (spec §2.2).
            query = self.norm_q(query)
            key = self.norm_k(key)
        query, key, value = (self._split_heads(v) for v in (query, key, value))
        if (cos is None) != (sin is None):
            raise AttentionShapeError("cos and sin must be given together")
        if cos is not None:
            query = apply_rope(query, cos, sin)
            key = apply_rope(key, cos, sin)
        return query, key, value

    def __call__(
        self,
        x: jax.Array,
        cos: jax.Array | None = None,
        sin: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> jax.Array:
        query, key, value = self.project_qkv(x, cos, sin)
        attn_out = dot_product_attention(query, key, value, mask, precision=self.precision)
        return self.project_out(attn_out)


class CrossAttention(_AttentionBase):
    """Text (+proprio) cross attention. No RoPE, no gate.

    __call__(x: [B, S, query_dim], context: [B, 129, query_dim], mask: [B, S, 129])
        -> [B, S, query_dim]

    Note the mask is 3D — one key mask per query token (spec §9.2 item 7).

    `compute_kv` / the `cached_kv` argument are maxdiffusion's context-KV cache hook,
    kept because the context is constant across denoising steps. It is NOT the M3
    prefill cache; that one caches the *video self-attention* K/V and is built from
    SelfAttention.project_qkv.
    """

    def compute_kv(self, context: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Context K/V, post-RMSNorm, pre head-split: each [B, L, inner_dim]."""
        key = self.key(context)
        value = self.value(context)
        if self.qk_norm:
            key = self.norm_k(key)
        return key, value

    def __call__(
        self,
        x: jax.Array,
        context: jax.Array | None = None,
        mask: jax.Array | None = None,
        cached_kv: tuple[jax.Array, jax.Array] | None = None,
    ) -> jax.Array:
        if (context is None) == (cached_kv is None):
            raise AttentionShapeError("pass exactly one of context / cached_kv")
        query = self.query(x)
        if self.qk_norm:
            query = self.norm_q(query)
        key, value = cached_kv if cached_kv is not None else self.compute_kv(context)
        query, key, value = (self._split_heads(v) for v in (query, key, value))
        attn_out = dot_product_attention(query, key, value, mask, precision=self.precision)
        return self.project_out(attn_out)
