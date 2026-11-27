"""3D (and 1D) rotary position embeddings — the highest-risk numerics in the port.

M1 implements this.

Responsibility
    Precompute cos/sin tables and apply RoPE to q/k, matching the torch reference
    elementwise. This is the single most likely place for a silent 0%-closed-loop
    bug (plan risk #1), so it is the first thing M1 pins down against a golden.

Ported from
    reference/maxdiffusion/src/maxdiffusion/models/wan/transformers/transformer_wan.py
        WanRotaryPosEmbed / get_frequencies / _apply_rope  (real-arithmetic form)
    reference/maxdiffusion/src/maxdiffusion/models/embeddings_flax.py
        get_1d_rotary_pos_embed
    Semantics from reference/FastWAM/src/fastwam/models/wan22/wan_video_dit.py
        precompute_freqs_cis_3d / precompute_freqs_cis / rope_apply

Conventions that must not drift (spec §2.5, §9.1)
    * Axis split of a 128-d head: frame 128-2*(128//3) = 44 -> 22 complex pairs,
      height 42 -> 21 pairs, width 42 -> 21 pairs. Total 64 pairs = 128 reals.
    * Pairing is INTERLEAVED (even, odd), i.e. torch view_as_complex — NOT the
      Llama rotate-half split.
        out_even = x_even*cos - x_odd*sin
        out_odd  = x_even*sin + x_odd*cos
    * theta = 10000, freqs = 1/theta**(arange(0,d,2)/d).
    * torch builds the tables in float64; TPU has no fp64, so build them in numpy
      float64 on the host and store fp32 constants. Positions are <= a few hundred,
      so fp32 is sufficient — but M1 must prove it elementwise, not assume it.

M1 findings, both measured in tests/test_dit_parity.py
    * `freqs_dtype` exists because the two upstreams disagree on table precision:
      torch builds in float64, maxdiffusion in float32. The default float64 follows
      torch (it is what the released weights were trained against); passing
      float32 reproduces maxdiffusion bit-for-bit so a JAX-vs-JAX parity test can
      isolate model logic from table precision. The gap between the two tables is
      ~1.8e-5 absolute on cos/sin — small, but not zero, so it is measured rather
      than assumed.
    * The axis split uses maxdiffusion's `2*(dim//6)` form rather than FastWAM's
      `dim//3`. They coincide at dim=128 (42/42/44, the only value FastWAM uses) but
      FastWAM's form produces an odd per-axis width for e.g. dim=64, which cannot be
      paired; the maxdiffusion form stays even for every dim, which is what lets the
      parity tests run at a small head_dim.
"""

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["rope_axis_dims", "precompute_freqs_3d", "precompute_freqs_1d", "apply_rope"]

ROPE_THETA = 10000.0


class RopeShapeError(ValueError):
    """A RoPE table or tensor shape that cannot be a correct FastWAM call."""


def rope_axis_dims(attn_head_dim: int) -> tuple[int, int, int]:
    """Split a head into (frame, height, width) widths, in reals.

    At attn_head_dim=128 this is (44, 42, 42), i.e. 22/21/21 complex pairs, which is
    exactly FastWAM's `(dim - 2*(dim//3), dim//3, dim//3)`.
    """
    if attn_head_dim % 2 != 0:
        raise RopeShapeError(f"attn_head_dim must be even, got {attn_head_dim}")
    hw_dim = 2 * (attn_head_dim // 6)
    frame_dim = attn_head_dim - 2 * hw_dim
    if frame_dim <= 0 or frame_dim % 2 != 0:
        raise RopeShapeError(f"attn_head_dim={attn_head_dim} does not split into three even RoPE axes")
    return frame_dim, hw_dim, hw_dim


def _cos_sin_table(dim: int, end: int, theta: float, freqs_dtype) -> tuple[np.ndarray, np.ndarray]:
    """(cos, sin) of `outer(arange(end), 1/theta**(arange(0,dim,2)/dim))`, as fp32.

    Built on the host in `freqs_dtype` (numpy), then stored fp32 — TPU has no fp64.
    """
    if dim % 2 != 0:
        raise RopeShapeError(f"RoPE axis width must be even, got {dim}")
    if end <= 0:
        raise RopeShapeError(f"RoPE table length must be positive, got {end}")
    fd = np.dtype(freqs_dtype)
    exponent = np.arange(0, dim, 2, dtype=fd) / np.array(dim, dtype=fd)
    inv_freqs = np.ones((), dtype=fd) / (np.array(theta, dtype=fd) ** exponent)
    angles = np.outer(np.arange(end, dtype=fd), inv_freqs)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def precompute_freqs_1d(
    attn_head_dim: int,
    end: int = 1024,
    *,
    theta: float = ROPE_THETA,
    freqs_dtype=np.float64,
) -> tuple[jax.Array, jax.Array]:
    """Action-tower 1D RoPE over the action-step axis (action_dit.py: end=1024).

    Returns:
        (cos, sin), each [end, attn_head_dim // 2] float32.
    """
    cos, sin = _cos_sin_table(attn_head_dim, end, theta, freqs_dtype)
    return jnp.asarray(cos), jnp.asarray(sin)


def precompute_freqs_3d(
    attn_head_dim: int,
    latent_frames: int,
    latent_height_tokens: int,
    latent_width_tokens: int,
    *,
    theta: float = ROPE_THETA,
    freqs_dtype=np.float64,
) -> tuple[jax.Array, jax.Array]:
    """Video-tower 3D RoPE tables, flattened in (f, h, w) token order.

    Args:
        attn_head_dim: 128.
        latent_frames / latent_height_tokens / latent_width_tokens: 3 / 12 / 10 for
            RoboTwin (post-patch token grid).

    Returns:
        (cos, sin), each [f*h*w, attn_head_dim // 2] float32.
    """
    grid = (latent_frames, latent_height_tokens, latent_width_tokens)
    if any(n <= 0 for n in grid):
        raise RopeShapeError(f"token grid must be positive, got {grid}")

    parts_cos, parts_sin = [], []
    for dim, size in zip(rope_axis_dims(attn_head_dim), grid):
        cos, sin = _cos_sin_table(dim, size, theta, freqs_dtype)
        parts_cos.append(cos)
        parts_sin.append(sin)

    def _grid_broadcast(parts: list[np.ndarray]) -> np.ndarray:
        # (f,1,1,df) + (1,h,1,dh) + (1,1,w,dw) -> (f,h,w,head_dim//2) -> (f*h*w, ...)
        f, h, w = grid
        blocks = [
            np.broadcast_to(parts[0][:, None, None, :], (f, h, w, parts[0].shape[-1])),
            np.broadcast_to(parts[1][None, :, None, :], (f, h, w, parts[1].shape[-1])),
            np.broadcast_to(parts[2][None, None, :, :], (f, h, w, parts[2].shape[-1])),
        ]
        return np.concatenate(blocks, axis=-1).reshape(f * h * w, -1)

    cos = _grid_broadcast(parts_cos)
    sin = _grid_broadcast(parts_sin)
    if cos.shape[-1] != attn_head_dim // 2:
        raise RopeShapeError(f"3D RoPE produced {cos.shape[-1]} pairs, expected {attn_head_dim // 2}")
    return jnp.asarray(cos), jnp.asarray(sin)


def apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate q or k in place-of-heads layout.

    Args:
        x: [B, S, num_heads, head_dim].
        cos, sin: [S, head_dim // 2].

    Returns:
        [B, S, num_heads, head_dim], same dtype as x (rotation is done in fp32).
    """
    if x.ndim != 4:
        raise RopeShapeError(f"apply_rope expects [B, S, heads, head_dim], got {x.shape}")
    seq_len, head_dim = x.shape[1], x.shape[3]
    if cos.shape != sin.shape:
        raise RopeShapeError(f"cos {cos.shape} and sin {sin.shape} must have the same shape")
    if cos.shape != (seq_len, head_dim // 2):
        raise RopeShapeError(
            f"RoPE table {cos.shape} does not match tokens/head_dim {(seq_len, head_dim // 2)}"
        )

    out_dtype = x.dtype
    x_pairs = x.astype(jnp.float32).reshape(*x.shape[:-1], head_dim // 2, 2)
    even, odd = x_pairs[..., 0], x_pairs[..., 1]
    # [S, D/2] -> [1, S, 1, D/2] so it broadcasts over batch and heads.
    cos_b = cos.astype(jnp.float32)[None, :, None, :]
    sin_b = sin.astype(jnp.float32)[None, :, None, :]
    rotated = jnp.stack([even * cos_b - odd * sin_b, even * sin_b + odd * cos_b], axis=-1)
    return rotated.reshape(x.shape).astype(out_dtype)
