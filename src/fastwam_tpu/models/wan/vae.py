"""Wan2.2 VAE 3.8 (z_dim=48). Frozen; used offline to materialize latents.

Ported from
    codebases/TinyWAM-wan22/tinywam/models/wan/vae.py   <- the starting point:
        already measured at 9.9e-6 against torch on decode, and the streaming-encode
        mismatch that once cost us a 0% closed loop is already fixed there.
    reference/maxdiffusion/src/maxdiffusion/models/wan/autoencoder_kl_wan_2p2.py
        (1438 lines, structurally complete) — cross-reference only. It has no loader,
        no test and no caller upstream, i.e. it is unvalidated code (report_maxdiffusion.md §8.1).
    Semantics from reference/FastWAM/src/fastwam/models/wan22/wan_video_vae.py
        class WanVideoVAE38

Differences from the TinyWAM source, all deliberate:
    * Wan2.1 (``is_residual=False``, z_dim=16) is stripped. Every class here is the
      Wan2.2 path only; the 2.1 branches that used to sit alongside are noted where
      they were removed so the diff against TinyWAM stays readable.
    * ``nnx.List`` replaces bare python lists of submodules (flax >= 0.12 rejects
      those). The pure-dict key layout is unchanged, so old checkpoints still map 1:1
      — except for python key *type*, see ``tree_keys.stringify_tree_keys``.
    * The architecture arrives as a frozen :class:`Wan22VaeConfig` read from the
      checkpoint's ``vae_config.json`` sidecar, and every field is checked against the
      known-good Wan2.2 geometry instead of silently defaulting.
    * encode/decode work in the *normalized* latent space by default (what the DiT
      sees), matching ``WanVideoVAE38.encode(x, scale)``. Pass ``normalize=False``
      for the raw posterior mean.

Geometry (spec §2.7)
    z_dim 48, dim 160, dec_dim 256, dim_mult [1,2,4,4], num_res_blocks 2,
    temperal_downsample [False, True, True]; 16x spatial (2 patchify + 8 conv),
    4x temporal. [B,3,T,H,W] -> [B,48,(T-1)//4+1,H/16,W/16].
    RoboTwin: [B,3,9,384,320] -> [B,48,3,24,20]; single observation -> [B,48,1,24,20].

Conventions that must not drift
    * The 48+48 latent mean/std constants (wan_video_vae.py:1360-1378) are hardcoded
      below and must be copied verbatim: mu = (mu - mean) * (1/std), inverted on decode.
    * encode is a causal streaming loop: chunk 0 is frame 0 alone, then chunks of 4,
      each carrying the last 2 temporal slices through every CausalConv3d. Getting
      the chunk boundaries wrong produces plausible-looking latents and a dead policy.
    * encode returns mu; log_var is discarded (deterministic).
    * The reference iterates the batch one sample at a time; we batch, which is a
      throughput-only change (every op is per-sample) and is covered by test_vae.py.
"""

import json
import math
import os
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from flax import nnx

from .tree_keys import restore_tree_key_types, stringify_tree_keys

__all__ = [
    "LATENT_MEAN",
    "LATENT_STD",
    "WAN22_VAE_GEOMETRY",
    "VaeError",
    "Wan22VaeConfig",
    "WanVAE38",
    "build_vae_from_sidecar",
    "config_from_sidecar",
    "load_vae",
    "read_vae_sidecar",
]

# Copied verbatim from reference/FastWAM/.../wan_video_vae.py:1360-1378 (class
# WanVideoVAE38.__init__). diffusers' Wan2.2 config carries the same 96 numbers, and
# the converted checkpoint repeats them in its sidecar -- load_vae cross-checks all
# three agree rather than trusting any one of them.
LATENT_MEAN: tuple[float, ...] = (
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
)
LATENT_STD: tuple[float, ...] = (
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
)

CACHE_T = 2  # diffusers caches the last 2 temporal frames per causal conv


class VaeError(ValueError):
    """Any VAE config/checkpoint problem. Always raised, never logged-and-continued."""


# --------------------------------------------------------------------------------------
# config: the checkpoint self-describes, we verify
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Wan22VaeConfig:
    """Architecture of the Wan2.2 VAE 3.8, as written by the converter's sidecar.

    ``temperal_downsample`` keeps upstream's typo because that is the sidecar key and
    the diffusers attribute name; renaming it here would break every existing
    checkpoint's ``vae_config.json``.
    """

    base_dim: int = 160
    decoder_base_dim: int = 256
    z_dim: int = 48
    dim_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    temperal_downsample: tuple[bool, ...] = (False, True, True)
    is_residual: bool = True
    in_channels: int = 12  # 3 * patch_size**2, i.e. pixels AFTER patchify
    out_channels: int = 12
    patch_size: int = 2
    # HF faithfulness only: the Wan2.2 diffusers config says clip_output=false, but
    # diffusers 0.35.1 clamps regardless. decode() follows the code, not the flag
    # (see the note there). Carried so the sidecar round-trips.
    clip_output: bool = False
    latents_mean: tuple[float, ...] = field(default=LATENT_MEAN)
    latents_std: tuple[float, ...] = field(default=LATENT_STD)

    def __post_init__(self) -> None:
        if not self.is_residual:
            raise VaeError("only the Wan2.2 (is_residual=True) family is ported; got is_residual=False")
        if len(self.latents_mean) != self.z_dim or len(self.latents_std) != self.z_dim:
            raise VaeError(
                f"latent stats must have z_dim={self.z_dim} entries, got "
                f"{len(self.latents_mean)}/{len(self.latents_std)}"
            )
        if self.in_channels != 3 * self.patch_size**2:
            raise VaeError(
                f"in_channels ({self.in_channels}) must be 3*patch_size^2 "
                f"({3 * self.patch_size**2}) -- patchify runs before the encoder"
            )
        if self.out_channels != self.in_channels:
            raise VaeError(f"out_channels ({self.out_channels}) must equal in_channels ({self.in_channels})")
        if len(self.temperal_downsample) != len(self.dim_mult) - 1:
            raise VaeError(
                f"temperal_downsample ({len(self.temperal_downsample)}) must have one entry per "
                f"downsampling stage (len(dim_mult)-1 = {len(self.dim_mult) - 1})"
            )

    @property
    def temporal_downsample_factor(self) -> int:
        return 2 ** sum(self.temperal_downsample)

    @property
    def spatial_downsample_factor(self) -> int:
        """Strided-conv stack (2 ** number of stages) times the pixel patchify."""
        return 2 ** len(self.temperal_downsample) * self.patch_size

    @property
    def temporal_upsample(self) -> tuple[bool, ...]:
        return tuple(reversed(self.temperal_downsample))


#: The one geometry this file implements. Anything else in a sidecar is a mismatch.
WAN22_VAE_GEOMETRY = Wan22VaeConfig()

_SIDECAR_FIELDS = (
    "base_dim",
    "decoder_base_dim",
    "z_dim",
    "dim_mult",
    "num_res_blocks",
    "temperal_downsample",
    "is_residual",
    "in_channels",
    "out_channels",
    "patch_size",
    "clip_output",
    "latents_mean",
    "latents_std",
)


def config_from_sidecar(sidecar: dict) -> Wan22VaeConfig:
    """Sidecar mapping -> validated config. Unknown key / missing key / wrong value raises.

    Unlike TinyWAM's ``build_vae_from_config``, an absent field is NOT quietly defaulted:
    a checkpoint that does not say what it is cannot be trusted to be a Wan2.2 VAE.
    """
    if not isinstance(sidecar, dict):
        raise VaeError(f"vae_config.json must be a mapping, got {type(sidecar).__name__}")
    unknown = sorted(set(sidecar) - set(_SIDECAR_FIELDS))
    if unknown:
        raise VaeError(f"vae_config.json has unknown key(s) {unknown}; known: {sorted(_SIDECAR_FIELDS)}")
    missing = sorted(set(_SIDECAR_FIELDS) - set(sidecar))
    if missing:
        raise VaeError(
            f"vae_config.json is missing {missing}. A sidecar-less or partial checkpoint is "
            "rejected on purpose: the sidecar is the VAE's identity."
        )

    cfg = Wan22VaeConfig(
        base_dim=int(sidecar["base_dim"]),
        decoder_base_dim=int(sidecar["decoder_base_dim"]),
        z_dim=int(sidecar["z_dim"]),
        dim_mult=tuple(int(v) for v in sidecar["dim_mult"]),
        num_res_blocks=int(sidecar["num_res_blocks"]),
        temperal_downsample=tuple(bool(v) for v in sidecar["temperal_downsample"]),
        is_residual=bool(sidecar["is_residual"]),
        in_channels=int(sidecar["in_channels"]),
        out_channels=int(sidecar["out_channels"]),
        patch_size=int(sidecar["patch_size"]),
        clip_output=bool(sidecar["clip_output"]),
        latents_mean=tuple(float(v) for v in sidecar["latents_mean"]),
        latents_std=tuple(float(v) for v in sidecar["latents_std"]),
    )

    # The latent stats are a hardcoded constant upstream, not a tunable. A checkpoint
    # that disagrees would silently shift the whole latent space the DiT was trained in.
    for name, ours, theirs in (
        ("latents_mean", LATENT_MEAN, cfg.latents_mean),
        ("latents_std", LATENT_STD, cfg.latents_std),
    ):
        bad = [(i, a, b) for i, (a, b) in enumerate(zip(ours, theirs, strict=True)) if abs(a - b) > 1e-6]
        if bad:
            raise VaeError(
                f"vae_config.json {name} disagrees with the constants hardcoded in "
                f"WanVideoVAE38 at index(es) {[i for i, _, _ in bad[:5]]}: {bad[:5]}"
            )
    return cfg


# --------------------------------------------------------------------------------------
# streaming cache
# --------------------------------------------------------------------------------------
class _Cache:
    """Mutable per-call feature cache, indexed like diffusers' (feat_map, feat_idx).

    ``enabled``: when False, convs run in plain whole-sequence mode (cache ignored).
    ``next()`` returns the running conv index and advances it (mirrors feat_idx[0]).
    Purely a trace-time python object: every entry is a concrete jax array of static
    shape, so the whole streaming loop unrolls cleanly under jit.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.store: dict = {}
        self.idx = 0

    def reset_idx(self):
        self.idx = 0

    def next(self) -> int:
        i = self.idx
        self.idx += 1
        return i


class WanCausalConv3d(nnx.Module):
    """Conv3d with causal temporal padding (pad only *before* on the time axis).

    Param tree: ``.conv.kernel`` / ``.conv.bias`` (inner nnx.Conv), matching the
    diffusers VAE weight names (``*.conv.weight`` / ``*.conv.bias`` after rename).
    Supports a streaming ``cache`` arg that prepends the previous chunk's tail
    frames and reduces the causal pad accordingly (exactly like diffusers).
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        use_bias: bool = True,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.kernel_size = _canon3(kernel_size)
        self.stride = _canon3(stride)
        self.pad = _canon3(padding)
        self.time_causal_pad = 2 * self.pad[0]  # total causal frames padded before
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=self.kernel_size,
            strides=self.stride,
            use_bias=use_bias,
            padding="VALID",
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
        )

    def _pad_and_conv(self, x, time_pad_before):
        pad = (
            (0, 0),
            (time_pad_before, 0),
            (self.pad[1], self.pad[1]),
            (self.pad[2], self.pad[2]),
            (0, 0),
        )
        if any(a or b for (a, b) in pad):
            x = jnp.pad(x, pad, mode="constant", constant_values=0.0)
        return self.conv(x)

    def __call__(self, x: jax.Array, cache: "_Cache | None" = None) -> jax.Array:
        # x: [B, T, H, W, C] (channels-last)
        if cache is None or not cache.enabled:
            return self._pad_and_conv(x, self.time_causal_pad)

        idx = cache.next()
        if self.time_causal_pad == 0:
            # No temporal causal padding (e.g. 1x1x1 / spatial-only) -> cache is a no-op.
            return self._pad_and_conv(x, 0)

        cache_x = x[:, -CACHE_T:, :, :, :]
        prev = cache.store.get(idx, None)
        if cache_x.shape[1] < 2 and prev is not None:
            # prepend the last frame of the previous chunk (diffusers cache-of-2 logic)
            cache_x = jnp.concatenate([prev[:, -1:, :, :, :], cache_x], axis=1)
        if prev is not None:
            x = jnp.concatenate([prev, x], axis=1)
            out = self._pad_and_conv(x, self.time_causal_pad - prev.shape[1])
        else:
            out = self._pad_and_conv(x, self.time_causal_pad)
        cache.store[idx] = cache_x
        return out


def _canon3(x: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (x, x, x) if isinstance(x, int) else tuple(x)


class WanRMS_norm(nnx.Module):
    """Wan RMS norm with ``scale = dim**0.5`` and learnable ``gamma`` (channels-last)."""

    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        eps: float = 1e-12,
        use_bias: bool = False,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.eps = eps
        self.scale = dim**0.5
        self.dtype = dtype
        self.gamma = nnx.Param(jnp.ones((dim,), dtype=weights_dtype))
        self.bias = nnx.Param(jnp.zeros((dim,), dtype=weights_dtype)) if use_bias else None

    def __call__(self, x: jax.Array) -> jax.Array:
        # diffusers: F.normalize(x, dim=channel) * scale * gamma + bias
        # F.normalize -> x / max(||x||_2, eps) over the channel axis (here last).
        in_dtype = x.dtype
        x = x.astype(jnp.float32)
        norm = jnp.linalg.norm(x, ord=2, axis=-1, keepdims=True)
        x = x / jnp.maximum(norm, self.eps)
        # `param[...]` is the flax >= 0.12 spelling; TinyWAM's `.value` now warns.
        x = x * self.scale * self.gamma[...].astype(jnp.float32)
        if self.bias is not None:
            x = x + self.bias[...].astype(jnp.float32)
        return x.astype(in_dtype)


class Identity(nnx.Module):
    def __call__(self, x: jax.Array, *args, **kwargs) -> jax.Array:
        return x


class WanUpsample(nnx.Module):
    """Nearest-exact spatial upsample on channels-last [N, H, W, C]."""

    def __init__(self, scale_factor: tuple[float, float]):
        self.scale_factor = scale_factor

    def __call__(self, x: jax.Array) -> jax.Array:
        in_dtype = x.dtype
        n, h, w, c = x.shape
        th, tw = int(h * self.scale_factor[0]), int(w * self.scale_factor[1])
        # torch "nearest-exact" == jax.image.resize method="nearest" for integer scales.
        out = jax.image.resize(x.astype(jnp.float32), (n, th, tw, c), method="nearest")
        return out.astype(in_dtype)


class WanZeroPad2d(nnx.Module):
    """torch nn.ZeroPad2d on channels-last [N, H, W, C]; padding = ((top,bottom),(left,right))."""

    def __init__(self, padding: tuple[tuple[int, int], tuple[int, int]]):
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        (t, b), (left, r) = self.padding
        return jnp.pad(x, ((0, 0), (t, b), (left, r), (0, 0)), mode="constant", constant_values=0.0)


class WanResample(nnx.Module):
    """Up/downsample used by both residual blocks.

    upsample2d: nearest 2x (H,W) then a 3x3 conv (dim -> upsample_out_dim).
    upsample3d: causal ``time_conv`` that doubles temporal resolution (with the
                first-frame "Rep" handling) then the same spatial path.
    downsample2d: zero-pad (0,1,0,1) then a stride-2 3x3 conv (dim -> dim).
    downsample3d: same spatial path plus a temporal stride-2 ``time_conv`` that is
                skipped on the first chunk of a stream (frame 0 stays uncompressed).
    Param names: ``.resample.layers.1.{kernel,bias}`` (nnx.Sequential conv),
    ``.time_conv.conv.{kernel,bias}`` for the temporal conv.

    (Wan2.1's ``upsample_out_dim = dim // 2`` default is gone: the 2.2 residual up
    block always passes the stage's out_dim explicitly.)
    """

    def __init__(
        self,
        dim: int,
        mode: str,
        rngs: nnx.Rngs,
        upsample_out_dim: int | None = None,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.dim = dim
        self.mode = mode
        self.dtype = dtype
        # time_conv is built into a local and assigned exactly once: nnx >= 0.12 fixes an
        # attribute's static/data status at its first assignment, so `= None` up front
        # then `= Module` later is rejected.
        time_conv = None

        if mode in ("upsample2d", "upsample3d"):
            if upsample_out_dim is None:
                raise VaeError("upsample_out_dim is required (Wan2.2 never uses the 2.1 dim//2 default)")
            self.resample = nnx.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0)),
                nnx.Conv(
                    dim,
                    upsample_out_dim,
                    kernel_size=(3, 3),
                    padding="SAME",
                    use_bias=True,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=weights_dtype,
                ),
            )
            if mode == "upsample3d":
                time_conv = WanCausalConv3d(
                    rngs=rngs,
                    in_channels=dim,
                    out_channels=dim * 2,
                    kernel_size=(3, 1, 1),
                    padding=(1, 0, 0),
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                )
        elif mode in ("downsample2d", "downsample3d"):
            # diffusers: nn.Sequential(nn.ZeroPad2d((0,1,0,1)), nn.Conv2d(dim, dim, 3, stride=2))
            # -> conv params live at resample index 1, like the upsample path.
            self.resample = nnx.Sequential(
                WanZeroPad2d(((0, 1), (0, 1))),
                nnx.Conv(
                    dim,
                    dim,
                    kernel_size=(3, 3),
                    strides=(2, 2),
                    padding="VALID",
                    use_bias=True,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=weights_dtype,
                ),
            )
            if mode == "downsample3d":
                time_conv = WanCausalConv3d(
                    rngs=rngs,
                    in_channels=dim,
                    out_channels=dim,
                    kernel_size=(3, 1, 1),
                    stride=(2, 1, 1),
                    padding=(0, 0, 0),
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                )
        else:
            raise VaeError(f"unknown resample mode {mode!r}")
        self.time_conv = time_conv

    def __call__(self, x: jax.Array, cache: "_Cache | None" = None) -> jax.Array:
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape
        x = x.astype(self.dtype)

        if self.mode == "downsample3d" and cache is None:
            # Without the streaming cache the temporal stride-2 conv has no defined
            # first-chunk semantics -> only the chunked encode() path may come here.
            raise VaeError("downsample3d needs the streaming cache (use the chunked encode path)")

        if self.mode == "upsample3d" and self.time_conv is not None:
            if cache is not None and cache.enabled:
                idx = cache.next()
                prev = cache.store.get(idx, None)
                if prev is None:
                    # First chunk: mark "Rep", SKIP time_conv (no temporal doubling).
                    cache.store[idx] = "Rep"
                else:
                    cache_x = x[:, -CACHE_T:, :, :, :]
                    if cache_x.shape[1] < 2 and prev is not None and not isinstance(prev, str):
                        cache_x = jnp.concatenate([prev[:, -1:, :, :, :], cache_x], axis=1)
                    if cache_x.shape[1] < 2 and isinstance(prev, str) and prev == "Rep":
                        cache_x = jnp.concatenate([jnp.zeros_like(cache_x), cache_x], axis=1)
                    if isinstance(prev, str) and prev == "Rep":
                        xt = self.time_conv(x)  # plain causal conv (cache=None)
                    else:
                        # feed prev as the time-conv cache via a tiny shim
                        xt = self._time_conv_with_prev(x, prev)
                    cache.store[idx] = cache_x
                    # double the temporal axis: [B, T, H, W, 2C] -> [B, 2T, H, W, C]
                    xt = xt.reshape(b, t, h, w, 2, c)
                    xt = jnp.stack([xt[:, :, :, :, 0, :], xt[:, :, :, :, 1, :]], axis=2)
                    x = xt.reshape(b, t * 2, h, w, c)
                    t = t * 2
            else:
                # non-cached whole-sequence (NOT frame-count faithful; cache path is canonical)
                x = self.time_conv(x)
                x = x.reshape(b, t, h, w, 2, c)
                x = x.transpose(0, 1, 4, 2, 3, 5).reshape(b, t * 2, h, w, c)
                t = t * 2

        x = x.reshape(b * t, h, w, c)
        x = self.resample(x)
        hn, wn, cn = x.shape[1:]
        x = x.reshape(b, t, hn, wn, cn)

        if self.mode == "downsample3d" and cache is not None and cache.enabled:
            # Temporal stride-2 conv across chunks (diffusers encoder semantics):
            # first chunk stores its (spatially resampled) frames and SKIPS the conv
            # (frame 0 stays uncompressed); later chunks prepend the previous chunk's
            # last frame, so T frames -> T//2 (no causal padding, stride 2).
            idx = cache.next()
            prev = cache.store.get(idx, None)
            if prev is None:
                cache.store[idx] = x
            else:
                cache.store[idx] = x[:, -1:]
                x = self.time_conv._pad_and_conv(jnp.concatenate([prev[:, -1:], x], axis=1), 0)
        return x

    def _time_conv_with_prev(self, x, prev):
        """Run time_conv prepending `prev` frames and reducing the causal pad (cache=2)."""
        tc = self.time_conv
        xx = jnp.concatenate([prev, x], axis=1)
        return tc._pad_and_conv(xx, tc.time_causal_pad - prev.shape[1])


class WanResidualBlock(nnx.Module):
    """norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 (+ optional conv_shortcut)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.nonlinearity = jax.nn.silu
        self.norm1 = WanRMS_norm(in_dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype)
        self.conv1 = WanCausalConv3d(
            rngs=rngs,
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        self.norm2 = WanRMS_norm(out_dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype)
        self.conv2 = WanCausalConv3d(
            rngs=rngs,
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        self.conv_shortcut = (
            WanCausalConv3d(
                rngs=rngs,
                in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=1,
                dtype=dtype,
                weights_dtype=weights_dtype,
            )
            if in_dim != out_dim
            else Identity()
        )

    def __call__(self, x: jax.Array, cache: "_Cache | None" = None) -> jax.Array:
        h = self.conv_shortcut(x)  # 1x1x1 (no temporal pad) -> cache no-op
        x = self.conv1(self.nonlinearity(self.norm1(x)), cache)
        x = self.conv2(self.nonlinearity(self.norm2(x)), cache)
        return x + h


class WanAttentionBlock(nnx.Module):
    """Spatial self-attention per frame (1x1 qkv conv), ``to_qkv`` / ``proj`` names."""

    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.dim = dim
        self.norm = WanRMS_norm(dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype)
        self.to_qkv = nnx.Conv(
            dim,
            dim * 3,
            kernel_size=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
        )
        self.proj = nnx.Conv(
            dim,
            dim,
            kernel_size=(1, 1),
            rngs=rngs,
            dtype=dtype,
            param_dtype=weights_dtype,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: [B, T, H, W, C]
        identity = x
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c)
        x = self.norm(x)
        qkv = self.to_qkv(x)  # [B*T, H, W, 3C]
        # Match diffusers exactly: channels-first qkv [B*T,3C,H,W] -> [B*T,1,3C,HW]
        # -> permute to [B*T,1,HW,3C] -> chunk along last axis into q,k,v (each C).
        qkv = jnp.transpose(qkv, (0, 3, 1, 2))  # [B*T, 3C, H, W]
        qkv = qkv.reshape(b * t, 1, c * 3, h * w)  # [B*T, 1, 3C, HW]
        qkv = jnp.transpose(qkv, (0, 1, 3, 2))  # [B*T, 1, HW, 3C]
        q, k, v = jnp.split(qkv, 3, axis=-1)  # each [B*T, 1, HW, C]
        # single-head scaled dot-product attention over the HW sequence (torch SDPA:
        # scale = 1/sqrt(head_dim=C)). jax.nn.dot_product_attention expects a different
        # axis layout, so compute it explicitly to stay faithful.
        scale = 1.0 / math.sqrt(c)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v.astype(jnp.float32))  # [B*T,1,HW,C]
        out = out.astype(x.dtype)
        out = jnp.squeeze(out, 1).reshape(b * t, h, w, c)  # channels-last [B*T,H,W,C]
        out = self.proj(out)
        out = out.reshape(b, t, h, w, c)
        return out + identity


class WanMidBlock(nnx.Module):
    """resnets[0] -> (attention -> resnet) x num_layers."""

    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        num_layers: int = 1,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.dim = dim
        resnets = [WanResidualBlock(dim, dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype))
            resnets.append(WanResidualBlock(dim, dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype))
        self.attentions = nnx.List(attentions)
        self.resnets = nnx.List(resnets)

    def __call__(self, x: jax.Array, cache: "_Cache | None" = None) -> jax.Array:
        x = self.resnets[0](x, cache)
        for attn, resnet in zip(self.attentions, self.resnets[1:], strict=True):
            x = attn(x)
            x = resnet(x, cache)
        return x


# --------------------------------------------------------------------------------------
# Wan2.2 specifics: pixel-(un)patchify + residual up/down blocks.
# The Wan2.2 VAE (TI2V-5B) reaches 16x spatial compression via an extra 2x pixel
# unshuffle (patchify) around an otherwise 8x conv stack, uses residual down/up blocks
# with parameter-free AvgDown3D / DupUp3D shortcuts, and a separate decoder_base_dim.
# --------------------------------------------------------------------------------------
def patchify(x: jax.Array, patch_size: int) -> jax.Array:
    """Pixel-unshuffle on channels-first [B, C, T, H, W] (matches diffusers.patchify).

    [B, C, T, H, W] -> [B, C*p*p, T, H//p, W//p]. p=1 is a no-op.
    """
    if patch_size == 1:
        return x
    b, c, t, h, w = x.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise VaeError(f"H({h}) and W({w}) must be divisible by patch_size({patch_size})")
    x = x.reshape(b, c, t, h // patch_size, patch_size, w // patch_size, patch_size)
    x = jnp.transpose(x, (0, 1, 6, 4, 2, 3, 5))
    x = x.reshape(b, c * patch_size * patch_size, t, h // patch_size, w // patch_size)
    return x


def unpatchify(x: jax.Array, patch_size: int) -> jax.Array:
    """Inverse of ``patchify`` on channels-first [B, C*p*p, T, H, W] (diffusers.unpatchify)."""
    if patch_size == 1:
        return x
    b, cpp, t, h, w = x.shape
    c = cpp // (patch_size * patch_size)
    x = x.reshape(b, c, patch_size, patch_size, t, h, w)
    x = jnp.transpose(x, (0, 1, 4, 5, 3, 6, 2))
    x = x.reshape(b, c, t, h * patch_size, w * patch_size)
    return x


class AvgDown3D(nnx.Module):
    """Parameter-free average-pool downsample shortcut (Wan2.2 residual down block).

    Channels-last I/O [B, T, H, W, C]; internally transposes to diffusers' channels-first
    layout to mirror the exact reshape/permute/mean. Front (causal) zero-pad on time.
    Holds NO weights, so it produces no checkpoint keys (weight-map compatible).
    """

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if in_channels * self.factor % out_channels != 0:
            raise VaeError(f"AvgDown3D: {in_channels}*{self.factor} not divisible by {out_channels}")
        self.group_size = in_channels * self.factor // out_channels

    def __call__(self, x: jax.Array) -> jax.Array:
        ft, fs = self.factor_t, self.factor_s
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # [B,T,H,W,C] -> [B,C,T,H,W]
        pad_t = (ft - x.shape[2] % ft) % ft
        if pad_t:
            x = jnp.pad(x, ((0, 0), (0, 0), (pad_t, 0), (0, 0), (0, 0)))
        b, c, t, h, w = x.shape
        x = x.reshape(b, c, t // ft, ft, h // fs, fs, w // fs, fs)
        x = jnp.transpose(x, (0, 1, 3, 5, 7, 2, 4, 6))
        x = x.reshape(b, c * self.factor, t // ft, h // fs, w // fs)
        x = x.reshape(b, self.out_channels, self.group_size, t // ft, h // fs, w // fs)
        x = jnp.mean(x, axis=2)
        return jnp.transpose(x, (0, 2, 3, 4, 1))  # -> [B,T',H',W',C_out]


class DupUp3D(nnx.Module):
    """Parameter-free duplicating upsample shortcut (Wan2.2 residual up block).

    Channels-last I/O; ``first_chunk`` drops the leading ``factor_t-1`` temporal frames
    so the streamed first latent frame is not temporally upsampled (matches diffusers).
    Holds NO weights.
    """

    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor_t * factor_s * factor_s
        if out_channels * self.factor % in_channels != 0:
            raise VaeError(f"DupUp3D: {out_channels}*{self.factor} not divisible by {in_channels}")
        self.repeats = out_channels * self.factor // in_channels

    def __call__(self, x: jax.Array, first_chunk: bool = False) -> jax.Array:
        ft, fs, oc = self.factor_t, self.factor_s, self.out_channels
        x = jnp.transpose(x, (0, 4, 1, 2, 3))  # [B,T,H,W,C] -> [B,C,T,H,W]
        x = jnp.repeat(x, self.repeats, axis=1)
        b, _, t, h, w = x.shape
        x = x.reshape(b, oc, ft, fs, fs, t, h, w)
        x = jnp.transpose(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = x.reshape(b, oc, t * ft, h * fs, w * fs)
        if first_chunk:
            x = x[:, :, ft - 1 :, :, :]
        return jnp.transpose(x, (0, 2, 3, 4, 1))  # -> [B,T',H',W',C_out]


class WanResidualDownBlock(nnx.Module):
    """Wan2.2 encoder stage: num_res_blocks residuals (+optional downsampler) + AvgDown3D
    shortcut. Submodule names (``resnets`` / ``downsampler`` / ``avg_shortcut``) match
    diffusers so pretrained keys map 1:1 (avg_shortcut is param-free -> no keys)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temperal_downsample: bool,
        down_flag: bool,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.avg_shortcut = AvgDown3D(
            in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1
        )
        resnets = []
        cur = in_dim
        for _ in range(num_res_blocks):
            resnets.append(
                WanResidualBlock(cur, out_dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype)
            )
            cur = out_dim
        self.resnets = nnx.List(resnets)
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(
                out_dim, mode=mode, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype
            )
        else:
            self.downsampler = None

    def __call__(self, x: jax.Array, cache: "_Cache | None" = None) -> jax.Array:
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x, cache)
        if self.downsampler is not None:
            x = self.downsampler(x, cache)
        return x + self.avg_shortcut(x_copy)


class WanResidualUpBlock(nnx.Module):
    """Wan2.2 decoder stage: (num_res_blocks+1) residuals (+optional upsampler) + DupUp3D
    shortcut. Submodule names match diffusers (``resnets`` / ``upsampler`` / ``avg_shortcut``)."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        temperal_upsample: bool,
        up_flag: bool,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.avg_shortcut = (
            DupUp3D(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2) if up_flag else None
        )
        resnets = []
        cur = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(cur, out_dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype)
            )
            cur = out_dim
        self.resnets = nnx.List(resnets)
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(
                out_dim,
                mode=mode,
                rngs=rngs,
                upsample_out_dim=out_dim,
                dtype=dtype,
                weights_dtype=weights_dtype,
            )
        else:
            self.upsampler = None

    def __call__(self, x: jax.Array, cache: "_Cache | None" = None, first_chunk: bool = False) -> jax.Array:
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x, cache)
        if self.upsampler is not None:
            x = self.upsampler(x, cache)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(x_copy, first_chunk=first_chunk)
        return x


class WanEncoder3d(nnx.Module):
    """Wan2.2 VAE encoder (patchified pixels -> pre-quant features).

    conv_in -> one WanResidualDownBlock per dim_mult stage -> mid_block -> norm/silu ->
    conv_out. ``z_dim`` here is the PARENT's 2*z_dim (mean + logvar channels).
    Streaming-causal: the caller feeds chunks (frame 0 alone, then 4 pixel frames per
    chunk -> 1 latent frame each) with a shared ``_Cache``, exactly like diffusers'
    feat_cache encode.

    (Wan2.1's flat list of WanResidualBlock + WanResample is not ported.)
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        z_dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        temperal_downsample: tuple[bool, ...],
        in_channels: int,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.nonlinearity = jax.nn.silu
        dims = [dim * u for u in [1, *dim_mult]]

        self.conv_in = WanCausalConv3d(
            rngs=rngs,
            in_channels=in_channels,
            out_channels=dims[0],
            kernel_size=3,
            padding=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )

        # down_blocks indices match diffusers' encoder.down_blocks.{i}.
        # (attn_scales is [] for every released Wan VAE -> no attention blocks here.)
        down_blocks = []
        out_dim = dims[0]
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            last = i == len(dim_mult) - 1
            down_blocks.append(
                WanResidualDownBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temperal_downsample=temperal_downsample[i] if not last else False,
                    down_flag=not last,
                    rngs=rngs,
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                )
            )
        self.down_blocks = nnx.List(down_blocks)

        self.mid_block = WanMidBlock(
            out_dim, rngs=rngs, num_layers=1, dtype=dtype, weights_dtype=weights_dtype
        )
        self.norm_out = WanRMS_norm(out_dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype)
        self.conv_out = WanCausalConv3d(
            rngs=rngs,
            in_channels=out_dim,
            out_channels=z_dim,
            kernel_size=3,
            padding=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )

    def __call__(self, x: jax.Array, cache: "_Cache | None" = None) -> jax.Array:
        x = self.conv_in(x, cache)
        for block in self.down_blocks:
            x = block(x, cache)
        x = self.mid_block(x, cache)
        x = self.conv_out(self.nonlinearity(self.norm_out(x)), cache)
        return x


class WanDecoder3d(nnx.Module):
    """conv_in -> mid_block -> up_blocks -> norm_out -> silu -> conv_out.

    (Wan2.1's WanUpBlock, and its "in_dim halves after the first stage" rule, are not
    ported: the 2.2 residual decoder keeps in_dim as the stage dim.)
    """

    def __init__(
        self,
        rngs: nnx.Rngs,
        dim: int,
        z_dim: int,
        dim_mult: tuple[int, ...],
        num_res_blocks: int,
        temperal_upsample: tuple[bool, ...],
        out_channels: int,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.nonlinearity = jax.nn.silu

        dims = [dim * u for u in [dim_mult[-1]] + list(dim_mult[::-1])]

        self.conv_in = WanCausalConv3d(
            rngs=rngs,
            in_channels=z_dim,
            out_channels=dims[0],
            kernel_size=3,
            padding=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        self.mid_block = WanMidBlock(
            dims[0], rngs=rngs, num_layers=1, dtype=dtype, weights_dtype=weights_dtype
        )

        up_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            up_flag = i != len(dim_mult) - 1
            up_blocks.append(
                WanResidualUpBlock(
                    in_dim,
                    out_dim,
                    num_res_blocks,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    rngs=rngs,
                    dtype=dtype,
                    weights_dtype=weights_dtype,
                )
            )
        self.up_blocks = nnx.List(up_blocks)

        self.norm_out = WanRMS_norm(out_dim, rngs=rngs, dtype=dtype, weights_dtype=weights_dtype)
        self.conv_out = WanCausalConv3d(
            rngs=rngs,
            in_channels=out_dim,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )

    def __call__(self, x: jax.Array, cache: "_Cache | None" = None, first_chunk: bool = False) -> jax.Array:
        x = self.conv_in(x, cache)
        x = self.mid_block(x, cache)
        for up_block in self.up_blocks:
            x = up_block(x, cache, first_chunk=first_chunk)
        x = self.conv_out(self.nonlinearity(self.norm_out(x)), cache)
        return x


# --------------------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------------------
class WanVAE38(nnx.Module):
    """Wan2.2 VAE 3.8 (z_dim=48), encode + decode, cache-faithful.

    Latent-space convention (spec §2.7, wan_video_vae.py:1003/1015): every public
    entry point works in the *normalized* space the DiT is trained in --
    ``(mu - mean) / std`` on the way in, ``z * std + mean`` on the way out. Pass
    ``normalize=False`` / ``denormalize=False`` to stay in the raw posterior-mean
    space (that is what a parity check against diffusers wants).

    Entry points
        encode(video)              whole clip, T = 1 + 4m           -> [B,48,1+m,h,w]
        encode_frame(frame)        one observation (the M3 hot path) -> [B,48,1,h,w]
        encode_step(frames, state) streaming causal, one unit at a time (M4/M3)
        decode(latents)            latents -> pixels (debug / eval videos)
    """

    def __init__(
        self,
        config: Wan22VaeConfig,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        weights_dtype: jnp.dtype = jnp.float32,
    ):
        self.config = config
        self.z_dim = config.z_dim
        self.patch_size = config.patch_size
        self.temporal_downsample_factor = config.temporal_downsample_factor
        self.spatial_downsample_factor = config.spatial_downsample_factor

        self.encoder = WanEncoder3d(
            rngs=rngs,
            dim=config.base_dim,
            z_dim=config.z_dim * 2,  # mean + logvar
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            temperal_downsample=config.temperal_downsample,
            in_channels=config.in_channels,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        # quant_conv: 2*z_dim -> 2*z_dim 1x1x1 causal conv, after the encoder.
        self.quant_conv = WanCausalConv3d(
            rngs=rngs,
            in_channels=config.z_dim * 2,
            out_channels=config.z_dim * 2,
            kernel_size=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        # post_quant_conv: z_dim -> z_dim 1x1x1 causal conv, before the decoder.
        self.post_quant_conv = WanCausalConv3d(
            rngs=rngs,
            in_channels=config.z_dim,
            out_channels=config.z_dim,
            kernel_size=1,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )
        self.decoder = WanDecoder3d(
            rngs=rngs,
            dim=config.decoder_base_dim,
            z_dim=config.z_dim,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            temperal_upsample=config.temporal_upsample,
            out_channels=config.out_channels,
            dtype=dtype,
            weights_dtype=weights_dtype,
        )

    # --- latent normalisation -------------------------------------------------------
    def normalize_latents(self, latents: jax.Array) -> jax.Array:
        """Raw posterior mean -> the DiT's space: ``(z - mean) * (1/std)``.

        latents: [B, z, F, H, W] channels-first.
        """
        mean, std = self._stats(latents.dtype)
        return (latents - mean) / std

    def denormalize_latents(self, latents: jax.Array) -> jax.Array:
        """Exact inverse of :meth:`normalize_latents`: ``z * std + mean``."""
        mean, std = self._stats(latents.dtype)
        return latents * std + mean

    def _stats(self, dtype) -> tuple[jax.Array, jax.Array]:
        shape = (1, self.z_dim, 1, 1, 1)
        mean = jnp.asarray(self.config.latents_mean, dtype=dtype).reshape(shape)
        std = jnp.asarray(self.config.latents_std, dtype=dtype).reshape(shape)
        return mean, std

    # --- encode ---------------------------------------------------------------------
    def encode(self, video: jax.Array, normalize: bool = True, channels_first: bool = True) -> jax.Array:
        """Streaming-causal deterministic encode: pixels -> posterior-MEAN latent.

        Faithful to diffusers' chunked feat_cache encode: frame 0 alone, then 4 pixel
        frames per chunk, each chunk conditioned on the cached features of the past --
        so latent frame i depends ONLY on pixel frames <= 1 + 4i.

        Args:
          video: pixels in [-1, 1] with T = 1 + 4m. channels_first=True (default):
                 [B, 3, T, H, W]; else [B, T, H, W, 3].
          normalize: return the DiT's normalized space (default) or the raw mean.
        Returns [B, 48, 1+m, H/16, W/16].
        """
        x = self._prepare_pixels(video, channels_first, where="encode")
        t = x.shape[2]
        step = self.temporal_downsample_factor
        if (t - 1) % step != 0:
            raise VaeError(f"encode expects T = 1 + {step}*m frames, got T={t}")

        x = self._patchify_to_channels_last(x)
        cache = _Cache(enabled=True)
        outs = []
        for start in [0, *range(1, t, step)]:
            cache.reset_idx()
            end = 1 if start == 0 else start + step
            outs.append(self.encoder(x[:, start:end], cache))  # each -> [B, 1, h, w, 2*z]
        return self._post_encode(jnp.concatenate(outs, axis=1), normalize)

    def encode_frame(
        self, frame: jax.Array, normalize: bool = True, channels_first: bool = True
    ) -> jax.Array:
        """Encode ONE observation -- the closed-loop inference hot path (M3).

        This is ``encode`` restricted to T=1, which is exactly chunk 0 of the streaming
        loop, so the result is bit-identical to the first latent frame of encoding a
        longer clip that starts with this frame.

        Args:
          frame: [B, 3, 1, H, W] or [B, 3, H, W] (channels_first=True); the
                 channels-last equivalents otherwise.
        Returns [B, 48, 1, H/16, W/16] -- RoboTwin: [B, 48, 1, 24, 20].
        """
        if frame.ndim == 4:
            axis = 2 if channels_first else 1
            frame = jnp.expand_dims(frame, axis=axis)
        if frame.ndim != 5:
            raise VaeError(f"encode_frame expects a 4- or 5-D array, got shape {frame.shape}")
        t = frame.shape[2] if channels_first else frame.shape[1]
        if t != 1:
            raise VaeError(f"encode_frame expects exactly one frame, got T={t}")
        return self.encode(frame, normalize=normalize, channels_first=channels_first)

    def encode_step(
        self,
        new_frames: jax.Array,
        cache_state: dict | None = None,
        normalize: bool = True,
        channels_first: bool = True,
    ) -> tuple[jax.Array, dict]:
        """Incremental streaming encode: one causal unit at a time, carrying the cache.

        T_step is 1 for the very first call (``cache_state is None``) and
        ``temporal_downsample_factor`` (4) afterwards. Sequential calls give latents
        IDENTICAL to :meth:`encode` over the same prefix at O(1) cost per step instead
        of O(T), so materializing an episode is O(T) rather than O(T^2).

        Returns ``(latent [B, 48, 1, h, w], new_cache_state)``. The returned state is a
        fresh dict; the caller threads it into the next call.
        """
        x = self._prepare_pixels(new_frames, channels_first, where="encode_step")
        first = cache_state is None
        want = 1 if first else self.temporal_downsample_factor
        if x.shape[2] != want:
            raise VaeError(
                f"encode_step expects T_step={want} on the "
                f"{'first' if first else 'subsequent'} call, got {x.shape[2]}"
            )

        x = self._patchify_to_channels_last(x)
        cache = _Cache(enabled=True)
        if cache_state is not None:
            cache.store = dict(cache_state)
        cache.reset_idx()
        out = self.encoder(x, cache)  # [B, 1, h, w, 2*z]
        return self._post_encode(out, normalize), cache.store

    def _prepare_pixels(self, x: jax.Array, channels_first: bool, where: str) -> jax.Array:
        """-> channels-first [B, 3, T, H, W], with the geometry checked."""
        if x.ndim != 5:
            raise VaeError(f"{where} expects a 5-D array, got shape {x.shape}")
        if not channels_first:
            x = jnp.transpose(x, (0, 4, 1, 2, 3))
        if x.shape[1] != 3:
            raise VaeError(f"{where} expects 3 pixel channels, got shape {x.shape} (channels-first)")
        f = self.spatial_downsample_factor
        h, w = x.shape[3], x.shape[4]
        if h % f or w % f:
            raise VaeError(f"{where}: H({h}) and W({w}) must both divide by {f}")
        return x

    def _patchify_to_channels_last(self, x: jax.Array) -> jax.Array:
        """Wan2.2 pixel-patchify (3 -> 3*p*p, spatial /p), then channels-last."""
        x = patchify(x, self.patch_size)
        return jnp.transpose(x, (0, 2, 3, 4, 1))

    def _post_encode(self, encoded: jax.Array, normalize: bool) -> jax.Array:
        """quant_conv, drop log_var, back to channels-first, optionally normalize."""
        enc = self.quant_conv(encoded)  # 1x1x1 -> no temporal interaction / cache
        mean = enc[..., : self.z_dim]  # posterior mean; log_var discarded
        out = jnp.transpose(mean, (0, 4, 1, 2, 3))  # -> [B, z, F, h, w]
        return self.normalize_latents(out) if normalize else out

    # --- decode ---------------------------------------------------------------------
    def decode(self, latents: jax.Array, denormalize: bool = True, channels_first: bool = True) -> jax.Array:
        """Latents -> pixels, cache-faithful streaming over latent frames.

        F latent frames yield ``1 + 4*(F-1)`` pixel frames: the first latent frame is
        NOT temporally upsampled. Decoding the whole temporal axis in one shot is wrong
        in both frame count and boundary values, which is why this loops.

        Args:
          latents: [B, 48, F, h, w] (channels_first=True) or [B, F, h, w, 48].
          denormalize: the input is in the DiT's normalized space (default).
        Returns [B, 3, 1+4*(F-1), h*16, w*16] in [-1, 1].
        """
        if latents.ndim != 5:
            raise VaeError(f"decode expects a 5-D array, got shape {latents.shape}")
        if not channels_first:
            latents = jnp.transpose(latents, (0, 4, 1, 2, 3))
        if latents.shape[1] != self.z_dim:
            raise VaeError(f"decode expects {self.z_dim} latent channels, got shape {latents.shape}")
        if denormalize:
            latents = self.denormalize_latents(latents)

        z = jnp.transpose(latents, (0, 2, 3, 4, 1))  # -> [B, F, h, w, C]
        num_frames = z.shape[1]
        # post_quant_conv is 1x1x1 (no temporal pad) -> whole-sequence is fine; cache no-op.
        x = self.post_quant_conv(z)

        cache = _Cache(enabled=True)
        outs = []
        for i in range(num_frames):
            cache.reset_idx()
            xi = x[:, i : i + 1, :, :, :]  # one latent frame [B,1,h,w,C]
            # first_chunk=True only on the first latent frame (DupUp3D temporal trim).
            outs.append(self.decoder(xi, cache, first_chunk=(i == 0)))
        out = jnp.concatenate(outs, axis=1)  # [B, F_pix, H', W', out_channels]
        out = jnp.transpose(out, (0, 4, 1, 2, 3))  # -> [B, out_channels, F_pix, H', W']
        out = unpatchify(out, self.patch_size)  # 3*p*p -> 3, spatial *p, BEFORE clamping
        # ALWAYS clamp, matching diffusers' BEHAVIOR rather than its config: the Wan2.2
        # HF config says clip_output=false, but diffusers 0.35.1 clamps regardless
        # (verified 2026-07-15 -- honoring the flag produced a 0.22 max|delta| purely in
        # the saturation zones). config.clip_output is sidecar provenance only.
        return jnp.clip(out, -1.0, 1.0)  # [B, 3, F_pix, H_pix, W_pix]

    # --- self-description -----------------------------------------------------------
    def latent_shape(self, video_frames: int, height: int, width: int) -> tuple[int, int, int, int]:
        """(C, T_l, H_l, W_l) this VAE produces for a clip of that geometry."""
        step = self.temporal_downsample_factor
        if (video_frames - 1) % step:
            raise VaeError(f"video_frames must be 1 + {step}*m, got {video_frames}")
        f = self.spatial_downsample_factor
        if height % f or width % f:
            raise VaeError(f"H({height}) and W({width}) must both divide by {f}")
        return (self.z_dim, (video_frames - 1) // step + 1, height // f, width // f)

    def check_geometry(self, shapes) -> None:
        """Raise unless this VAE produces exactly the latents ``Config.shapes`` promises.

        ``shapes`` is a :class:`fastwam_tpu.config.DerivedShapes`. Called by whoever
        builds the training/inference graph, so a VAE swapped under the config is a
        loud failure and not a silently reshaped tensor.
        """
        from fastwam_tpu.config import VAE_SPATIAL_FACTOR, VAE_TEMPORAL_FACTOR

        checks = (
            ("z_dim", self.z_dim, shapes.latent_channels),
            ("temporal factor", self.temporal_downsample_factor, VAE_TEMPORAL_FACTOR),
            ("spatial factor", self.spatial_downsample_factor, VAE_SPATIAL_FACTOR),
            (
                "latent frames",
                (shapes.video_frames - 1) // self.temporal_downsample_factor + 1,
                shapes.latent_frames,
            ),
        )
        bad = [f"{name}: vae={got} config={want}" for name, got, want in checks if got != want]
        if bad:
            raise VaeError("VAE geometry contradicts the config: " + "; ".join(bad))


# --------------------------------------------------------------------------------------
# Config-driven construction: a converted checkpoint self-describes its VAE variant.
#
# tools/convert_wan (TinyWAM) writes <ckpt>/vae_config.json; every loader -- offline
# latent materialization, closed-loop eval, this repo -- reads it and passes it here.
# The sidecar is the VAE's IDENTITY: the checkpoint, not a CLI flag, decides what
# geometry a dataset gets encoded with, which is what keeps the eval harness
# VAE-family-agnostic.
# --------------------------------------------------------------------------------------
def read_vae_sidecar(ckpt_dir: str) -> dict:
    """Read ``<ckpt_dir>/vae_config.json``. Missing or unreadable RAISES.

    TinyWAM's version fell back to the Wan2.1 defaults on any error, because it had two
    families to serve and old checkpoints with no sidecar. Here a checkpoint that does
    not identify itself is an error -- there is exactly one supported family, and
    "silently loaded the wrong VAE" is the failure mode this project refuses.
    """
    path = ckpt_dir.rstrip("/") + "/vae_config.json"
    if path.startswith("gs://"):
        import gcsfs

        try:
            fs = gcsfs.GCSFileSystem()
            with fs.open(path) as fh:
                return json.loads(fh.read())
        except Exception as exc:
            # gcsfs uses Application Default Credentials, which `gcloud auth login` does
            # NOT create -- so a workstation where gsutil works fine still reads gs://
            # anonymously and 401s. TPU VMs get ADC from the metadata server, so this is
            # normally a local-only trap; fix with `gcloud auth application-default login`.
            raise VaeError(
                f"could not read {path}. If this is a 401/anonymous-caller error, gcsfs has no "
                "Application Default Credentials -- run `gcloud auth application-default login` "
                "(gsutil's own login is separate and does not count)."
            ) from exc
    local = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(local):
        raise VaeError(f"no vae_config.json next to the checkpoint: {local}")
    with open(local) as fh:
        return json.load(fh)


def build_vae_from_sidecar(
    sidecar: dict,
    rngs: nnx.Rngs | None = None,
    dtype=jnp.float32,
    weights_dtype=jnp.float32,
) -> WanVAE38:
    """Sidecar mapping -> an (uninitialized-weights) WanVAE38 with a validated config."""
    return WanVAE38(
        config_from_sidecar(sidecar),
        rngs=rngs if rngs is not None else nnx.Rngs(0),
        dtype=dtype,
        weights_dtype=weights_dtype,
    )


def load_vae(
    ckpt_dir: str,
    dtype=jnp.float32,
    weights_dtype=jnp.float32,
) -> WanVAE38:
    """Restore the Wan2.2 VAE from an Orbax dir, sidecar-driven. Any mismatch raises.

    ``ckpt_dir`` may be local or ``gs://``. THE shared loader: offline latent
    materialization (M4) and the eval harness import this rather than owning copies.
    """
    import orbax.checkpoint as ocp

    sidecar = read_vae_sidecar(ckpt_dir)
    vae = build_vae_from_sidecar(sidecar, dtype=dtype, weights_dtype=weights_dtype)

    graphdef, abstract_state = nnx.split(vae)
    template = nnx.to_pure_dict(abstract_state)
    path = ckpt_dir if ckpt_dir.startswith("gs://") else os.path.abspath(os.path.expanduser(ckpt_dir))

    checkpointer = ocp.StandardCheckpointer()
    try:
        restored = checkpointer.restore(path, target=stringify_tree_keys(template))
    except Exception as exc:
        raise VaeError(
            f"restoring the Wan2.2 VAE from {path} failed. The checkpoint's tree must match "
            "this module exactly -- no partial restores, no strict=False."
        ) from exc
    restored = restore_tree_key_types(restored, template)

    _assert_same_tree(template, restored, path)
    nnx.replace_by_pure_dict(abstract_state, restored)
    return nnx.merge(graphdef, abstract_state)


def _assert_same_tree(template: dict, restored: dict, where: str) -> None:
    """Every key present with the right shape/dtype, nothing extra. Raises on any drift."""

    def flatten(tree, prefix=""):
        out = {}
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(flatten(value, path))
            else:
                out[path] = value
        return out

    want, got = flatten(template), flatten(restored)
    missing = sorted(set(want) - set(got))
    extra = sorted(set(got) - set(want))
    if missing or extra:
        raise VaeError(f"{where}: checkpoint tree mismatch; missing={missing[:8]} extra={extra[:8]}")
    bad = [
        f"{k}: want {tuple(want[k].shape)} {want[k].dtype}, got {tuple(got[k].shape)} {got[k].dtype}"
        for k in sorted(want)
        if tuple(want[k].shape) != tuple(got[k].shape) or want[k].dtype != got[k].dtype
    ]
    if bad:
        raise VaeError(f"{where}: {len(bad)} tensor(s) mismatch, first few: {bad[:5]}")
