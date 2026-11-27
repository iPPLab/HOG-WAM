"""Dump VAE parity goldens from the TinyWAM-wan22 implementation (JAX vs JAX).

The golden is TinyWAM-wan22's ``tinywam/models/wan/vae.py``: the same diffusers port
this repo's VAE was derived from, measured at 9.9e-6 against torch on decode. Since
both sides are JAX the comparison is not "does the math match torch" (TinyWAM already
answered that) but "did the wan22-only rewrite change a single number". It should not
have, so the bar is fp32 round-off, not 1e-4.

TinyWAM pins jax 0.6.2 / flax 0.10.7 and this repo pins jax >= 0.9 / flax >= 0.12, and
flax 0.12 rejects the bare-python-list-of-submodules that 0.10 allowed. So the two
implementations cannot be imported into one process. This script runs under TinyWAM's
own venv, writes goldens to .npz, and tests/test_vae_parity.py reads them back here.

    # structural golden: a tiny VAE whose weights travel inside the npz, so the test
    # is self-contained (no 2.4 GB checkpoint, no network).
    <tinywam-python> tools/dump_vae_golden.py \
        --tinywam-vae-source <TinyWAM-wan22>/tinywam/models/wan/vae.py \
        --out tests/fixtures/vae_golden.npz

    # real-weight golden: the released Wan2.2 VAE. Weights are NOT stored; the test
    # needs the same checkpoint via FASTWAM_VAE_CKPT.
    <tinywam-python> tools/dump_vae_golden.py \
        --tinywam-vae-source <TinyWAM-wan22>/tinywam/models/wan/vae.py \
        --out tests/fixtures/vae_golden_real.npz --ckpt <vae-orbax-path>

This file imports nothing from fastwam_tpu on purpose -- it has to run under a venv
where this package is not installed.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

# A SECOND, INDEPENDENT transcription of the 48+48 constants at
# reference/FastWAM/src/fastwam/models/wan22/wan_video_vae.py:1360-1378. Typed out here
# rather than imported because this script runs where fastwam_tpu is not installed --
# and test_vae_parity.py asserts the two copies agree, so the duplication is a
# transcription check instead of a drift risk.
LATENT_MEAN = [
    -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557,
    -0.1382, 0.0542, 0.2813, 0.0891, 0.1570, -0.0098, 0.0375, -0.1825,
    -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
    -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230,
    -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.0520, 0.3748,
    0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
]
LATENT_STD = [
    0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013,
    0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978,
    0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
    0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093,
    0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887,
    0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
]

# Small enough to run on CPU in seconds, large enough to exercise every branch:
# 4 stages (so both downsample2d and downsample3d fire), patchify, the residual
# shortcuts, and the mid-block attention. z_dim stays 48 so the real latent stats
# apply unchanged.
TINY_CONFIG = {
    "base_dim": 8,
    "decoder_base_dim": 8,
    "z_dim": 48,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "temperal_downsample": [False, True, True],
    "is_residual": True,
    "in_channels": 12,
    "out_channels": 12,
    "patch_size": 2,
    "clip_output": False,
    "latents_mean": LATENT_MEAN,
    "latents_std": LATENT_STD,
}

# (T, H, W) of the pixel probes. T=5 -> 2 latent frames, which is the smallest input
# that exercises BOTH the frame-0-alone chunk and a 4-frame chunk.
CLIP_SHAPE = (5, 64, 64)
SEED = 20260803


def load_tinywam_vae_module(source: Path):
    """Import TinyWAM's vae.py by path, bypassing tinywam.models.wan.__init__ (which
    pulls in the transformer and its dependencies for no reason here)."""
    spec = importlib.util.spec_from_file_location("tinywam_golden_vae", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def flatten(tree, prefix=""):
    out = {}
    for key, value in tree.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, dict):
            out.update(flatten(value, path))
        else:
            out[path] = value
    return out


def unflatten(flat, like):
    """Rebuild ``like``'s nesting (and its key types) from "/"-joined paths."""

    def walk(sub, prefix):
        out = {}
        for key, value in sub.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            out[key] = walk(value, path) if isinstance(value, dict) else flat[path]
        return out

    return walk(like, "")


def random_weights(template_flat: dict, seed: int = SEED) -> dict:
    """Deterministic numpy weights keyed by path. Both venvs get bit-identical values,
    which jax's own PRNG could not promise across a 0.6 -> 0.11 jump."""
    rng = np.random.default_rng(seed)
    out = {}
    for path in sorted(template_flat):  # sorted -> order-independent of tree traversal
        shape = tuple(template_flat[path].shape)
        if path.endswith("/gamma"):
            value = 1.0 + rng.normal(0.0, 0.01, shape)
        elif path.endswith("/bias"):
            value = rng.normal(0.0, 0.01, shape)
        else:
            value = rng.normal(0.0, 0.05, shape)
        out[path] = value.astype(np.float32)
    return out


def probes(z_dim: int, spatial_factor: int, seed: int = SEED) -> dict:
    """The fixed inputs both implementations are fed."""
    rng = np.random.default_rng(seed + 1)
    t, h, w = CLIP_SHAPE
    lh, lw = h // spatial_factor, w // spatial_factor
    return {
        # pixels in [-1, 1], the VAE's declared input range
        "clip": rng.uniform(-1.0, 1.0, (1, 3, t, h, w)).astype(np.float32),
        "frame": rng.uniform(-1.0, 1.0, (1, 3, 1, h, w)).astype(np.float32),
        # raw (un-normalized) latents for decode, in a plausible posterior-mean range
        "latents_raw": rng.normal(0.0, 0.5, (1, z_dim, 2, lh, lw)).astype(np.float32),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tinywam-vae-source", required=True, type=Path)
    parser.add_argument(
        "--ckpt",
        default=None,
        help="real Wan2.2 VAE Orbax dir (local or gs://). Omit for the tiny self-contained golden.",
    )
    args = parser.parse_args(argv)

    import jax.numpy as jnp
    from flax import nnx

    vae_mod = load_tinywam_vae_module(args.tinywam_vae_source)
    print(f"[golden] tinywam vae from {args.tinywam_vae_source}", flush=True)

    payload = {}
    if args.ckpt:
        vae = vae_mod.load_vae(args.ckpt, with_encoder=True)
        config = vae_mod.read_vae_sidecar(args.ckpt)
        payload["__weights__"] = np.array(0)  # marker: weights come from the checkpoint
    else:
        config = dict(TINY_CONFIG)
        vae = vae_mod.build_vae_from_config(nnx.Rngs(0), config, with_encoder=True)
        graphdef, state = nnx.split(vae)
        template = nnx.to_pure_dict(state)
        weights = random_weights(flatten(template))
        nnx.replace_by_pure_dict(state, unflatten({k: jnp.asarray(v) for k, v in weights.items()}, template))
        vae = nnx.merge(graphdef, state)
        for path, value in weights.items():
            payload[f"w/{path}"] = value
        print(f"[golden] tiny model, {len(weights)} weight tensors", flush=True)

    payload["__config__"] = np.frombuffer(json.dumps(config).encode(), dtype=np.uint8)

    inputs = probes(vae.z_dim, vae.spatial_downsample_factor)
    for name, value in inputs.items():
        payload[f"in/{name}"] = value

    clip = jnp.asarray(inputs["clip"])
    frame = jnp.asarray(inputs["frame"])
    latents_raw = jnp.asarray(inputs["latents_raw"])

    # TinyWAM's encode/decode are RAW-latent; normalization lives in normalize_latents.
    payload["out/encode_raw"] = np.asarray(vae.encode(clip))
    payload["out/encode_frame_raw"] = np.asarray(vae.encode(frame))
    payload["out/normalized"] = np.asarray(vae.normalize_latents(jnp.asarray(payload["out/encode_raw"])))
    payload["out/decode"] = np.asarray(vae.decode(latents_raw))

    # streaming: frame 0 alone, then one 4-frame unit -- must equal encode() on the clip.
    step0, cache = vae.encode_step(clip[:, :, :1])
    step1, _ = vae.encode_step(clip[:, :, 1:5], cache)
    payload["out/encode_step0_raw"] = np.asarray(step0)
    payload["out/encode_step1_raw"] = np.asarray(step1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"[golden] wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)", flush=True)
    for key in sorted(k for k in payload if k.startswith("out/")):
        arr = payload[key]
        print(f"[golden]   {key:26s} {arr.shape} mean={arr.mean():+.6f} std={arr.std():.6f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
