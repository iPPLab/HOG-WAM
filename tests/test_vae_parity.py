"""Numerical parity of the ported VAE against TinyWAM-wan22 (JAX vs JAX).

TinyWAM-wan22's `tinywam/models/wan/vae.py` is where this port came from, and it was
measured at 9.9e-6 against the torch/diffusers reference on decode. So the question
here is narrow: did stripping Wan2.1, moving to `nnx.List`, and folding normalisation
into the entry points change any number? The answer should be "no", and the tolerance
is fp32 round-off (1e-6-ish), not a model-level 1e-4.

The two implementations cannot share a process -- TinyWAM pins jax 0.6.2 / flax 0.10.7
and flax 0.12 rejects its bare python lists of submodules -- so the golden side runs
under TinyWAM's own venv and hands over .npz. Regenerate with:

    <tinywam-python> tools/dump_vae_golden.py --tinywam-vae-source <vae.py> \
        --out tests/fixtures/vae_golden.npz

Two goldens:
  * ``vae_golden.npz``      -- tiny model, weights INSIDE the file. Always runs.
  * ``vae_golden_real.npz`` -- the released Wan2.2 VAE. Needs the same 2.4 GB Orbax
    checkpoint on this side, so it runs only when FASTWAM_VAE_CKPT points at one.
"""

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from flax import nnx  # noqa: E402

from fastwam_tpu.models.wan.vae import (  # noqa: E402
    LATENT_MEAN,
    LATENT_STD,
    WanVAE38,
    build_vae_from_sidecar,
    config_from_sidecar,
    load_vae,
)

FIXTURES = Path(__file__).parent / "fixtures"
TINY_GOLDEN = Path(os.environ.get("FASTWAM_VAE_GOLDEN", FIXTURES / "vae_golden.npz"))
REAL_GOLDEN = Path(os.environ.get("FASTWAM_VAE_GOLDEN_REAL", FIXTURES / "vae_golden_real.npz"))
REAL_CKPT = os.environ.get("FASTWAM_VAE_CKPT")

# The port is meant to be arithmetically identical, so this is a ceiling on fp32
# reassociation noise, not a modelling tolerance. Two XLA versions (jax 0.6.2 vs 0.11)
# tile the same convolutions differently, which reorders the accumulations; for scale,
# batching the SAME implementation costs 2.0e-6 on its own (see test_vae.py BATCH_TOL).
# Measured 2026-08-03 on the tiny model: encode 1.2e-6 raw / 2.1e-6 normalized,
# encode_frame 7.7e-7, encode_step 8.9e-7 and 1.2e-6, decode 2.5e-6.
TOL = 1e-5

_REGEN = (
    "missing; regenerate with\n    "
    "<tinywam-python> tools/dump_vae_golden.py --tinywam-vae-source <vae.py> --out {}"
)


def _unflatten_into(flat: dict, like: dict) -> dict:
    """Rebuild ``like``'s nesting (and key types) from "/"-joined paths."""

    def walk(sub, prefix):
        out = {}
        for key, value in sub.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            out[key] = walk(value, path) if isinstance(value, dict) else flat[path]
        return out

    return walk(like, "")


def _load(path: Path):
    if not path.exists():
        pytest.skip(_REGEN.format(path))
    return np.load(path)


@pytest.fixture(scope="module")
def tiny_golden():
    return _load(TINY_GOLDEN)


@pytest.fixture(scope="module")
def tiny_vae(tiny_golden):
    """Our model carrying the golden's exact weights, so only the code differs."""
    config = json.loads(bytes(tiny_golden["__config__"]).decode())
    vae = build_vae_from_sidecar(config)
    graphdef, state = nnx.split(vae)
    template = nnx.to_pure_dict(state)
    weights = {k[len("w/") :]: jnp.asarray(tiny_golden[k]) for k in tiny_golden.files if k.startswith("w/")}
    nnx.replace_by_pure_dict(state, _unflatten_into(weights, template))
    return nnx.merge(graphdef, state)


def _report(name, ours, golden):
    ours, golden = np.asarray(ours, np.float64), np.asarray(golden, np.float64)
    assert ours.shape == golden.shape, f"{name}: shape {ours.shape} vs golden {golden.shape}"
    err = float(np.abs(ours - golden).max())
    print(f"[parity] {name:24s} shape={ours.shape} max|delta|={err:.3e}")
    assert err <= TOL, f"{name}: max|delta| {err:.3e} > {TOL:.0e}"
    return err


# --------------------------------------------------------------------------- #
# the transcription of the hardcoded constants
# --------------------------------------------------------------------------- #
def test_latent_stats_match_the_independent_transcription():
    """tools/dump_vae_golden.py types the 48+48 constants out a second time. If the two
    copies of wan_video_vae.py:1360-1378 disagree, one of them was mistyped."""
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
    try:
        import dump_vae_golden
    finally:
        sys.path.pop(0)
    assert tuple(dump_vae_golden.LATENT_MEAN) == LATENT_MEAN
    assert tuple(dump_vae_golden.LATENT_STD) == LATENT_STD


# --------------------------------------------------------------------------- #
# tiny model, weights travel with the golden
# --------------------------------------------------------------------------- #
def test_tiny_config_survives_the_round_trip(tiny_golden):
    config = json.loads(bytes(tiny_golden["__config__"]).decode())
    assert config_from_sidecar(config).z_dim == 48


def test_tiny_encode(tiny_vae, tiny_golden):
    clip = jnp.asarray(tiny_golden["in/clip"])
    _report("encode (raw)", tiny_vae.encode(clip, normalize=False), tiny_golden["out/encode_raw"])
    _report("encode (normalized)", tiny_vae.encode(clip), tiny_golden["out/normalized"])


def test_tiny_encode_frame(tiny_vae, tiny_golden):
    frame = jnp.asarray(tiny_golden["in/frame"])
    _report(
        "encode_frame (raw)",
        tiny_vae.encode_frame(frame, normalize=False),
        tiny_golden["out/encode_frame_raw"],
    )


def test_tiny_encode_step(tiny_vae, tiny_golden):
    clip = jnp.asarray(tiny_golden["in/clip"])
    step0, cache = tiny_vae.encode_step(clip[:, :, :1], normalize=False)
    step1, _ = tiny_vae.encode_step(clip[:, :, 1:5], cache, normalize=False)
    _report("encode_step 0", step0, tiny_golden["out/encode_step0_raw"])
    _report("encode_step 1", step1, tiny_golden["out/encode_step1_raw"])


def test_tiny_decode(tiny_vae, tiny_golden):
    latents = jnp.asarray(tiny_golden["in/latents_raw"])
    _report("decode", tiny_vae.decode(latents, denormalize=False), tiny_golden["out/decode"])


# --------------------------------------------------------------------------- #
# the real released VAE
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_golden():
    return _load(REAL_GOLDEN)


@pytest.fixture(scope="module")
def real_vae():
    if not REAL_CKPT:
        pytest.skip("set FASTWAM_VAE_CKPT to the Wan2.2 VAE Orbax dir to run real-weight parity")
    return load_vae(REAL_CKPT)


def test_real_weights_load_strictly(real_vae):
    assert isinstance(real_vae, WanVAE38)
    assert (real_vae.z_dim, real_vae.spatial_downsample_factor, real_vae.temporal_downsample_factor) == (
        48,
        16,
        4,
    )
    assert real_vae.config.latents_mean == LATENT_MEAN
    assert real_vae.config.latents_std == LATENT_STD


def test_real_encode(real_vae, real_golden):
    clip = jnp.asarray(real_golden["in/clip"])
    _report("real encode (raw)", real_vae.encode(clip, normalize=False), real_golden["out/encode_raw"])
    _report("real encode (norm)", real_vae.encode(clip), real_golden["out/normalized"])


def test_real_encode_frame(real_vae, real_golden):
    frame = jnp.asarray(real_golden["in/frame"])
    _report(
        "real encode_frame",
        real_vae.encode_frame(frame, normalize=False),
        real_golden["out/encode_frame_raw"],
    )


def test_real_encode_step(real_vae, real_golden):
    clip = jnp.asarray(real_golden["in/clip"])
    step0, cache = real_vae.encode_step(clip[:, :, :1], normalize=False)
    step1, _ = real_vae.encode_step(clip[:, :, 1:5], cache, normalize=False)
    _report("real encode_step 0", step0, real_golden["out/encode_step0_raw"])
    _report("real encode_step 1", step1, real_golden["out/encode_step1_raw"])


def test_real_decode(real_vae, real_golden):
    latents = jnp.asarray(real_golden["in/latents_raw"])
    _report("real decode", real_vae.decode(latents, denormalize=False), real_golden["out/decode"])


def test_real_robotwin_single_observation(real_vae):
    """The M3 hot path at its real size: one 384x320 canvas -> [1,48,1,24,20] (spec §4.5).

    Only the single-frame case runs here. The 9-frame clip -> [1,48,3,24,20] contract is
    the same code with a longer loop and is asserted in test_shapes.py on a narrowed
    model, because shape does not depend on channel width -- and 9 frames of 384x320
    through 704 M params of CPU convolution costs ~40 min for no extra information.
    """
    import jax

    frame = jax.random.uniform(jax.random.key(0), (1, 3, 1, 384, 320), minval=-1.0, maxval=1.0)
    latents = real_vae.encode_frame(frame)
    assert latents.shape == (1, 48, 1, 24, 20)
    assert real_vae.latent_shape(9, 384, 320) == (48, 3, 24, 20)
    # normalized latents should sit near unit scale -- the stats are fitted to this VAE
    print(f"[parity] real 384x320 frame -> {latents.shape} std={float(latents.std()):.3f}")
