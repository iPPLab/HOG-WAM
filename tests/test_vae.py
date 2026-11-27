"""Wan2.2 VAE: config strictness, geometry, causality, and the entry-point contract.

Everything here runs on random weights and a tiny model, so it is fast and needs no
checkpoint. Numerical agreement with the implementation this was ported from lives in
test_vae_parity.py.
"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from flax import nnx  # noqa: E402

from fastwam_tpu.config import (  # noqa: E402
    VAE_SPATIAL_FACTOR,
    VAE_TEMPORAL_FACTOR,
    load_config,
)
from fastwam_tpu.models.wan.vae import (  # noqa: E402
    LATENT_MEAN,
    LATENT_STD,
    WAN22_VAE_GEOMETRY,
    VaeError,
    Wan22VaeConfig,
    WanVAE38,
    config_from_sidecar,
    patchify,
    unpatchify,
)

TASK = "training/full.example.yaml"

# Same shrunken geometry the parity golden uses: every structural branch, ~1 s on CPU.
TINY = Wan22VaeConfig(base_dim=8, decoder_base_dim=8)


@pytest.fixture(scope="module")
def tiny_vae():
    return WanVAE38(TINY, nnx.Rngs(0))


def _clip(t=5, h=64, w=64, seed=0):
    return jax.random.uniform(jax.random.key(seed), (1, 3, t, h, w), minval=-1.0, maxval=1.0)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_canonical_geometry_matches_the_spec():
    """spec §2.7: z_dim 48, dim 160, dec_dim 256, 16x spatial, 4x temporal."""
    c = WAN22_VAE_GEOMETRY
    assert (c.z_dim, c.base_dim, c.decoder_base_dim, c.num_res_blocks) == (48, 160, 256, 2)
    assert c.dim_mult == (1, 2, 4, 4)
    assert c.temperal_downsample == (False, True, True)
    assert (c.spatial_downsample_factor, c.temporal_downsample_factor) == (16, 4)
    assert (c.spatial_downsample_factor, c.temporal_downsample_factor) == (
        VAE_SPATIAL_FACTOR,
        VAE_TEMPORAL_FACTOR,
    )
    assert len(LATENT_MEAN) == len(LATENT_STD) == 48


def test_sidecar_round_trips():
    sidecar = {
        "base_dim": 160,
        "decoder_base_dim": 256,
        "z_dim": 48,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "temperal_downsample": [False, True, True],
        "is_residual": True,
        "in_channels": 12,
        "out_channels": 12,
        "patch_size": 2,
        "clip_output": False,
        "latents_mean": list(LATENT_MEAN),
        "latents_std": list(LATENT_STD),
    }
    assert config_from_sidecar(sidecar) == WAN22_VAE_GEOMETRY


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda s: s.pop("z_dim"), "missing"),
        (lambda s: s.update(surprise=1), "unknown key"),
        (lambda s: s.update(is_residual=False), "is_residual=False"),
        (lambda s: s.update(in_channels=3), "patchify"),
        (lambda s: s.update(latents_std=[1.0] * 48), "disagrees with the constants"),
        (lambda s: s.update(temperal_downsample=[False, True]), "one entry per"),
    ],
    ids=["missing-key", "unknown-key", "wan21-family", "unpatchified", "wrong-stats", "short-td"],
)
def test_bad_sidecar_raises(mutate, needle):
    """Every load failure is loud. The reference's strict=False is the bug being fixed."""
    sidecar = {
        "base_dim": 160, "decoder_base_dim": 256, "z_dim": 48, "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2, "temperal_downsample": [False, True, True], "is_residual": True,
        "in_channels": 12, "out_channels": 12, "patch_size": 2, "clip_output": False,
        "latents_mean": list(LATENT_MEAN), "latents_std": list(LATENT_STD),
    }
    mutate(sidecar)
    with pytest.raises(VaeError, match=needle):
        config_from_sidecar(sidecar)


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_patchify_round_trip():
    x = jax.random.normal(jax.random.key(0), (2, 3, 4, 8, 6))
    p = patchify(x, 2)
    assert p.shape == (2, 12, 4, 4, 3)
    assert jnp.array_equal(unpatchify(p, 2), x)


def test_encode_shapes(tiny_vae):
    """[B,3,T,H,W] -> [B,48,(T-1)//4+1,H/16,W/16]."""
    assert tiny_vae.encode(_clip(t=5, h=64, w=64)).shape == (1, 48, 2, 4, 4)
    assert tiny_vae.encode(_clip(t=1, h=64, w=64)).shape == (1, 48, 1, 4, 4)
    assert tiny_vae.latent_shape(9, 384, 320) == (48, 3, 24, 20)


def test_decode_shapes(tiny_vae):
    """F latent frames -> 1 + 4*(F-1) pixel frames: frame 0 is NOT upsampled."""
    z = jax.random.normal(jax.random.key(0), (1, 48, 2, 4, 4)) * 0.5
    out = tiny_vae.decode(z)
    assert out.shape == (1, 3, 5, 64, 64)
    assert tiny_vae.decode(z[:, :, :1]).shape == (1, 3, 1, 64, 64)
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0


def test_channels_last_input_matches_channels_first(tiny_vae):
    clip = _clip()
    a = tiny_vae.encode(clip)
    b = tiny_vae.encode(jnp.transpose(clip, (0, 2, 3, 4, 1)), channels_first=False)
    assert jnp.array_equal(a, b)


@pytest.mark.parametrize(
    ("kwargs", "needle"),
    [
        (dict(t=3), "T = 1 \\+ 4\\*m"),
        (dict(h=60), "must both divide by 16"),
        (dict(w=70), "must both divide by 16"),
    ],
    ids=["bad-frame-count", "bad-height", "bad-width"],
)
def test_encode_rejects_bad_geometry(tiny_vae, kwargs, needle):
    with pytest.raises(VaeError, match=needle):
        tiny_vae.encode(_clip(**kwargs))


def test_check_geometry_against_the_task_config(tiny_vae):
    shapes = load_config(TASK).shapes
    tiny_vae.check_geometry(shapes)  # must not raise
    assert tiny_vae.latent_shape(shapes.video_frames, 384, 320) == shapes.latent_shape


# --------------------------------------------------------------------------- #
# the property the whole design rests on: temporal causality
# --------------------------------------------------------------------------- #
def test_encode_is_causal_in_time(tiny_vae):
    """Latent frame i must depend only on pixel frames <= 1+4i.

    This is what lets M3 encode a single observation at inference and get the same
    number training saw. Break it and the closed loop dies silently -- it already did
    once (project memory: FastWAM 耦合移植, 闭环 0% 根因 = VAE 流式编码错配).
    """
    clip = _clip(t=5)
    tampered = clip.at[:, :, 1:].set(jax.random.normal(jax.random.key(9), clip[:, :, 1:].shape))
    assert jnp.array_equal(tiny_vae.encode(clip)[:, :, :1], tiny_vae.encode(tampered)[:, :, :1])


def test_encode_frame_equals_the_first_latent_frame(tiny_vae):
    """The M3 hot path is chunk 0 of the streaming loop, bit for bit."""
    clip = _clip(t=5)
    assert jnp.array_equal(tiny_vae.encode_frame(clip[:, :, :1]), tiny_vae.encode(clip)[:, :, :1])


def test_encode_frame_accepts_a_4d_frame(tiny_vae):
    clip = _clip(t=1)
    assert jnp.array_equal(tiny_vae.encode_frame(clip[:, :, 0]), tiny_vae.encode_frame(clip))
    with pytest.raises(VaeError, match="exactly one frame"):
        tiny_vae.encode_frame(_clip(t=5))


def test_encode_step_equals_whole_clip_encode(tiny_vae):
    """Streaming O(1)/step must equal the O(T) whole-clip encode, or the offline
    materialization (M4) and the closed loop (M3) disagree about the same pixels."""
    clip = _clip(t=9)
    whole = tiny_vae.encode(clip)
    first, cache = tiny_vae.encode_step(clip[:, :, :1])
    outs = [first]
    for start in (1, 5):
        latent, cache = tiny_vae.encode_step(clip[:, :, start : start + 4], cache)
        outs.append(latent)
    assert jnp.array_equal(jnp.concatenate(outs, axis=2), whole)


def test_encode_step_rejects_wrong_chunk_sizes(tiny_vae):
    clip = _clip(t=9)
    with pytest.raises(VaeError, match="T_step=1 on the first call"):
        tiny_vae.encode_step(clip[:, :, :4])
    _, cache = tiny_vae.encode_step(clip[:, :, :1])
    with pytest.raises(VaeError, match="T_step=4 on the subsequent call"):
        tiny_vae.encode_step(clip[:, :, 1:2], cache)


def test_encode_step_does_not_mutate_the_caller_state(tiny_vae):
    """encode_step returns a fresh dict; re-running from an old state must reproduce."""
    clip = _clip(t=9)
    _, cache0 = tiny_vae.encode_step(clip[:, :, :1])
    a, _ = tiny_vae.encode_step(clip[:, :, 1:5], cache0)
    b, _ = tiny_vae.encode_step(clip[:, :, 1:5], cache0)
    assert jnp.array_equal(a, b)


# --------------------------------------------------------------------------- #
# batching: our only intentional deviation from the reference
#
# WanVideoVAE38.encode loops `for video in videos`; we run the batch axis through.
# Every op is per-sample, so the two are the SAME COMPUTATION -- but not bit-identical
# outputs: XLA picks different convolution tilings for N=1 and N=3, which reassociates
# the accumulations. Measured 2.0e-6 (encode) / 1.6e-6 (decode) max|delta| on the tiny
# model, i.e. fp32 round-off through a 4-stage conv stack. BATCH_TOL is a ceiling on
# that noise; a real semantic bug (wrong cache indexing, a leaked batch axis) would
# land orders of magnitude above it, not just above zero.
# --------------------------------------------------------------------------- #
BATCH_TOL = 1e-5


def test_batched_encode_equals_per_sample_encode(tiny_vae):
    clip = jax.random.uniform(jax.random.key(3), (3, 3, 5, 64, 64), minval=-1.0, maxval=1.0)
    batched = tiny_vae.encode(clip)
    for i in range(clip.shape[0]):
        delta = float(jnp.abs(tiny_vae.encode(clip[i : i + 1]) - batched[i : i + 1]).max())
        assert delta < BATCH_TOL, f"sample {i}: max|delta| {delta:.3e}"


def test_batched_decode_equals_per_sample_decode(tiny_vae):
    z = jax.random.normal(jax.random.key(4), (3, 48, 2, 4, 4)) * 0.5
    batched = tiny_vae.decode(z)
    for i in range(z.shape[0]):
        delta = float(jnp.abs(tiny_vae.decode(z[i : i + 1]) - batched[i : i + 1]).max())
        assert delta < BATCH_TOL, f"sample {i}: max|delta| {delta:.3e}"


# --------------------------------------------------------------------------- #
# latent normalisation
# --------------------------------------------------------------------------- #
def test_normalize_is_the_documented_affine(tiny_vae):
    """mu = (mu - mean) * (1/std), per wan_video_vae.py:1003."""
    z = jax.random.normal(jax.random.key(5), (2, 48, 3, 4, 4))
    mean = jnp.asarray(LATENT_MEAN).reshape(1, 48, 1, 1, 1)
    std = jnp.asarray(LATENT_STD).reshape(1, 48, 1, 1, 1)
    assert jnp.allclose(tiny_vae.normalize_latents(z), (z - mean) / std, atol=1e-6)
    assert jnp.allclose(tiny_vae.denormalize_latents(tiny_vae.normalize_latents(z)), z, atol=1e-5)


def test_encode_normalize_flag(tiny_vae):
    clip = _clip()
    raw = tiny_vae.encode(clip, normalize=False)
    assert jnp.allclose(tiny_vae.encode(clip), tiny_vae.normalize_latents(raw), atol=1e-6)


def test_decode_undoes_encode_normalization(tiny_vae):
    """decode(encode(x)) and decode(encode_raw(x), denormalize=False) are the same
    picture -- i.e. the two halves of the affine really are inverses in the pipeline."""
    clip = _clip()
    a = tiny_vae.decode(tiny_vae.encode(clip))
    b = tiny_vae.decode(tiny_vae.encode(clip, normalize=False), denormalize=False)
    assert float(jnp.abs(a - b).max()) < 1e-5


# --------------------------------------------------------------------------- #
# runtime
# --------------------------------------------------------------------------- #
def test_encode_jits(tiny_vae):
    """The streaming loop is python control flow over static shapes, so it must trace."""
    clip = _clip()
    fn = nnx.jit(lambda m, x: m.encode(x))
    assert jnp.allclose(fn(tiny_vae, clip), tiny_vae.encode(clip), atol=1e-5)


def test_encode_frame_jits(tiny_vae):
    frame = _clip(t=1)
    fn = nnx.jit(lambda m, x: m.encode_frame(x))
    assert jnp.allclose(fn(tiny_vae, frame), tiny_vae.encode_frame(frame), atol=1e-5)


def test_param_tree_is_what_the_released_checkpoint_holds(tiny_vae):
    """196 tensors with the diffusers names; the parameter-free shortcuts contribute
    none. A rename here silently breaks every existing checkpoint."""
    _, state = nnx.split(tiny_vae)
    flat = nnx.to_pure_dict(state)

    def paths(tree, prefix=""):
        for key, value in tree.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                yield from paths(value, path)
            else:
                yield path

    keys = set(paths(flat))
    assert len(keys) == 196
    assert {k.split("/")[0] for k in keys} == {"encoder", "decoder", "quant_conv", "post_quant_conv"}
    assert "encoder/down_blocks/1/downsampler/time_conv/conv/kernel" in keys
    assert "decoder/up_blocks/0/upsampler/resample/layers/1/kernel" in keys
    assert not any("avg_shortcut" in k for k in keys)  # AvgDown3D / DupUp3D hold no weights
