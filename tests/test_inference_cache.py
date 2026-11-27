"""M3 numerical gate: the KV-cache inference path, and the policy that wraps it.

Two things are proven here and nothing else is worth proving twice:

1. **The cache changes no numbers.** `FastWAM.infer_action` (prefill once, 10 action
   passes) against `FastWAM.denoise_action` (the same 10 steps with the video tower
   re-run each time), which M2 already pinned against the float64 golden. If these two
   agree, the cached path inherits M2's parity for free; if they disagree, no amount of
   golden-chasing helps, because the disagreement is an implementation slip in the
   caching, not a semantics question. The mask-level identity underneath is
   `test_mot_parity.py::test_action_output_equals_single_frame_prefill`.
2. **The policy's arithmetic.** Canvas -> [-1,1] -> [1,3,H,W], proprio z-score with the
   +-5 clamp on the STATE stats, action un-z-score with NO clamp on the ACTION stats,
   the fixed-noise convention, and the shapes in between. Cross-wiring the two stat sets
   is the kind of bug that produces a plausible-looking robot that never succeeds, so it
   gets an explicit test with deliberately different numbers on each side.

Widths are `test_mot_parity.py`'s small config (2 layers, 4 heads x 24) so the file runs
on CPU in seconds. The VAE is NOT built here — at 1 GB it is not a unit-test object, so
the policy tests inject a stub encoder with the right signature and the real one is
exercised by `tests/test_vae.py` and the S0 smoke run.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from test_mot_parity import (
    ACTION_DIM,
    ACTION_HORIZON,
    BATCH,
    CONTEXT_LEN,
    PROPRIO_DIM,
    build_fastwam,
    load_flat_params,
    rel_error,
    report,
    sample_batch,
    small_model_config,
)

from fastwam_tpu.inference.policy import FastWAMPolicy, PolicyError
from fastwam_tpu.models.fastwam import FastWAM
from fastwam_tpu.models.mot import VideoKVCache, build_mot_attention_mask

CACHE_TOL = 1e-6
FULL_CONFIG = "training/full.example.yaml"


def denoise_kwargs(batch, seed=91):
    return dict(
        first_frame_latents=batch["latents"][:, :, :1],
        noise_action=jax.random.normal(jax.random.key(seed), batch["action"].shape),
        context=batch["context"],
        context_mask=batch["context_mask"],
        proprio=batch["proprio"][:, 0],
        num_inference_steps=10,
    )


# --------------------------------------------------------------------------- #
# GATE: the cache is exact
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scan_layers", [False, True])
def test_cached_matches_uncached_denoise(scan_layers):
    """The M3 acceptance gate: `infer_action` == `denoise_action` to ~1e-6.

    Run for both layer layouts because they are genuinely different code paths in the
    cache (a python loop over a stacked list vs an `nnx.scan` over `(blocks, k, v)`), and
    `scan_layers` is the layout any real TPU run uses.
    """
    from test_mot_parity import LAYERS, flat_params

    from tools.convert_weights import stack_blocks_for_scan

    model = build_fastwam()
    if scan_layers:
        scanned = build_fastwam(scan_layers=True)
        load_flat_params(scanned.mot.video, stack_blocks_for_scan(flat_params(model.mot.video), LAYERS))
        load_flat_params(scanned.mot.action, stack_blocks_for_scan(flat_params(model.mot.action), LAYERS))
        load_flat_params(scanned.proprio_encoder, flat_params(model.proprio_encoder))
        model = scanned

    batch = sample_batch()
    kwargs = denoise_kwargs(batch)
    cached = model.infer_action(**kwargs)
    plain = model.denoise_action(**kwargs)
    assert cached.shape == plain.shape == (BATCH, ACTION_HORIZON, ACTION_DIM)
    name = f"infer_action vs denoise_action (scan={scan_layers})"
    assert report(name, rel_error(cached, plain)) < CACHE_TOL


def test_cached_matches_the_full_joint_path():
    """...and therefore also `denoise_joint`, which is the reference's own assertion
    (`test_action_with_infer_action`, atol/rtol 1e-2 upstream). Held here at 1e-5: the
    joint path denoises three latent frames the cached path never computes, so the two
    differ only by fp32 reassociation, not by anything the mask lets through."""
    model = build_fastwam()
    batch = sample_batch()
    kwargs = denoise_kwargs(batch)
    cached = model.infer_action(**kwargs)
    joint = model.denoise_joint(
        noise_video=jax.random.normal(jax.random.key(93), batch["latents"].shape), **kwargs
    )["action"]
    assert report("infer_action vs denoise_joint", rel_error(cached, joint)) < 1e-5


def test_prefill_cache_shape_and_content():
    """The cache is exactly the K/V the joint forward builds for the video prefix.

    Checked by rebuilding layer 0's key from the block itself: if `prefill_video_cache`
    ever cached the pre-RoPE or pre-RMSNorm tensor, every downstream number would still
    be finite and plausible, and only the closed-loop score would notice."""
    model = build_fastwam()
    batch = sample_batch()
    first_frame = batch["latents"][:, :, :1]
    context, context_mask = model.append_proprio(
        batch["context"], batch["context_mask"], batch["proprio"][:, 0]
    )
    state = model.mot.video.pre_dit(first_frame, jnp.zeros((BATCH,)), context, context_mask)
    per_frame = state.tokens_per_frame
    mask = build_mot_attention_mask(state.tokens.shape[1], ACTION_HORIZON, per_frame)
    cache = model.mot.prefill_video_cache(state, mask[: state.tokens.shape[1], : state.tokens.shape[1]])

    layers = model.mot.num_layers
    inner = model.mot.num_heads * model.mot.head_dim
    assert isinstance(cache, VideoKVCache)
    assert cache.k.shape == cache.v.shape == (layers, BATCH, per_frame, inner)

    block = model.mot.video.blocks[0]
    (_, key, value), _ = block.pre_attention(state.tokens, state.t_mod, state.cos, state.sin)
    merged_k = np.asarray(key).reshape(BATCH, per_frame, inner)
    merged_v = np.asarray(value).reshape(BATCH, per_frame, inner)
    assert rel_error(cache.k[0], merged_k) == 0.0
    assert rel_error(cache.v[0], merged_v) == 0.0


def test_prefill_rejects_a_mask_that_is_not_the_video_block():
    model = build_fastwam()
    batch = sample_batch()
    state = model.mot.video.pre_dit(
        batch["latents"][:, :, :1], jnp.zeros((BATCH,)), batch["context"], batch["context_mask"]
    )
    wrong = build_mot_attention_mask(state.tokens.shape[1], ACTION_HORIZON, state.tokens_per_frame)
    with pytest.raises(Exception, match="square"):
        model.mot.prefill_video_cache(state, wrong)


def test_cached_action_forward_rejects_a_mismatched_cache():
    """A cache prefilled for a different Sv (or a different batch) must raise, not
    broadcast. Silently attending over the wrong prefix is the failure mode this whole
    milestone is trying not to have."""
    model = build_fastwam()
    batch = sample_batch()
    state = model.mot.video.pre_dit(
        batch["latents"][:, :, :1], jnp.zeros((BATCH,)), batch["context"], batch["context_mask"]
    )
    video_tokens = state.tokens.shape[1]
    mask = build_mot_attention_mask(video_tokens, ACTION_HORIZON, state.tokens_per_frame)
    cache = model.mot.prefill_video_cache(state, mask[:video_tokens, :video_tokens])
    action_state = model.mot.action.pre_dit(
        batch["action"], jnp.zeros((BATCH,)), batch["context"], batch["context_mask"]
    )

    truncated = VideoKVCache(k=cache.k[:, :, :-1], v=cache.v[:, :, :-1])
    with pytest.raises(Exception, match="video tokens"):
        model.mot.forward_action_with_video_cache(action_state, truncated, mask, video_tokens)
    with pytest.raises(Exception, match="layers"):
        model.mot.forward_action_with_video_cache(
            action_state, VideoKVCache(k=cache.k[:1], v=cache.v[:1]), mask, video_tokens
        )


def test_infer_action_rejects_more_than_one_latent_frame():
    """The prefill is only valid for a single observed frame; three would silently cache
    K/V the action rows are masked away from anyway."""
    model = build_fastwam()
    batch = sample_batch()
    kwargs = denoise_kwargs(batch)
    kwargs["first_frame_latents"] = batch["latents"]
    with pytest.raises(Exception, match="one encoded observation"):
        model.infer_action(**kwargs)


def test_infer_action_runs_under_jit():
    """The server jits the whole replan; a path that only works eagerly is not usable."""
    model = build_fastwam()
    batch = sample_batch()
    kwargs = denoise_kwargs(batch)

    @nnx.jit
    def run(module, first_frame, noise, context, context_mask, proprio):
        return module.infer_action(
            first_frame_latents=first_frame,
            noise_action=noise,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_inference_steps=10,
        )

    got = run(
        model,
        kwargs["first_frame_latents"],
        kwargs["noise_action"],
        kwargs["context"],
        kwargs["context_mask"],
        kwargs["proprio"],
    )
    want = model.infer_action(**kwargs)
    assert report("infer_action jit vs eager", rel_error(got, want)) < 1e-6


# --------------------------------------------------------------------------- #
# the policy
# --------------------------------------------------------------------------- #
CANVAS_H, CANVAS_W = 32, 24  # a small stand-in for 384x320; the policy never assumes 384


class StubVae:
    """`encode_frame(frame, normalize=True, channels_first=True) -> [B, C, 1, h, w]`.

    Deliberately not a subclass of anything: the policy is supposed to need exactly this
    one method, and a stub that satisfies it is the proof. The map is affine in the pixel
    mean so a test can tell "the policy fed the VAE the right thing" from the output.
    """

    def __init__(self, channels, height, width, layers=1):
        self.channels, self.height, self.width = channels, height, width
        self.calls = []

    def encode_frame(self, frame, normalize=True, channels_first=True):
        self.calls.append((tuple(frame.shape), bool(normalize), bool(channels_first)))
        mean = jnp.mean(frame.astype(jnp.float32), axis=(1, 2, 3), keepdims=False)  # [B]
        base = jnp.arange(self.channels * self.height * self.width, dtype=jnp.float32)
        base = base.reshape(1, self.channels, 1, self.height, self.width) / 1000.0
        return base + mean.reshape(-1, 1, 1, 1, 1)


class StubStats:
    """`OfficialStats`-shaped, with action and proprio constants that are DIFFERENT.

    That difference is the whole point: swapping the two sets is a silent bug upstream
    (`global_mean[0]` is -0.2436 for action vs -0.2251 for state, so a swap survives every
    shape check), and here it changes the numbers by a factor of ten. The two also have
    different WIDTHS at the parity scale (action 6, state 5), which is a second, cruder
    tripwire on the same mistake — at the released widths both are 14 and only the values
    tell them apart.
    """

    def __init__(self, action_dim, proprio_dim):
        from fastwam_tpu.data.normalize import Normalizer

        self.action = Normalizer(
            np.full(action_dim, 1.0, np.float32), np.full(action_dim, 2.0, np.float32)
        )
        self.proprio = Normalizer(
            np.full(proprio_dim, -3.0, np.float32), np.full(proprio_dim, 0.5, np.float32)
        )
        self.num_episodes = 27500
        self.num_transition = 6075103
        self.sha256 = "0" * 64
        self.source = "<stub>"

    def describe(self):
        return "stub stats"


def build_policy(**overrides):
    """A policy over the small model, with a latent grid the small video tower accepts."""
    from test_mot_parity import IN_CH, LATENT_SHAPE

    model = build_fastwam()
    vae = StubVae(IN_CH, LATENT_SHAPE[3], LATENT_SHAPE[4])
    kwargs = dict(
        model=model,
        vae=vae,
        stats=StubStats(ACTION_DIM, PROPRIO_DIM),
        action_horizon=ACTION_HORIZON,
        num_inference_steps=4,
        canvas_hw=(CANVAS_H, CANVAS_W),
        context_len=CONTEXT_LEN,
        text_dim=None,
    )
    kwargs.update(overrides)
    return FastWAMPolicy(**kwargs)


def sample_context():
    from test_mot_parity import TEXT_DIM

    context = np.asarray(jax.random.normal(jax.random.key(11), (CONTEXT_LEN, TEXT_DIM)), np.float32)
    return context, np.ones((CONTEXT_LEN,), bool)


def sample_canvas(seed=3):
    return np.asarray(
        jax.random.randint(jax.random.key(seed), (CANVAS_H, CANVAS_W, 3), 0, 256), np.uint8
    )


def test_policy_end_to_end_shapes_and_dtype():
    policy = build_policy()
    context, mask = sample_context()
    qpos = np.linspace(-1.0, 1.0, PROPRIO_DIM, dtype=np.float32)
    chunk = policy.infer_action(sample_canvas(), qpos, context, mask)
    assert isinstance(chunk, np.ndarray)
    assert chunk.shape == (ACTION_HORIZON, ACTION_DIM)
    assert chunk.dtype == np.float32
    assert np.isfinite(chunk).all()
    # The VAE saw ONE channels-first frame, normalized — the three things the official
    # `_encode_input_image_latents_tensor` + `WanVideoVAE38.encode` do. `encode_frame`
    # accepts the 4-D [B,3,H,W] spelling and adds the T axis itself.
    assert policy.vae.calls[-1] == ((1, 3, CANVAS_H, CANVAS_W), True, True)


def test_canvas_is_mapped_to_minus_one_one_channels_first():
    """`image * (2/255) - 1` then `permute(2,0,1).unsqueeze(0)` (deploy_policy.py:229-233).

    Checked on the extremes so an off-by-one in the scale (e.g. /127.5 vs *(2/255)) shows
    up: 0 -> -1 exactly, 255 -> +1 exactly."""
    policy = build_policy()
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas[..., 0] = 255
    tensor = np.asarray(policy.canvas_to_tensor(canvas))
    assert tensor.shape == (1, 3, CANVAS_H, CANVAS_W)
    assert tensor[0, 0].min() == pytest.approx(1.0)
    assert tensor[0, 1].max() == pytest.approx(-1.0)
    # channel-first, not a transposed-HW muddle
    red_plane = np.asarray(tensor)[0].transpose(1, 2, 0)[..., 0]
    assert np.array_equal(red_plane, np.full((CANVAS_H, CANVAS_W), 1.0, np.float32))


def test_proprio_uses_state_stats_forward_with_the_clamp():
    policy = build_policy()
    stats = policy.stats
    qpos = np.full(PROPRIO_DIM, 7.0, np.float32)
    got = policy.normalize_proprio(qpos)
    want = np.clip((qpos - stats.proprio.mean) / (stats.proprio.std + 1e-8), -5.0, 5.0)
    assert np.allclose(got, want, atol=1e-5)
    # The clamp is real: (7 - -3)/0.5 = 20 -> 5.
    assert np.allclose(got, 5.0)
    # And it is the STATE stats, not the action ones: fed the same 7.0 the action
    # constants give (7-1)/2 = 3, nowhere near the clamp — so a swapped stat set would
    # be a plausible-looking number, not an exception.
    same_value = np.full(stats.action.dim, 7.0, np.float32)
    action_side = np.clip((same_value - stats.action.mean) / stats.action.std, -5.0, 5.0)
    assert np.allclose(action_side, 3.0)
    assert not np.allclose(got[0], action_side[0])


def test_action_is_denormalized_with_action_stats_and_no_clamp():
    policy = build_policy()
    stats = policy.stats
    normalized = np.full((ACTION_HORIZON, ACTION_DIM), 40.0, np.float32)
    got = policy.denormalize_action(normalized)
    want = normalized * (stats.action.std + 1e-8) + stats.action.mean
    assert np.allclose(got, want, atol=1e-3)
    # No clamp anywhere on the way out: 40 * 2 + 1 = 81 must survive.
    assert got.max() > 50.0
    # Round trip through the forward map (inside +-5, where the clamp is inactive).
    inside = np.full((ACTION_HORIZON, ACTION_DIM), 1.5, np.float32)
    assert np.allclose(stats.action.forward(policy.denormalize_action(inside)), inside, atol=1e-4)


def test_fixed_noise_is_reused_across_replans_and_resample_is_not():
    """D4: the official recreates `torch.Generator().manual_seed(42)` on EVERY call, so
    every replan of every episode uses the identical action noise. We cannot reproduce
    torch's stream, but we reproduce the property — and expose the alternative."""
    context, mask = sample_context()
    qpos = np.zeros(PROPRIO_DIM, np.float32)
    canvas = sample_canvas()

    fixed = build_policy(noise_mode="fixed")
    first = fixed.infer_action(canvas, qpos, context, mask)
    second = fixed.infer_action(canvas, qpos, context, mask)
    assert np.array_equal(first, second), "fixed noise must make a repeated replan bit-identical"
    fixed.reset()
    assert np.array_equal(fixed.infer_action(canvas, qpos, context, mask), first)

    resampled = build_policy(noise_mode="resample")
    a = resampled.infer_action(canvas, qpos, context, mask)
    b = resampled.infer_action(canvas, qpos, context, mask)
    assert not np.allclose(a, b), "resample mode must draw a fresh key per replan"


def test_policy_matches_the_model_it_wraps():
    """The policy is a wrapper, not a second implementation: given the same latent, the
    same noise and the same context, its chunk is `infer_action` denormalized."""
    policy = build_policy(noise_mode="fixed")
    context, mask = sample_context()
    qpos = np.linspace(-2.0, 2.0, PROPRIO_DIM, dtype=np.float32)
    canvas = sample_canvas()
    got = policy.infer_action(canvas, qpos, context, mask)

    tensor = policy.canvas_to_tensor(canvas)
    latents = policy.vae.encode_frame(tensor, normalize=True, channels_first=True)
    want_norm = policy.model.infer_action(
        first_frame_latents=latents,
        noise_action=policy.noise,
        context=jnp.asarray(context)[None],
        context_mask=jnp.asarray(mask)[None],
        proprio=jnp.asarray(policy.normalize_proprio(qpos))[None],
        num_inference_steps=policy.num_inference_steps,
    )
    want = policy.denormalize_action(np.asarray(want_norm[0], np.float32))
    assert report("policy vs model.infer_action", rel_error(got, want)) < 1e-5


def test_policy_rejects_the_wrong_canvas_and_context_shapes():
    policy = build_policy()
    context, mask = sample_context()
    qpos = np.zeros(PROPRIO_DIM, np.float32)
    with pytest.raises(PolicyError, match="canvas"):
        policy.infer_action(np.zeros((CANVAS_H + 1, CANVAS_W, 3), np.uint8), qpos, context, mask)
    with pytest.raises(PolicyError, match="canvas"):
        policy.infer_action(np.zeros((CANVAS_H, CANVAS_W, 3), np.float32), qpos, context, mask)
    with pytest.raises(PolicyError, match="qpos"):
        policy.infer_action(sample_canvas(), np.zeros(PROPRIO_DIM + 1, np.float32), context, mask)
    with pytest.raises(PolicyError, match="context"):
        policy.infer_action(sample_canvas(), qpos, context[:-1], mask[:-1])
    with pytest.raises(PolicyError, match="context_mask"):
        policy.infer_action(sample_canvas(), qpos, context, mask[:-1])


def test_policy_refuses_a_context_mask_with_a_hole():
    """The convention is zero-the-padding-then-declare-everything-visible (spec §2.8). A
    mask with a False in it means the caller built the embedding some other way, and the
    model would then be reading a context it was never trained to see."""
    policy = build_policy()
    context, mask = sample_context()
    holed = mask.copy()
    holed[3] = False
    with pytest.raises(PolicyError, match="all-True"):
        policy.infer_action(sample_canvas(), np.zeros(PROPRIO_DIM, np.float32), context, holed)


def test_conventions_banner_names_every_value_that_can_silently_break_the_score():
    """The startup banner is a control, not decoration: R1 in the design memo is exactly
    'a convention drifted and nothing said so'. Assert it says the numbers out loud."""
    policy = build_policy()
    banner = policy.describe()
    tokens = (
        f"horizon={ACTION_HORIZON}",
        f"denoise={policy.num_inference_steps}",
        "replan=24",
        "shift=",
        "canvas=",
        "channel_order=RGB",
        "flip=none",
        "clamp",
        "noise=fixed",
        "kv_cache=ON",
    )
    for token in tokens:
        assert token in banner, f"{token!r} missing from the conventions banner:\n{banner}"


def test_real_geometry_config_derives_the_prefill_shapes():
    """The released config's own numbers, so a yaml edit that breaks the 120/152 geometry
    fails here rather than 40 minutes into a campaign."""
    from fastwam_tpu import load_config

    shapes = load_config(FULL_CONFIG).shapes
    assert (shapes.prefill_video_tokens, shapes.prefill_joint_seq) == (120, 152)
    assert shapes.action_tokens == 32
    mask = np.asarray(
        build_mot_attention_mask(
            shapes.prefill_video_tokens, shapes.action_tokens, shapes.tokens_per_frame
        )
    )
    assert mask.shape == (152, 152)
    # At Sv = one frame three of the four blocks are fully visible, and the fourth is
    # not: the video rows still cannot see the action columns. That asymmetry is the
    # half of the theorem that makes the cache reusable across denoising steps (the
    # video K/V do not move when the action tokens do), so it is worth an assertion
    # rather than an `all()` that would happily pass on a mask with the block filled in.
    video, action = 120, 32
    assert mask[:video, :video].all()
    assert mask[video:, :video].all()
    assert mask[video:, video:].all()
    assert not mask[:video, video:].any()
    assert int(mask.sum()) == video * video + action * (video + action)


def test_small_config_is_the_released_one_narrowed():
    """Guard against the small parity config drifting away from the real schedulers —
    the cache gate would still pass while measuring the wrong loop."""
    from test_mot_parity import small_model_config

    from fastwam_tpu import load_config

    released = load_config(FULL_CONFIG).model
    small = small_model_config()
    assert small.action_scheduler == released.action_scheduler
    assert small.video_scheduler == released.video_scheduler
    assert dataclasses.asdict(small.loss) == dataclasses.asdict(released.loss)


# --------------------------------------------------------------------------- #
# the canvas, across the two implementations that must agree
# --------------------------------------------------------------------------- #
def _pil_canvas(head: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """`deploy_policy.py:130-133, 221-234` — the official EVAL canvas, verbatim.

    Kept here rather than imported from the eval worktree's client adapter, because the
    point is to pin the contract itself: if either side drifts, this fires.
    """
    from PIL import Image

    def resize(image, size_wh):
        pil = Image.fromarray(np.asarray(image, np.uint8), mode="RGB")
        return np.asarray(pil.resize(size_wh, resample=Image.BILINEAR), np.uint8)

    top = resize(head, (320, 256))
    bottom = np.concatenate([resize(left, (160, 128)), resize(right, (160, 128))], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _structured_frame(seed: int) -> np.ndarray:
    """Smooth gradients plus a little noise. White noise would hide a resize-kernel
    difference completely — every kernel averages noise to the same grey."""
    y, x = np.mgrid[0:240, 0:320]
    base = np.sin(x / 11.0 + seed) * 60 + np.cos(y / 7.0 - seed) * 60 + 128
    stacked = np.stack([base, np.roll(base, 13, 1), np.roll(base, 29, 0)], -1)
    stacked = stacked + np.random.default_rng(seed).normal(0, 6, stacked.shape)
    return np.clip(stacked, 0, 255).astype(np.uint8)


def test_eval_canvas_matches_the_training_canvas():
    """The eval client (PIL, uint8) and the training loader (`data/canvas.py`, float32
    antialiased bilinear) must produce the same picture.

    They are two different implementations on purpose — the official eval resizes uint8
    with PIL and the official training pipeline resizes float with torchvision — so the
    honest bar is the uint8 quantisation the official eval also carries, i.e. ONE step
    of 2/255. Anything structural (tile order, a flip, BGR) is ~180x larger, and the two
    negative controls below pin that gap so this test cannot pass vacuously.
    """
    from fastwam_tpu.data.canvas import CANVAS_HW, compose_canvas

    head, left, right = (_structured_frame(s) for s in (1, 2, 3))
    from_eval = _pil_canvas(head, left, right).astype(np.float32) * (2.0 / 255.0) - 1.0
    from_training = compose_canvas(
        {"head": head[None], "left_wrist": left[None], "right_wrist": right[None]},
        source_color_order="rgb",  # the SIM gives true RGB; the L3 JPEGs are BGR-baked
    )[0]

    assert from_eval.shape == from_training.shape == (*CANVAS_HW, 3)
    worst = float(np.max(np.abs(from_eval - from_training)))
    assert report("eval canvas vs training canvas", worst) <= 2.0 / 255.0 + 1e-6

    # Negative controls: the test would be worthless if these were the same order.
    flipped = float(np.max(np.abs(from_eval[:, ::-1] - from_training)))
    swapped = float(np.max(np.abs(from_eval[..., ::-1] - from_training)))
    assert flipped > 50 * worst, f"an H-flip only moved it {flipped:.4f}; no teeth"
    assert swapped > 50 * worst, f"a BGR swap only moved it {swapped:.4f}; no teeth"


def test_the_canvas_tiles_are_the_portrait_board_not_the_landscape_one():
    """384x320 portrait, head over left|right. Our own `RobotwinTriple` is 320x512
    landscape with 192/128 tiles — a completely different picture, and reusing it is the
    single most likely way to lose this milestone quietly."""
    from fastwam_tpu.data.canvas import CANVAS_HW, TILES

    assert CANVAS_HW == (384, 320)
    assert TILES == (
        ("head", (0, 0, 256, 320)),
        ("left_wrist", (256, 0, 128, 160)),
        ("right_wrist", (256, 160, 128, 160)),
    )


def test_cache_is_exact_in_bfloat16_too():
    """The gate above runs fp32; the deployment runs bf16, and that is where a cache bug
    would actually live.

    It matters because bf16 has 8 mantissa bits: if the cached and uncached paths ever
    computed the video K/V through *different* op sequences, fp32 would hide it at ~1e-7
    and bf16 would show it at ~1e-2 — the same magnitude as a real error. Running the
    identity at the deployment dtype is what rules that out.
    """
    model = FastWAM(
        small_model_config(), rngs=nnx.Rngs(0), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16
    )
    batch = sample_batch()
    kwargs = denoise_kwargs(batch)
    kwargs["noise_action"] = kwargs["noise_action"].astype(jnp.bfloat16)
    cached = model.infer_action(**kwargs)
    plain = model.denoise_action(**kwargs)
    assert cached.dtype == plain.dtype == jnp.bfloat16
    assert report("infer_action vs denoise_action (bf16)", rel_error(cached, plain)) < CACHE_TOL


# --------------------------------------------------------------------------- #
# the deployment loader
# --------------------------------------------------------------------------- #
def test_restore_into_reads_the_three_subtree_layout(tmp_path):
    """`inference/load.py::restore_into` against a checkpoint written the way
    `tools/convert_weights.py convert-official` writes the real one.

    This is the first code that runs at S0 and the only piece M3 cannot exercise against
    the 12 GB artifact on this host, so it gets a synthetic one at the parity widths.
    Three things are actually at stake and all three are invisible until a real restore:

      * the `<root>/{video,action,proprio_encoder}` split and the dotted subtree names;
      * the list-index key spelling — `save_params` nests `blocks.0.…` under the STRING
        key "0" (flax 0.10's, and what the published tree uses) while flax 0.12's
        `to_pure_dict` hands back ints. Orbax calls that a structure mismatch on a tree
        that is identical;
      * the dtype cast, since deployment restores bf16 targets.
    """
    from fastwam_tpu.inference.load import LoadError, restore_into
    from tools.convert_weights import save_params

    def flat(module):
        return {
            ".".join(str(part) for part in path): np.asarray(value[...])
            for path, value in nnx.to_flat_state(nnx.state(module, nnx.Param))
        }

    source = build_fastwam()
    root = tmp_path / "official_uncond"
    for name, module in (
        ("video", source.mot.video),
        ("action", source.mot.action),
        ("proprio_encoder", source.proprio_encoder),
    ):
        save_params(flat(module), str(root / name))

    subtrees = {
        "mot.video": str(root / "video"),
        "mot.action": str(root / "action"),
        "proprio_encoder": str(root / "proprio_encoder"),
    }
    target = FastWAM(small_model_config(), rngs=nnx.Rngs(7))  # different rng = different weights
    assert not np.allclose(
        flat(target.mot.video)["blocks.0.attn1.query.kernel"],
        flat(source.mot.video)["blocks.0.attn1.query.kernel"],
    ), "the fixture must start from different weights or the test proves nothing"

    restored = restore_into(target, subtrees)
    for name, (got, want) in {
        "video": (flat(restored.mot.video), flat(source.mot.video)),
        "action": (flat(restored.mot.action), flat(source.mot.action)),
        "proprio": (flat(restored.proprio_encoder), flat(source.proprio_encoder)),
    }.items():
        assert set(got) == set(want), f"{name}: key sets differ"
        for key in got:
            assert np.array_equal(got[key], want[key]), f"{name}.{key} did not round-trip"
    print(f"[parity] restore_into: {len(flat(restored.mot.video))} + "
          f"{len(flat(restored.mot.action))} + {len(flat(restored.proprio_encoder))} arrays exact")

    # And the deployment spelling: a bf16 target casts on the way in.
    as_bf16 = restore_into(FastWAM(small_model_config(), rngs=nnx.Rngs(8)), subtrees,
                           param_dtype=jnp.bfloat16)
    assert as_bf16.mot.video.blocks[0].attn1.query.kernel.dtype == jnp.bfloat16

    # A subtree that does not cover the whole model must raise, not half-load.
    with pytest.raises(LoadError, match="cover"):
        restore_into(FastWAM(small_model_config(), rngs=nnx.Rngs(9)),
                     {"mot.video": str(root / "video")})
    with pytest.raises(LoadError, match="no parameter subtree"):
        restore_into(FastWAM(small_model_config(), rngs=nnx.Rngs(9)),
                     {"mot.nonexistent": str(root / "video")})


def test_restore_into_places_arrays_on_the_device(tmp_path):
    """Every restored array is committed to ONE device, with no host copy left behind.

    This exists because a size-based check cannot work at test scale. The real failure
    (2026-08-04) was `restore_into` handing orbax a `ShapeDtypeStruct` with no sharding,
    so every array was materialised host-side first and the staging was never released:
    27.3 GB of host RSS on the 6.02 B checkpoint, on top of a full copy on the GPU. At
    these widths the same bug leaks a few hundred MB and looks like nothing, so what is
    asserted is the PROPERTY that prevents it — arrays carry a concrete single-device
    sharding — not the symptom.

    Downstream that leak cost 300 s: it pushed the co-resident UMT5 encoder into swap and
    the eval client timed out on its first `set_task`, three layers from the cause.
    """
    from fastwam_tpu.inference.load import restore_into
    from tools.convert_weights import save_params

    def flat(module):
        return {
            ".".join(str(part) for part in path): np.asarray(value[...])
            for path, value in nnx.to_flat_state(nnx.state(module, nnx.Param))
        }

    source = build_fastwam()
    root = tmp_path / "official_uncond"
    for name, module in (
        ("video", source.mot.video),
        ("action", source.mot.action),
        ("proprio_encoder", source.proprio_encoder),
    ):
        save_params(flat(module), str(root / name))
    subtrees = {
        "mot.video": str(root / "video"),
        "mot.action": str(root / "action"),
        "proprio_encoder": str(root / "proprio_encoder"),
    }

    want_device = jax.devices()[0]
    restored = restore_into(FastWAM(small_model_config(), rngs=nnx.Rngs(11)), subtrees)

    arrays = [value[...] for _, value in nnx.to_flat_state(nnx.state(restored, nnx.Param))]
    assert arrays, "no parameters to check"
    for array in arrays:
        assert isinstance(array, jax.Array)
        assert not array.is_deleted(), "a restored buffer was left deleted"
        # `is_fully_addressable` + a single-device sharding is what "it lives on that
        # device, not in a host staging buffer" looks like from the API.
        assert array.sharding.device_set == {want_device}, (
            f"array landed on {array.sharding.device_set}, expected {{{want_device}}} — "
            "restore_into lost its SingleDeviceSharding and is staging host-side again"
        )
        assert len(array.devices()) == 1

    # And an explicit device argument is honoured, which is what a multi-GPU server would use.
    explicit = restore_into(
        FastWAM(small_model_config(), rngs=nnx.Rngs(12)), subtrees, device=jax.devices()[-1]
    )
    probe = nnx.state(explicit, nnx.Param)["proprio_encoder"]["kernel"][...]
    assert probe.sharding.device_set == {jax.devices()[-1]}
    print(f"[parity] restore_into placement: {len(arrays)} arrays on {want_device}")


def test_load_fastwam_reads_the_training_whole_tree_layout(tmp_path):
    """The OTHER on-disk form: a training run's bf16 snapshot through the layout sniffing.

    `load_fastwam` serves two mutually exclusive directory shapes. The released one —
    `<root>/{video,action,proprio_encoder}`, blocks unrolled — is covered above. This is
    the shape every SELF-trained checkpoint takes: `save_params_bf16` writes
    `nnx.to_pure_dict` of a `scan_layers=True` model as ONE StandardCheckpointer tree
    (blocks stacked on a leading n_layers axis), and `load_fastwam` must sniff that
    layout off the directory, build a scanned model, and hand `restore_into` the
    whole-tree spelling (`subtrees={"": root}`).

    Until 2026-08-05 none of that had a test: the branch was only ever "verified" by
    loading without an exception, and the first checkpoints served through it scored 0%
    in closed loop — which cost a manual audit (85/85 arrays md5-identical against a raw
    orbax read of the real v6e snapshot) to exonerate. This is that audit, pinned at
    parity widths so it runs on every box.
    """
    import orbax.checkpoint as ocp

    from fastwam_tpu import load_config
    from fastwam_tpu.inference.load import load_fastwam

    # `load_fastwam` wants the whole Config (it pins derived geometry); the parity
    # widths ride in through the `model` field.
    config = dataclasses.replace(load_config(FULL_CONFIG), model=small_model_config())

    def flat(module):
        return {
            ".".join(str(part) for part in path): value[...]
            for path, value in nnx.to_flat_state(nnx.state(module, nnx.Param))
        }

    # Write the snapshot exactly the way training does (train.py -> save_params_bf16):
    # to_pure_dict of the Param state, cast to bf16, saved as one tree.
    source = FastWAM(small_model_config(), rngs=nnx.Rngs(21), scan_layers=True)
    pure = nnx.to_pure_dict(nnx.state(source, nnx.Param))
    snapshot = jax.tree.map(lambda x: x.astype(jnp.bfloat16), pure)
    root = tmp_path / "params_bf16_step0000010"
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(str(root), snapshot)
    ckptr.wait_until_finished()

    # Guard against the same-seed trap: a loader that silently kept its own init must
    # not be able to pass the equality below.
    fresh = FastWAM(small_model_config(), rngs=nnx.Rngs(0), scan_layers=True)
    assert not np.allclose(
        np.asarray(flat(fresh)["mot.video.blocks.attn1.query.kernel"], np.float32),
        np.asarray(flat(source)["mot.video.blocks.attn1.query.kernel"], np.float32),
    ), "the fixture must start from different weights or the test proves nothing"

    restored = load_fastwam(str(root), config)

    # The sniff must have produced the SCANNED build: scanned flat keys carry no
    # per-layer index ("blocks.attn1…", not "blocks.0.attn1…").
    got = flat(restored)
    want = {key: value.astype(jnp.bfloat16) for key, value in flat(source).items()}
    assert not any(".blocks.0." in key for key in got), (
        "load_fastwam built an UNROLLED model for a training snapshot; the restore only "
        "survives by stacking on-device, which does not fit the deployment card"
    )
    assert set(got) == set(want), "the sniffed scanned build does not mirror the training tree"
    for key in sorted(want):
        assert got[key].dtype == jnp.bfloat16, f"{key}: restored as {got[key].dtype}"
        assert got[key].shape == want[key].shape, (
            f"{key}: {got[key].shape} != {want[key].shape} — a stacked/unrolled mix-up"
        )
        assert np.array_equal(np.asarray(got[key]), np.asarray(want[key])), (
            f"{key} did not survive the save->sniff->restore round-trip"
        )
    print(f"[parity] load_fastwam whole-tree: {len(got)} arrays exact through the scanned path")
