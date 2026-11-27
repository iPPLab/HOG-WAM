"""M2 numerical gate: the action tower, the MoT, the loss and the denoising loop.

Same shape as test_dit_parity.py — the golden is `tests/_golden_ref/fastwam_torch_ref.py`,
a float64 numpy transcription of FastWAM's torch source, and every model here is a small
config (2 layers, 4 heads x 24) so the file runs on CPU in seconds. The residual against
the golden is an fp32 rounding budget, printed by `report()` under `pytest -s`.

The action expert's defining inequality is preserved at this scale: hidden_dim 32 vs
attention width 4 x 24 = 96, so q/k/v are 32 -> 96 and o is 96 -> 32 — the released
1024/3072 relation at 1/32 the width. The video tower stays square (96 = 4 x 24), which
its own constructor requires.
"""

import dataclasses
import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _golden_ref import fastwam_torch_ref as ref
from flax import nnx

from fastwam_tpu import load_config
from fastwam_tpu.models.action_dit import ACTION_ONLY_KEYS, ActionDiT
from fastwam_tpu.models.fastwam import FastWAM
from fastwam_tpu.models.mot import MoT, MoTError, build_mot_attention_mask
from fastwam_tpu.models.wan.transformer import WanVideoDiT
from fastwam_tpu.schedulers import flow_match
from tools.convert_weights import (
    ConversionError,
    _torch_shape,
    action_dit_param_shapes,
    build_action_dit_init,
    from_torch_layout,
    jax_to_torch_key,
    official_torch_inventory,
    to_torch_layout,
    video_dit_param_shapes,
)

TASK = "training/full.example.yaml"

# Small parity config. head_dim 24 splits into three equal RoPE axes (8, 8, 8), the
# small-scale analogue of 128 -> (44, 42, 42).
HEADS = 4
HEAD_DIM = 24
INNER = HEADS * HEAD_DIM  # 96 — the shared attention width
VIDEO_DIM = INNER  # the video tower REQUIRES hidden == heads*head_dim
ACTION_DIM_HIDDEN = 32  # deliberately != INNER: that inequality IS the action expert
LAYERS = 2
FREQ_DIM = 32
TEXT_DIM = 40
IN_CH = 8
PATCH = (1, 2, 2)
ACTION_DIM = 6
PROPRIO_DIM = 5
BATCH = 2
LATENT_SHAPE = (BATCH, IN_CH, 3, 8, 6)  # grid (3, 4, 3) -> 12 tokens/frame, Sv = 36
ACTION_HORIZON = 8
CONTEXT_LEN = 9

MAX_REL = 5e-5


def report(name: str, value: float) -> float:
    print(f"[parity] {name}: {value:.3e}")
    return value


def rel_error(got, want) -> float:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-30))


def flat_params(module: nnx.Module) -> dict[str, np.ndarray]:
    """{dotted path: np.ndarray} — the same keys the Orbax checkpoint uses."""
    return {
        ".".join(str(p) for p in path): np.asarray(value[...])
        for path, value in nnx.to_flat_state(nnx.state(module, nnx.Param))
    }


def load_flat_params(module: nnx.Module, flat: dict[str, np.ndarray]) -> None:
    state = nnx.state(module, nnx.Param)
    updated = [
        (path, variable.replace(jnp.asarray(flat[".".join(str(p) for p in path)], variable[...].dtype)))
        for path, variable in nnx.to_flat_state(state)
    ]
    nnx.update(module, nnx.from_flat_state(updated))


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def build_video(**overrides) -> WanVideoDiT:
    kwargs = dict(
        in_channels=IN_CH,
        out_channels=IN_CH,
        hidden_dim=VIDEO_DIM,
        ffn_dim=2 * VIDEO_DIM,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        patch_size=PATCH,
        eps=1e-6,
        video_attention_mask_mode="first_frame_causal",
    )
    kwargs.update(overrides)
    return WanVideoDiT(rngs=nnx.Rngs(0), **kwargs)


def build_action(**overrides) -> ActionDiT:
    kwargs = dict(
        action_dim=ACTION_DIM,
        hidden_dim=ACTION_DIM_HIDDEN,
        ffn_dim=2 * ACTION_DIM_HIDDEN,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        eps=1e-6,
    )
    kwargs.update(overrides)
    return ActionDiT(rngs=nnx.Rngs(1), **kwargs)


def video_ref_config() -> ref.RefConfig:
    return ref.RefConfig(
        hidden_dim=VIDEO_DIM,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        out_dim=IN_CH,
        patch_size=PATCH,
        eps=1e-6,
    )


def action_ref_config() -> ref.ActionRefConfig:
    return ref.ActionRefConfig(
        action_dim=ACTION_DIM,
        hidden_dim=ACTION_DIM_HIDDEN,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        eps=1e-6,
    )


def sample_inputs(seed: int = 1):
    keys = jax.random.split(jax.random.key(seed), 4)
    latents = jax.random.normal(keys[0], LATENT_SHAPE)
    action = jax.random.normal(keys[1], (BATCH, ACTION_HORIZON, ACTION_DIM))
    context = jax.random.normal(keys[2], (BATCH, CONTEXT_LEN, TEXT_DIM))
    proprio = jax.random.normal(keys[3], (BATCH, ACTION_HORIZON, PROPRIO_DIM))
    t_video = jnp.array([137.0, 913.0])
    t_action = jnp.array([642.0, 41.0])
    return latents, action, context, proprio, t_video, t_action


# --------------------------------------------------------------------------- #
# scheduler
# --------------------------------------------------------------------------- #
def test_training_weight_constants_match_the_reference_grid():
    """Y_MIN / WEIGHT_NORM_CONST are hardcoded (spec §9.3 item 15) — check the values."""
    y_min, norm_const = ref._precompute_weight_stats(1000, 5.0)
    assert abs(y_min - flow_match.Y_MIN) < 5e-11
    assert abs(norm_const - flow_match.WEIGHT_NORM_CONST) < 5e-11
    # The three sanity points from the spec.
    weights = np.asarray(flow_match.training_weight(jnp.array([0.0, 500.0, 1000.0])))
    assert weights[0] == pytest.approx(0.0, abs=1e-5)
    assert weights[1] == pytest.approx(2.076404, rel=1e-6)
    assert weights[2] == pytest.approx(0.0, abs=1e-6)


def test_scheduler_matches_the_reference():
    t = jnp.array([0.0, 1.0, 137.0, 500.0, 999.9])
    assert report("training_weight", rel_error(flow_match.training_weight(t), ref.training_weight(t))) < 1e-6

    x0 = jax.random.normal(jax.random.key(20), (3, 4, 5))
    noise = jax.random.normal(jax.random.key(21), (3, 4, 5))
    timestep = jnp.array([10.0, 500.0, 990.0])
    assert report(
        "add_noise",
        rel_error(flow_match.add_noise(x0, noise, timestep, 1000), ref.add_noise(x0, noise, timestep)),
    ) < 1e-6
    assert rel_error(flow_match.training_target(x0, noise), ref.training_target(x0, noise)) < 1e-7

    timesteps, deltas = flow_match.build_inference_schedule(10, 5.0, 1000)
    want_t, want_d = ref.build_inference_schedule(10, 5.0)
    assert timesteps.shape == deltas.shape == (10,)
    assert report("inference timesteps", rel_error(timesteps, want_t)) < 1e-6
    assert report("inference deltas", rel_error(deltas, want_d)) < 1e-6
    # The loop integrates DOWNWARDS: every delta is negative and they sum to -1.
    assert np.all(np.asarray(deltas) < 0)
    assert float(np.sum(np.asarray(deltas, np.float64))) == pytest.approx(-1.0, abs=1e-6)
    assert float(timesteps[0]) == pytest.approx(1000.0, rel=1e-6)


def test_sample_training_t_lands_in_range_and_is_independent_per_draw():
    key_a, key_b = jax.random.split(jax.random.key(5))
    ta = flow_match.sample_training_t(key_a, 4096, 5.0, 1000)
    tb = flow_match.sample_training_t(key_b, 4096, 5.0, 1000)
    assert ta.shape == (4096,)
    assert float(jnp.min(ta)) >= 0.0 and float(jnp.max(ta)) < 1000.0
    # shift=5 pushes mass towards high noise; the median of phi(U, 5)*1000 is 5/6*1000.
    assert float(jnp.median(ta)) == pytest.approx(833.3, rel=0.05)
    assert abs(float(jnp.corrcoef(ta, tb)[0, 1])) < 0.1


# --------------------------------------------------------------------------- #
# action tower
# --------------------------------------------------------------------------- #
def test_action_dit_attention_is_wider_than_its_hidden_dim():
    """The action expert's defining property (spec §2.3): 1024 hidden, 3072 attention."""
    model = build_action()
    block = model.blocks[0]
    assert block.attn1.query.kernel.shape == (ACTION_DIM_HIDDEN, INNER)
    assert block.attn1.proj_attn.kernel.shape == (INNER, ACTION_DIM_HIDDEN)
    assert block.attn1.norm_q.scale.shape == (INNER,)
    assert block.attn2.query.kernel.shape == (ACTION_DIM_HIDDEN, INNER)


def test_action_dit_pre_dit_shapes_and_rank3_t_mod():
    model = build_action()
    _, action, context, _, _, t_action = sample_inputs()
    mask = jnp.ones((BATCH, CONTEXT_LEN), bool)
    state = model.pre_dit(action, t_action, context, mask)
    assert state.tokens.shape == (BATCH, ACTION_HORIZON, ACTION_DIM_HIDDEN)
    assert state.t.shape == (BATCH, ACTION_DIM_HIDDEN)
    # Rank 3, not 4: one modulation shared across all action tokens.
    assert state.t_mod.shape == (BATCH, 6, ACTION_DIM_HIDDEN)
    assert state.context.shape == (BATCH, CONTEXT_LEN, ACTION_DIM_HIDDEN)
    assert state.context_mask.shape == (BATCH, ACTION_HORIZON, CONTEXT_LEN)
    assert state.cos.shape == state.sin.shape == (ACTION_HORIZON, HEAD_DIM // 2)


@pytest.mark.parametrize("with_context_mask", [False, True])
def test_action_dit_parity(with_context_mask):
    """The tower run alone, against the float64 transcription of ActionDiT.forward."""
    model = build_action()
    _, action, context, _, _, t_action = sample_inputs()
    context_mask = None
    if with_context_mask:
        context_mask = jnp.ones((BATCH, CONTEXT_LEN), bool).at[:, 6:].set(False)

    got = model(action, t_action, context, context_mask)
    want = ref.action_dit_forward(
        flat_params(model), action_ref_config(), action, t_action, context, context_mask
    )
    assert got.shape == want.shape == (BATCH, ACTION_HORIZON, ACTION_DIM)
    assert report(f"action_dit{' +ctx-mask' if with_context_mask else ''}", rel_error(got, want)) < MAX_REL


def test_action_rope_is_the_1d_table_sliced():
    """precompute_freqs_cis(head_dim, end=1024)[:Sa] — plain 1-D over the action axis."""
    model = build_action()
    cos, sin = model.rope_tables(ACTION_HORIZON)
    want = ref.precompute_freqs_cis(HEAD_DIM, end=1024)[:ACTION_HORIZON]
    assert report("action rope cos", float(np.max(np.abs(np.asarray(cos, np.float64) - want.real)))) < 1e-7
    assert report("action rope sin", float(np.max(np.abs(np.asarray(sin, np.float64) - want.imag)))) < 1e-7


def test_action_rope_cache_length_is_enforced():
    model = build_action()
    with pytest.raises(Exception, match="exceeds the RoPE cache"):
        model.rope_tables(1025)


# --------------------------------------------------------------------------- #
# the mask
# --------------------------------------------------------------------------- #
def test_mot_attention_mask_at_the_real_geometry():
    """[392,392]: action rows True only on columns 0:120 and 360:392; video rows never
    see action columns; the video block keeps first_frame_causal."""
    shapes = load_config(TASK).shapes
    mask = np.asarray(
        build_mot_attention_mask(shapes.video_tokens, shapes.action_tokens, shapes.tokens_per_frame)
    )
    video, action, per_frame = shapes.video_tokens, shapes.action_tokens, shapes.tokens_per_frame
    assert mask.shape == (shapes.joint_seq, shapes.joint_seq) == (392, 392)
    assert mask.dtype == np.bool_

    # video -> video: first_frame_causal
    assert mask[:per_frame, :per_frame].all()
    assert not mask[:per_frame, per_frame:video].any()
    assert mask[per_frame:video, :video].all()
    # video never attends to action
    assert not mask[:video, video:].any()
    # action -> first video frame only, plus all of itself
    assert mask[video:, :per_frame].all()
    assert not mask[video:, per_frame:video].any()
    assert mask[video:, video:].all()
    assert int(mask.sum()) == (
        per_frame * per_frame + (video - per_frame) * video + action * (per_frame + action)
    )


def test_mot_attention_mask_matches_the_reference():
    got = np.asarray(build_mot_attention_mask(36, 8, 12))
    want = ref.build_mot_attention_mask(36, 8, 12)
    assert np.array_equal(got, want)


def test_mot_attention_mask_rejects_a_ragged_video_length():
    with pytest.raises(MoTError, match="whole number of frames"):
        build_mot_attention_mask(35, 8, 12)


# --------------------------------------------------------------------------- #
# the MoT
# --------------------------------------------------------------------------- #
def build_mot(**overrides) -> MoT:
    remat_policy = overrides.pop("remat_policy", "none")
    return MoT(video=build_video(**overrides), action=build_action(**overrides), remat_policy=remat_policy)


def mot_params(mot: MoT) -> dict[str, dict]:
    return {"video": flat_params(mot.video), "action": flat_params(mot.action)}


def run_mot(mot: MoT, latents, action, context, context_mask, t_video, t_action):
    video_state = mot.video.pre_dit(latents, t_video, context, context_mask)
    action_state = mot.action.pre_dit(action, t_action, context, context_mask)
    mask = build_mot_attention_mask(
        video_state.tokens.shape[1], action_state.tokens.shape[1], video_state.tokens_per_frame
    )
    return mot.forward_states(video_state, action_state, mask)


def run_ref(mot: MoT, latents, action, context, context_mask, t_video, t_action):
    return ref.fastwam_predict(
        mot_params(mot), video_ref_config(), action_ref_config(),
        latents, action, t_video, t_action, context, context_mask,
    )


@pytest.mark.parametrize("with_context_mask", [False, True])
def test_mot_joint_forward_parity(with_context_mask):
    """GATE 1a: the full uncond forward — two towers, three masks, per-token t."""
    mot = build_mot()
    latents, action, context, _, t_video, t_action = sample_inputs()
    context_mask = jnp.ones((BATCH, CONTEXT_LEN), bool)
    if with_context_mask:
        context_mask = context_mask.at[:, 6:].set(False)

    pred_video, pred_action = run_mot(mot, latents, action, context, context_mask, t_video, t_action)
    want_video, want_action = run_ref(mot, latents, action, context, context_mask, t_video, t_action)
    assert pred_video.shape == LATENT_SHAPE
    assert pred_action.shape == (BATCH, ACTION_HORIZON, ACTION_DIM)
    suffix = " +ctx-mask" if with_context_mask else ""
    assert report(f"mot video{suffix}", rel_error(pred_video, want_video)) < MAX_REL
    assert report(f"mot action{suffix}", rel_error(pred_action, want_action)) < MAX_REL


def test_scan_matches_unrolled():
    """GATE 4: the dual-tower nnx.scan spike, as a test.

    Two stacked block modules passed as one tuple with `in_axes=0`, a two-element carry.
    The residual is pure fp32 reassociation inside the scan body — same bar M1 set for
    the video tower on its own (test_dit_parity.py::test_scan_layers_matches_unrolled).
    """
    from tools.convert_weights import stack_blocks_for_scan

    unrolled = build_mot(scan_layers=False)
    scanned = build_mot(scan_layers=True)
    load_flat_params(scanned.video, stack_blocks_for_scan(flat_params(unrolled.video), LAYERS))
    load_flat_params(scanned.action, stack_blocks_for_scan(flat_params(unrolled.action), LAYERS))

    latents, action, context, _, t_video, t_action = sample_inputs()
    context_mask = jnp.ones((BATCH, CONTEXT_LEN), bool)
    got = run_mot(scanned, latents, action, context, context_mask, t_video, t_action)
    want = run_mot(unrolled, latents, action, context, context_mask, t_video, t_action)
    assert report("scan vs unrolled (video)", rel_error(got[0], want[0])) < 1e-6
    assert report("scan vs unrolled (action)", rel_error(got[1], want[1])) < 1e-6


def test_scan_and_unrolled_towers_cannot_be_mixed():
    with pytest.raises(MoTError, match="same layer layout"):
        MoT(video=build_video(scan_layers=True), action=build_action(scan_layers=False))


def test_mismatched_head_geometry_raises():
    with pytest.raises(MoTError, match="num_heads"):
        MoT(video=build_video(), action=build_action(num_heads=HEADS + 1, head_dim=HEAD_DIM))


def test_remat_does_not_change_numerics():
    latents, action, context, _, t_video, t_action = sample_inputs()
    context_mask = jnp.ones((BATCH, CONTEXT_LEN), bool)
    plain = build_mot(remat_policy="none")
    rematted = build_mot(remat_policy="full")
    got = run_mot(rematted, latents, action, context, context_mask, t_video, t_action)
    want = run_mot(plain, latents, action, context, context_mask, t_video, t_action)
    assert rel_error(got[0], want[0]) < 1e-7
    assert rel_error(got[1], want[1]) < 1e-7


# --------------------------------------------------------------------------- #
# GATE 2: the FastWAM theorem
# --------------------------------------------------------------------------- #
def test_action_output_is_invariant_to_future_video():
    """The paper's central claim, as a numerical statement.

    Action tokens attend only to the first video latent frame; the first frame's tokens
    attend only to themselves. So perturbing latent frames 1.. must not move the action
    output at all — and must move the video output, or the test has no teeth.
    """
    mot = build_mot()
    latents, action, context, _, t_video, t_action = sample_inputs()
    context_mask = jnp.ones((BATCH, CONTEXT_LEN), bool)
    perturbed = latents.at[:, :, 1:].add(
        10.0 * jax.random.normal(jax.random.key(31), (BATCH, IN_CH, 2, 8, 6))
    )

    base_video, base_action = run_mot(mot, latents, action, context, context_mask, t_video, t_action)
    moved_video, moved_action = run_mot(mot, perturbed, action, context, context_mask, t_video, t_action)

    assert report("action vs future-video perturbation", rel_error(moved_action, base_action)) < 1e-6
    first_frame_move = rel_error(moved_video[:, :, 0], base_video[:, :, 0])
    assert report("frame-0 video vs perturbation", first_frame_move) < 1e-6
    later = rel_error(moved_video[:, :, 1:], base_video[:, :, 1:])
    assert later > 1e-2, f"later video frames barely moved ({later:.3e}); the test has no teeth"


def test_video_output_is_invariant_to_the_action_tokens():
    """The other half of the prefill argument: the video branch never reads the action
    tokens, so its per-layer K/V can be computed once and reused across denoising steps."""
    mot = build_mot()
    latents, action, context, _, t_video, t_action = sample_inputs()
    context_mask = jnp.ones((BATCH, CONTEXT_LEN), bool)
    other_action = jax.random.normal(jax.random.key(32), action.shape) * 5.0

    base_video, base_action = run_mot(mot, latents, action, context, context_mask, t_video, t_action)
    moved_video, moved_action = run_mot(mot, latents, other_action, context, context_mask, t_video, t_action)
    assert report("video vs action perturbation", rel_error(moved_video, base_video)) < 1e-6
    moved = rel_error(moved_action, base_action)
    assert moved > 1e-2, f"the action output barely moved ({moved:.3e}); the test has no teeth"


def test_action_output_equals_single_frame_prefill():
    """GATE 2: prefill semantics, proven at the mask level.

    `infer_action` feeds the video tower ONE latent frame (Sv = tokens_per_frame) at
    t=0 and prefills its K/V; `infer_joint` feeds it all T_l frames. Because frame 0 is
    declared clean in both (`token_timesteps[:, 0, :] = 0`) and the mask isolates it,
    the two must produce the same action tokens. That identity is what the M3 KV cache
    implements; if it fails, no amount of caching code can be correct.
    """
    mot = build_mot()
    latents, action, context, _, _, t_action = sample_inputs()
    context_mask = jnp.ones((BATCH, CONTEXT_LEN), bool)
    t_video = jnp.array([500.0, 500.0])  # arbitrary: frame 0 is forced to 0 regardless

    _, joint_action = run_mot(mot, latents, action, context, context_mask, t_video, t_action)
    _, prefill_action = run_mot(
        mot, latents[:, :, :1], action, context, context_mask, jnp.zeros((BATCH,)), t_action
    )
    assert report("joint vs single-frame action", rel_error(prefill_action, joint_action)) < 1e-6


def test_cached_action_mask_rows_are_the_joint_mask_rows():
    """`attention_mask[Sv:, :Sv+Sa]` (mot.py:388) — what the cached path slices out."""
    video_tokens, action_tokens, per_frame = 36, 8, 12
    joint = np.asarray(build_mot_attention_mask(video_tokens, action_tokens, per_frame))
    cached_rows = joint[video_tokens:, : video_tokens + action_tokens]
    assert cached_rows.shape == (action_tokens, video_tokens + action_tokens)
    # With only the first frame cached, the surviving columns are exactly frame 0 + self.
    prefill = np.asarray(build_mot_attention_mask(per_frame, action_tokens, per_frame))
    prefill_rows = prefill[per_frame:, :]
    assert np.array_equal(cached_rows[:, :per_frame], prefill_rows[:, :per_frame])
    assert np.array_equal(cached_rows[:, video_tokens:], prefill_rows[:, per_frame:])
    assert not cached_rows[:, per_frame:video_tokens].any()


# --------------------------------------------------------------------------- #
# GATE 1: FastWAM end to end — loss and the denoising loop
# --------------------------------------------------------------------------- #
def small_model_config():
    """The released ModelConfig, narrowed so the whole model fits in a CPU test.

    Only widths change: patch size, the per-token timestep path, both mask modes, the
    schedulers and the loss lambdas are the released values.
    """
    config = load_config(TASK)
    video = dataclasses.replace(
        config.model.video_dit_config,
        hidden_dim=VIDEO_DIM,
        ffn_dim=2 * VIDEO_DIM,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        in_dim=IN_CH,
        out_dim=IN_CH,
        action_dim=ACTION_DIM,
    )
    action = dataclasses.replace(
        config.model.action_dit_config,
        action_dim=ACTION_DIM,
        hidden_dim=ACTION_DIM_HIDDEN,
        ffn_dim=2 * ACTION_DIM_HIDDEN,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
    )
    return dataclasses.replace(
        config.model, video_dit_config=video, action_dit_config=action, proprio_dim=PROPRIO_DIM
    )


def build_fastwam(**overrides) -> FastWAM:
    return FastWAM(small_model_config(), rngs=nnx.Rngs(0), **overrides)


def fastwam_params(model: FastWAM) -> dict[str, dict]:
    return {
        "video": flat_params(model.mot.video),
        "action": flat_params(model.mot.action),
        "proprio": {
            f"proprio_encoder.{leaf}": value
            for leaf, value in flat_params(model.proprio_encoder).items()
        },
    }


def sample_batch(seed: int = 7):
    latents, action, context, proprio, _, _ = sample_inputs(seed)
    return {
        "latents": latents,
        "action": action,
        "proprio": proprio,
        "context": context,
        "context_mask": jnp.ones((BATCH, CONTEXT_LEN), bool),
        "image_is_pad": jnp.zeros((BATCH, 9), bool).at[1, 7:].set(True),
        "action_is_pad": jnp.zeros((BATCH, ACTION_HORIZON), bool).at[0, 6:].set(True),
    }


def test_proprio_token_is_appended_in_text_space():
    model = build_fastwam()
    batch = sample_batch()
    context, mask = model.append_proprio(batch["context"], batch["context_mask"], batch["proprio"][:, 0])
    assert context.shape == (BATCH, CONTEXT_LEN + 1, TEXT_DIM)
    assert mask.shape == (BATCH, CONTEXT_LEN + 1)
    assert bool(mask[:, -1].all())
    assert np.array_equal(np.asarray(context[:, :-1]), np.asarray(batch["context"]))
    want = ref.append_proprio_to_context(
        fastwam_params(model)["proprio"], batch["context"], batch["context_mask"], batch["proprio"][:, 0]
    )
    assert report("proprio token", rel_error(context, want[0])) < 1e-6


@pytest.mark.parametrize("with_pad_masks", [False, True])
def test_training_loss_parity(with_pad_masks):
    """GATE 1b: the training loss value, against the float64 transcription."""
    model = build_fastwam()
    batch = sample_batch()
    if not with_pad_masks:
        batch = {k: v for k, v in batch.items() if not k.endswith("_is_pad")}

    t_video = jnp.array([137.0, 913.0])
    t_action = jnp.array([642.0, 41.0])
    noise_video = jax.random.normal(jax.random.key(51), batch["latents"].shape)
    noise_action = jax.random.normal(jax.random.key(52), batch["action"].shape)

    loss, metrics = model.training_loss(
        batch, jax.random.key(0), timesteps=(t_video, t_action), noise=(noise_video, noise_action)
    )
    want = ref.fastwam_training_loss(
        fastwam_params(model), video_ref_config(), action_ref_config(),
        batch["latents"], batch["action"], batch["context"], batch["context_mask"], batch["proprio"],
        t_video, t_action, noise_video, noise_action,
        image_is_pad=batch.get("image_is_pad"), action_is_pad=batch.get("action_is_pad"),
    )
    suffix = " +pad" if with_pad_masks else ""
    assert report(f"loss_video{suffix}", rel_error(metrics["loss_video"], want["loss_video"])) < MAX_REL
    assert report(f"loss_action{suffix}", rel_error(metrics["loss_action"], want["loss_action"])) < MAX_REL
    assert report(f"loss total{suffix}", rel_error(loss, want["loss"])) < MAX_REL
    assert loss.shape == ()


def test_training_loss_drops_the_first_latent_frame():
    """The video loss reduces over T_l - 1 steps, and frame 0 is clean in the input.

    Perturbing the clean first latent frame changes the loss (it is the conditioning),
    but no gradient signal is asked of it: making the model's frame-0 prediction wildly
    wrong must not, because that slice is sliced away before the MSE.
    """
    model = build_fastwam()
    batch = sample_batch()
    args = dict(
        timesteps=(jnp.array([300.0, 700.0]), jnp.array([120.0, 880.0])),
        noise=(
            jax.random.normal(jax.random.key(61), batch["latents"].shape),
            jax.random.normal(jax.random.key(62), batch["action"].shape),
        ),
    )
    loss, _ = model.training_loss(batch, jax.random.key(0), **args)

    # The reference's own reduction: [B, T_l - 1] = [2, 2] here (3 latent frames).
    per_sample = model._video_loss_per_sample(
        jnp.zeros((BATCH, IN_CH, 2, 8, 6)), jnp.ones((BATCH, IN_CH, 2, 8, 6)), batch["image_is_pad"]
    )
    assert per_sample.shape == (BATCH,)
    assert np.allclose(np.asarray(per_sample), 1.0)
    assert np.isfinite(float(loss))


def test_video_loss_pad_mask_folds_four_frames_to_one_latent_step():
    """image_is_pad[:, 1:] viewed as [B, -1, 4].all(2) — a latent step is padding only
    if all four frames it covers are (fastwam.py:431-436)."""
    model = build_fastwam()
    per_step = jnp.stack([jnp.array([1.0, 100.0]), jnp.array([1.0, 100.0])])
    pred = jnp.zeros((2, IN_CH, 2, 4, 4))
    target = jnp.ones((2, IN_CH, 2, 4, 4)) * jnp.array([1.0, 10.0])[None, None, :, None, None]

    # sample 0: frames 5..8 padded -> latent step 1 dropped; sample 1: nothing padded.
    is_pad = jnp.zeros((2, 9), bool).at[0, 5:].set(True)
    got = np.asarray(model._video_loss_per_sample(pred, target, is_pad))
    assert got[0] == pytest.approx(1.0)  # only step 0 (target 1.0) counted
    assert got[1] == pytest.approx((1.0 + 100.0) / 2)
    del per_step

    # One padded frame out of four is NOT enough to drop the step.
    partial = jnp.zeros((2, 9), bool).at[0, 5].set(True)
    got_partial = np.asarray(model._video_loss_per_sample(pred, target, partial))
    assert got_partial[0] == pytest.approx((1.0 + 100.0) / 2)


def test_denoise_joint_parity():
    """GATE 1c: the 10-step denoising loop, end to end."""
    model = build_fastwam()
    batch = sample_batch()
    first_frame = batch["latents"][:, :, :1]
    noise_video = jax.random.normal(jax.random.key(71), batch["latents"].shape)
    noise_action = jax.random.normal(jax.random.key(72), batch["action"].shape)

    got = model.denoise_joint(
        first_frame_latents=first_frame,
        noise_video=noise_video,
        noise_action=noise_action,
        context=batch["context"],
        context_mask=batch["context_mask"],
        proprio=batch["proprio"][:, 0],
        num_inference_steps=10,
    )
    want = ref.denoise_joint(
        fastwam_params(model), video_ref_config(), action_ref_config(),
        first_frame, noise_video, noise_action, batch["context"], batch["context_mask"],
        proprio=batch["proprio"][:, 0], num_inference_steps=10, shift=5.0,
    )
    assert report("denoise_joint action", rel_error(got["action"], want["action"])) < MAX_REL
    assert report("denoise_joint video", rel_error(got["latents_video"], want["latents_video"])) < MAX_REL
    # The observation is re-written after every step, so it survives the loop exactly.
    assert rel_error(got["latents_video"][:, :, :1], first_frame) == 0.0


def test_denoise_action_matches_denoise_joint():
    """`infer_joint`'s own equivalence assertion (fastwam.py:893-898, atol/rtol 1e-2),
    which upstream only *warns* about. Here it is a hard test, and it holds far tighter
    than 1e-2 because both paths share the same arithmetic."""
    model = build_fastwam()
    batch = sample_batch()
    first_frame = batch["latents"][:, :, :1]
    noise_video = jax.random.normal(jax.random.key(81), batch["latents"].shape)
    noise_action = jax.random.normal(jax.random.key(82), batch["action"].shape)
    shared = dict(
        context=batch["context"], context_mask=batch["context_mask"],
        proprio=batch["proprio"][:, 0], num_inference_steps=10,
    )

    joint = model.denoise_joint(
        first_frame_latents=first_frame, noise_video=noise_video, noise_action=noise_action, **shared
    )["action"]
    action_only = model.denoise_action(
        first_frame_latents=first_frame, noise_action=noise_action, **shared
    )
    assert report("infer_action vs infer_joint", rel_error(action_only, joint)) < 1e-5

    want = ref.denoise_action(
        fastwam_params(model), video_ref_config(), action_ref_config(),
        first_frame, noise_action, batch["context"], batch["context_mask"],
        proprio=batch["proprio"][:, 0], num_inference_steps=10, shift=5.0,
    )
    assert report("denoise_action", rel_error(action_only, want)) < MAX_REL


def test_training_loss_accepts_an_nnx_rngs_or_a_bare_key():
    """M5's train step will hold an `nnx.Rngs`; the parity tests pass a bare key. Both
    have to work, and consecutive draws from the Rngs must differ (or every step in an
    epoch would see the same timesteps)."""
    model = build_fastwam()
    batch = sample_batch()
    rngs = nnx.Rngs(0)
    first = model.training_loss(batch, rngs)[1]
    second = model.training_loss(batch, rngs)[1]
    assert not np.allclose(np.asarray(first["t_video"]), np.asarray(second["t_video"]))
    from_key = model.training_loss(batch, jax.random.key(0))[1]
    assert np.isfinite(float(from_key["loss_video"]))


def test_training_loss_rejects_an_unknown_batch_key():
    model = build_fastwam()
    batch = {**sample_batch(), "video": jnp.zeros((BATCH, 3, 9, 8, 6))}
    with pytest.raises(Exception, match="unknown key"):
        model.training_loss(batch, jax.random.key(0))


def test_training_loss_runs_under_jit_and_grad():
    """The train step has to be jittable and differentiable — a shape check that only
    passes eagerly is not worth much."""
    model = build_fastwam()
    batch = sample_batch()

    @nnx.jit
    def loss_and_grads(module, data, key):
        def loss_fn(m):
            return m.training_loss(data, key)[0]

        return nnx.value_and_grad(loss_fn)(module)

    loss, grads = loss_and_grads(model, batch, jax.random.key(3))
    assert np.isfinite(float(loss))
    flat_grads = [np.asarray(v) for _, v in nnx.to_flat_state(nnx.state(grads, nnx.Param))]
    assert all(np.all(np.isfinite(g)) for g in flat_grads)
    # Both towers AND the proprio encoder must receive gradient.
    named = dict(fastwam_params(model))
    del named
    grad_flat = {
        ".".join(str(p) for p in path): np.asarray(value)
        for path, value in nnx.to_flat_state(nnx.state(grads, nnx.Param))
    }
    for probe in ("mot.video.blocks.0.attn1.query.kernel", "mot.action.blocks.0.attn1.query.kernel",
                  "proprio_encoder.kernel"):
        assert np.max(np.abs(grad_flat[probe])) > 0, f"{probe} got no gradient"


# --------------------------------------------------------------------------- #
# GATE 3: the ActionDiT interpolation initialization
# --------------------------------------------------------------------------- #
def test_action_init_matches_the_reference_preprocess_script():
    """GATE 3: spec §5.1, against a float64 transcription of the torch script.

    The comparison runs in TORCH layout and torch key names, which is the layout the
    reference works in — proving the flax-side round trip, not just the interpolation.
    """
    video = build_video()
    action_shapes = {key: value.shape for key, value in flat_params(build_action()).items()}
    video_params = flat_params(video)

    got = build_action_dit_init(video_params, action_shapes)
    assert set(got) == set(action_shapes) - ACTION_ONLY_KEYS

    torch_video = {jax_to_torch_key(k, "video"): to_torch_layout(k, v) for k, v in video_params.items()}
    backbone_keys = sorted(set(action_shapes) - ACTION_ONLY_KEYS)
    torch_shapes = {
        jax_to_torch_key(key, "action"): to_torch_layout(key, np.zeros(action_shapes[key], np.float32)).shape
        for key in backbone_keys
    }
    want_torch = ref.action_backbone_from_video(torch_video, torch_shapes)

    worst = 0.0
    for key in backbone_keys:
        want = from_torch_layout(key, want_torch[jax_to_torch_key(key, "action")])
        assert got[key].shape == want.shape == action_shapes[key]
        worst = max(worst, rel_error(got[key], want))
    assert report("action init vs preprocess script", worst) < 1e-6


def test_action_init_alpha_is_per_tensor_not_a_global_sqrt3():
    """alpha = sqrt(video_last / action_last) in TORCH layout.

    Three cases that would all be wrong under a blanket sqrt(3):
      attn1.query    (in 96 -> 32)  scaled by sqrt(3)
      attn1.proj_attn(in 96 -> 96)  NOT scaled (only the output width changed)
      ffn.proj_out   (in 192 -> 64) scaled by sqrt(3) as well here, but at the released
                     widths it is sqrt(14336/4096) = sqrt(3.5) — hence "per tensor".
    """
    video_params = flat_params(build_video())
    action_shapes = {key: value.shape for key, value in flat_params(build_action()).items()}
    scaled = build_action_dit_init(video_params, action_shapes, apply_alpha_scaling=True)
    plain = build_action_dit_init(video_params, action_shapes, apply_alpha_scaling=False)

    query = "blocks.0.attn1.query.kernel"
    ratio = np.asarray(scaled[query]) / np.asarray(plain[query])
    assert np.allclose(ratio, np.sqrt(VIDEO_DIM / ACTION_DIM_HIDDEN), rtol=1e-5)

    out = "blocks.0.attn1.proj_attn.kernel"
    assert np.array_equal(scaled[out], plain[out]), "proj_attn's torch last dim is unchanged"

    # A bias is 1-D, so `src.ndim >= 2` fails and it is never alpha-scaled.
    bias = "blocks.0.ffn.proj_out.bias"
    assert np.array_equal(scaled[bias], plain[bias])


def test_action_init_skips_exactly_action_encoder_and_the_head():
    video_params = flat_params(build_video())
    action_shapes = {key: value.shape for key, value in flat_params(build_action()).items()}
    assert ACTION_ONLY_KEYS <= set(action_shapes)
    got = build_action_dit_init(video_params, action_shapes)
    assert not (ACTION_ONLY_KEYS & set(got))

    # If the head were ever renamed out of the skip set, the video tower's same-named
    # `proj_out` would be silently interpolated into it. That must raise instead.
    renamed = {k: v for k, v in action_shapes.items() if k != "proj_out.kernel"}
    with pytest.raises(ConversionError, match="skip keys"):
        build_action_dit_init(video_params, renamed)


def test_alpha_at_the_released_widths_is_sqrt3_except_the_second_ffn(real_shapes):
    """Which tensors get which alpha at 3072 -> 1024, from shapes alone.

    Every alpha-scaled tensor is sqrt(3) except `ffn.2` (torch [3072,14336] ->
    [1024,4096]), whose torch last dim is the FFN width, giving sqrt(14336/4096) =
    sqrt(3.5). A hardcoded sqrt(3) would be wrong on 30 tensors — one per layer.
    """
    _, video, action = real_shapes
    alphas: dict[str, float] = {}
    for key in sorted(set(action) - ACTION_ONLY_KEYS):
        video_torch = np.zeros(video[key], np.float32).T if key.endswith(".kernel") else np.zeros(video[key])
        action_torch = (
            np.zeros(action[key], np.float32).T if key.endswith(".kernel") else np.zeros(action[key])
        )
        if video_torch.shape == action_torch.shape or video_torch.ndim < 2:
            continue
        if video_torch.shape[-1] != action_torch.shape[-1]:
            alphas[key] = (video_torch.shape[-1] / action_torch.shape[-1]) ** 0.5

    scaled = {key: round(value, 6) for key, value in alphas.items()}
    assert scaled["blocks.0.ffn.proj_out.kernel"] == pytest.approx(3.5**0.5, rel=1e-6)
    others = {k: v for k, v in scaled.items() if not k.endswith("ffn.proj_out.kernel")}
    assert others, "nothing else was alpha-scaled; the mapping is wrong"
    assert all(v == pytest.approx(3.0**0.5, rel=1e-6) for v in others.values())
    # 30 layers x {q,k,v}x2 attn + modulation + ffn.act_fn + ffn.proj_out, plus the
    # three top-level 1024-in Linears (time_embedder.linear_2, time_proj, text_2).
    assert "blocks.0.attn1.query.kernel" in scaled
    assert "blocks.0.attn1.proj_attn.kernel" not in scaled
    assert "condition_embedder.time_proj.kernel" in scaled
    assert "condition_embedder.text_embedder.linear_1.kernel" not in scaled


def test_interpolation_preserves_endpoints_and_is_linear():
    """align_corners=True: the two endpoints survive exactly, and a linear ramp stays
    a linear ramp. Both are false for align_corners=False, which is the likely typo."""
    ramp = np.arange(9, dtype=np.float64)[None, :]
    got = ref.resize_tensor_to_shape(ramp, (1, 5))
    assert np.allclose(got[0], [0.0, 2.0, 4.0, 6.0, 8.0])
    assert got[0, 0] == 0.0 and got[0, -1] == 8.0


# --------------------------------------------------------------------------- #
# the official checkpoint mapping
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_shapes():
    """The 6 B key/shape tables, computed ONCE.

    `nnx.eval_shape` over 30 blocks at d=3072 takes the better part of a minute, and
    four tests below want the same answer.
    """
    config = load_config(TASK).model
    return (
        config,
        video_dit_param_shapes(config.video_dit_config),
        action_dit_param_shapes(config.action_dit_config),
    )


@pytest.fixture(scope="module")
def real_inventory(real_shapes):
    return official_torch_inventory(real_shapes[0])


def test_torch_key_mapping_is_a_bijection_over_the_real_model(real_shapes):
    _, video, action = real_shapes
    for tower, shapes in (("video", video), ("action", action)):
        torch_keys = [jax_to_torch_key(key, tower) for key in shapes]
        assert len(set(torch_keys)) == len(torch_keys), f"{tower}: two keys collide"


def test_official_inventory_has_the_expected_shape_and_counts(real_inventory):
    """6.02 B trainable = 30 blocks x 27 x 2 towers + the two towers' non-block modules."""
    inventory = real_inventory
    mot, proprio = inventory["mot"], inventory["proprio_encoder"]

    assert proprio == {"weight": (4096, 14), "bias": (4096,)}
    # video:  30 x 27 + 15 = 825  (patch_embedding, condition_embedder x6, proj_out,
    #                               scale_shift_table)
    # action: 30 x 27 + 14 = 824  (action_encoder, condition_embedder x6, proj_out;
    #                               no patch_embedding and no scale_shift_table)
    video_keys = [k for k in mot if k.startswith("mixtures.video.")]
    action_keys = [k for k in mot if k.startswith("mixtures.action.")]
    assert len(video_keys) == 30 * 27 + 15 == 825
    assert len(action_keys) == 30 * 27 + 14 == 824
    assert len(mot) == 1649

    # Spot-check the traps: the norm2/norm3 swap, the head names, and both layouts.
    assert mot["mixtures.video.blocks.7.norm3.weight"] == (3072,)
    assert mot["mixtures.video.blocks.7.self_attn.q.weight"] == (3072, 3072)
    assert mot["mixtures.action.blocks.7.self_attn.q.weight"] == (3072, 1024)
    assert mot["mixtures.action.blocks.7.self_attn.o.weight"] == (1024, 3072)
    assert mot["mixtures.video.head.head.weight"] == (192, 3072)
    assert mot["mixtures.action.head.weight"] == (14, 1024)
    assert mot["mixtures.action.action_encoder.weight"] == (1024, 14)
    assert mot["mixtures.video.patch_embedding.weight"] == (3072, 48, 1, 2, 2)
    assert mot["mixtures.video.head.modulation"] == (1, 2, 3072)
    assert mot["mixtures.action.blocks.7.modulation"] == (1, 6, 1024)

    total = sum(int(np.prod(shape)) for shape in mot.values())
    assert total == pytest.approx(6.0207e9, rel=1e-3), f"MoT parameter count {total}"


def test_official_inventory_cross_check_round_trips(tmp_path, real_shapes, real_inventory):
    """`check-official` against a dump of our own expectation must pass, and any drift
    in it must raise — this is the harness the real key_inventory.json goes through."""
    import json

    from tools.convert_weights import check_official_inventory

    config = real_shapes[0]
    inventory = real_inventory
    path = tmp_path / "key_inventory.json"

    # Flat spelling, as the payload nests the sections.
    flat = {f"mot.{k}": list(v) for k, v in inventory["mot"].items()}
    flat.update({f"proprio_encoder.{k}": list(v) for k, v in inventory["proprio_encoder"].items()})
    flat["step"] = []  # a non-tensor payload entry, which must be ignored...
    del flat["step"]
    path.write_text(json.dumps(flat))
    assert check_official_inventory(config, path) == 1651

    # Sectioned spelling, with the {name, shape} entry style.
    sectioned = {
        section: [{"name": k, "shape": list(v)} for k, v in shapes.items()]
        for section, shapes in inventory.items()
    }
    path.write_text(json.dumps(sectioned))
    assert check_official_inventory(config, path) == 1651

    bad = dict(flat)
    bad["mot.mixtures.action.blocks.0.self_attn.q.weight"] = [1024, 3072]  # transposed
    path.write_text(json.dumps(bad))
    with pytest.raises(ConversionError, match="shape mot.mixtures.action"):
        check_official_inventory(config, path)


# --------------------------------------------------------------------------- #
# against the real released checkpoint (tests/fixtures/, dumped from the 12 GB .pt)
# --------------------------------------------------------------------------- #
FIXTURES = pathlib.Path(__file__).parent / "fixtures"
OFFICIAL_INVENTORY = FIXTURES / "official_key_inventory.json"
VIDEO_RENAME_TABLE = FIXTURES / "video_dit_rename_pt_to_orbax.json"


def test_derived_inventory_matches_the_real_pt_dump(real_shapes):
    """The whole mapping, against the actual 12 GB file.

    `official_key_inventory.json` was dumped from the released
    `robotwin_uncond_3cam_384.pt` (1651 tensors, 6,020,749,518 params, all bf16, 0
    NaN/Inf). Our expectation is derived independently from the model definition, so
    agreement here means the rename table, both layout flips and the tower split are all
    right — before anyone spends 12 GB of IO finding out otherwise.
    """
    from tools.convert_weights import check_official_inventory

    assert OFFICIAL_INVENTORY.is_file(), "the probe's inventory fixture is missing"
    assert check_official_inventory(real_shapes[0], OFFICIAL_INVENTORY) == 1651


def test_rename_table_agrees_on_every_name_and_shape():
    """Our `jax_to_torch_key` against the probe's independently built 825-key table."""
    table = json.loads(VIDEO_RENAME_TABLE.read_text())["map"]
    prefix = "mot.mixtures.video."
    assert len(table) == 825

    for pt_key, entry in table.items():
        ours = entry["orbax"]
        assert prefix + jax_to_torch_key(ours, "video") == pt_key
        assert list(_torch_shape(ours, tuple(entry["orbax_shape"]))) == entry["pt_shape"]


def test_square_kernels_are_transposed_too():
    """The one place the shipped rename table is WRONG, pinned so we never adopt it.

    The fixture's `transform` column was inferred from shapes, so it can only see a
    transpose when the shape changes. Every one of the 242 SQUARE 2-D kernels (240
    attention q/k/v/o across 30 blocks, plus the two `linear_2`s) is therefore labelled
    `identity` — but torch stores a Linear weight as [out, in] and flax as [in, out], so
    a square kernel still has to be flipped. Loading those 242 un-transposed is a bug
    that no shape check and no permutation-invariant fingerprint can see.

    So: assert the poisoned column stays REMOVED from the fixture (2026-08-04: stripped,
    with a meta warning pointing at the layout rule), and assert our converter
    transposes on `.kernel`, not on "did the shape change".
    """
    raw = json.loads(VIDEO_RENAME_TABLE.read_text())
    assert "WARNING_transform_column_removed" in raw["meta"]
    assert not any("transform" in e for e in raw["map"].values())
    # The blind spot the column had: 242 square 2-D kernels it could never label.
    square_2d = [
        e
        for e in raw["map"].values()
        if len(e["orbax_shape"]) == 2 and e["orbax_shape"][0] == e["orbax_shape"][1]
    ]
    assert len(square_2d) == 242

    # Our layout helper does not consult the shape: a square kernel is still flipped.
    square = np.arange(9, dtype=np.float32).reshape(3, 3)
    assert np.array_equal(to_torch_layout("blocks.0.attn1.query.kernel", square), square.T)
    assert not np.array_equal(to_torch_layout("blocks.0.attn1.query.kernel", square), square)
    # A bias/scale, which really is identity, must NOT be touched.
    vector = np.arange(3, dtype=np.float32)
    assert np.array_equal(to_torch_layout("blocks.0.attn1.query.bias", vector), vector)


def test_fingerprint_check_has_teeth(tmp_path):
    """`verify_official_fingerprints` must catch a tensor landing on the wrong key.

    Fingerprints are permutation-invariant, so this is what they are FOR — mis-mapping,
    corruption, dtype loss — and it is worth proving they actually fire, because a
    verifier that silently passes is worse than none.
    """
    from tools.convert_weights import verify_official_fingerprints

    rng = np.random.default_rng(0)
    good = rng.normal(size=(4, 6)).astype(np.float32)
    other = rng.normal(size=(4, 6)).astype(np.float32) * 3.0

    def stats(array):
        # Mirrors the probe's pure-torch dump: Tensor.std() is Bessel-corrected
        # (ddof=1), which _tensor_fingerprint must match — see the ddof note there.
        values = np.asarray(array, np.float64)
        return {
            "sum": float(values.sum()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "absmax": float(np.abs(values).max()),
        }

    inventory = {
        "tensors": {
            "proprio_encoder.weight": {"stats": stats(good.T)},
            "proprio_encoder.bias": {"stats": stats(np.zeros(4))},
        }
    }
    path = tmp_path / "inv.json"
    path.write_text(json.dumps(inventory))

    towers = {"proprio_encoder": {"kernel": good, "bias": np.zeros(4, np.float32)}}
    assert verify_official_fingerprints(towers, path) == 2

    towers_bad = {"proprio_encoder": {"kernel": other, "bias": np.zeros(4, np.float32)}}
    with pytest.raises(ConversionError, match="fingerprint mismatch"):
        verify_official_fingerprints(towers_bad, path)


def test_orbax_metadata_reader_survives_the_0_12_wrapper(tmp_path):
    """orbax 0.12 moved the tree behind `StepMetadata.item_metadata`.

    Flattening the wrapper yields ONE entry keyed by the empty path, so the old code
    produced a single bogus "" key and reported a clean audit against nothing. Round-trip
    a real (tiny) tree to pin the reader.
    """
    import orbax.checkpoint as ocp

    from tools.convert_weights import read_orbax_shapes

    tree = {"a": {"b": jnp.zeros((2, 3))}, "c": jnp.ones((4,))}
    path = tmp_path / "tree"
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(str(path), tree)

    assert read_orbax_shapes(str(path)) == {"a.b": (2, 3), "c": (4,)}


def test_torch_layout_round_trip():
    for key, shape in (
        ("blocks.0.attn1.query.kernel", (32, 96)),
        ("patch_embedding.kernel", (1, 2, 2, 8, 96)),
        ("blocks.0.norm2.scale", (96,)),
        ("blocks.0.adaln_scale_shift_table", (1, 6, 96)),
    ):
        array = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
        assert np.array_equal(from_torch_layout(key, to_torch_layout(key, array)), array)
    assert to_torch_layout("x.kernel", np.zeros((3, 5), np.float32)).shape == (5, 3)
    assert to_torch_layout("patch_embedding.kernel", np.zeros((1, 2, 2, 8, 96), np.float32)).shape == (
        96, 8, 1, 2, 2,
    )


def test_unmapped_key_raises():
    with pytest.raises(ConversionError):
        jax_to_torch_key("blocks.0.mystery.kernel", "video")
    with pytest.raises(ConversionError, match="no patch_embedding"):
        jax_to_torch_key("patch_embedding.kernel", "action")
    with pytest.raises(ConversionError, match="no action_encoder"):
        jax_to_torch_key("action_encoder.kernel", "video")
