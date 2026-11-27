"""`flow.tau_copies` / `flow.copy_lengths`: M independent tau samples per video forward.

Four modules have to agree on one packing (`copies.py`): the config resolver, the joint
attention mask, the action tower's RoPE, and the trainer's target packing. None of the
ways they can disagree raises — a copy masked at one offset and rope'd at another has
entirely legal shapes — so the gates here are behavioural, not structural:

* **M = 1 is the released path, to the bit.** Not "close": the same arrays. If adding the
  feature moved the default numbers at all, every existing checkpoint and loss curve would
  become incomparable to anything trained afterwards.
* **Copies cannot see each other.** Asserted the way it actually matters — perturb one
  copy's input and its tau, and require every other copy's output, and the video's, to be
  unchanged bit for bit. A mask typo passes a shape check and fails this.
* **The config -> tuple translation is covered.** Upstream's copy tests all handed the
  model layer an explicit tuple, so the one line that turned `tau_copies=4` into a single
  128-row copy went unnoticed until it hit a TPU. That translation is tested here first.

Run:  JAX_PLATFORMS=cpu uv run pytest tests/test_tau_copies.py
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from fastwam_tpu import load_config
from fastwam_tpu.config import ConfigError
from fastwam_tpu.copies import CopyLengthError, normalize_copy_lengths, resolve_copy_lengths
from fastwam_tpu.models.fastwam import FastWAM
from fastwam_tpu.models.mot import build_mot_attention_mask
from fastwam_tpu.models.wan.transformer import video_self_attention_mask_numpy
from fastwam_tpu.schedulers import flow_match
from fastwam_tpu.training.train import flops_per_sample

TASK = "training/full.example.yaml"

# Same narrowing as test_mot_parity: only widths change. head_dim 24 splits into three
# equal RoPE axes, and the action tower keeps hidden != heads*head_dim, which is the
# inequality that makes it the action expert rather than a second video tower.
HEADS, HEAD_DIM = 4, 24
VIDEO_DIM = HEADS * HEAD_DIM
ACTION_HIDDEN = 32
LAYERS, FREQ_DIM, TEXT_DIM = 2, 32, 40
IN_CH, PATCH = 8, (1, 2, 2)
ACTION_DIM, PROPRIO_DIM = 6, 5
BATCH = 2
LATENT_SHAPE = (BATCH, IN_CH, 3, 8, 6)  # grid (3, 4, 3) -> 12 tokens/frame, Sv = 36
TOKENS_PER_FRAME, VIDEO_TOKENS = 12, 36
ACTION_HORIZON, CONTEXT_LEN = 8, 9


def small_model_config():
    config = load_config(TASK)
    video = dataclasses.replace(
        config.model.video_dit_config,
        hidden_dim=VIDEO_DIM, ffn_dim=2 * VIDEO_DIM, num_heads=HEADS, attn_head_dim=HEAD_DIM,
        num_layers=LAYERS, freq_dim=FREQ_DIM, text_dim=TEXT_DIM, in_dim=IN_CH, out_dim=IN_CH,
        action_dim=ACTION_DIM,
    )
    action = dataclasses.replace(
        config.model.action_dit_config,
        action_dim=ACTION_DIM, hidden_dim=ACTION_HIDDEN, ffn_dim=2 * ACTION_HIDDEN,
        num_heads=HEADS, attn_head_dim=HEAD_DIM, num_layers=LAYERS, freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
    )
    return dataclasses.replace(
        config.model, video_dit_config=video, action_dit_config=action, proprio_dim=PROPRIO_DIM
    )


def build_fastwam() -> FastWAM:
    return FastWAM(small_model_config(), rngs=nnx.Rngs(0))


def sample_batch(seed: int = 7):
    keys = jax.random.split(jax.random.key(seed), 4)
    return {
        "latents": jax.random.normal(keys[0], LATENT_SHAPE),
        "action": jax.random.normal(keys[1], (BATCH, ACTION_HORIZON, ACTION_DIM)),
        "proprio": jax.random.normal(keys[3], (BATCH, ACTION_HORIZON, PROPRIO_DIM)),
        "context": jax.random.normal(keys[2], (BATCH, CONTEXT_LEN, TEXT_DIM)),
        "context_mask": jnp.ones((BATCH, CONTEXT_LEN), bool),
        "image_is_pad": jnp.zeros((BATCH, 9), bool).at[1, 7:].set(True),
        "action_is_pad": jnp.zeros((BATCH, ACTION_HORIZON), bool).at[0, 6:].set(True),
    }


def rel_error(got, want) -> float:
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-30))


# --------------------------------------------------------------------------- #
# the config -> tuple translation (the layer upstream's tests missed)
# --------------------------------------------------------------------------- #
def test_tau_copies_means_m_full_copies_not_one_long_one():
    """The regression that reached a live slice: `tau_copies=4` is (32,)*4, never (128,).

    Both are tuples, both sum to 128, and only one of them is four copies. The wrong one
    trains a single 128-row chunk with RoPE running 0..127 — right shapes, no error, and a
    model nobody asked for.
    """
    assert resolve_copy_lengths(4, None, 32) == (32, 32, 32, 32)
    assert resolve_copy_lengths(1, None, 32) == (32,)
    assert sum(resolve_copy_lengths(4, None, 32)) == 128  # the shape that made it look fine


def test_explicit_lengths_are_prefixes_and_set_their_own_count():
    assert resolve_copy_lengths(1, [32, 16, 8, 4], 32) == (32, 16, 8, 4)
    assert resolve_copy_lengths(4, [32, 16, 8, 4], 32) == (32, 16, 8, 4)  # agreeing count is fine


@pytest.mark.parametrize(
    "tau_copies, lengths, horizon",
    [
        (2, [32, 16, 8, 4], 32),  # count disagrees with the list
        (1, [64], 32),  # a copy longer than the data's horizon
        (1, [32, 0], 32),  # an empty copy
        (0, None, 32),  # no copies at all
        (1, [], 32),  # an empty list is not "no opinion"; that is None
    ],
)
def test_impossible_layouts_raise(tau_copies, lengths, horizon):
    with pytest.raises(CopyLengthError):
        resolve_copy_lengths(tau_copies, lengths, horizon)


def test_normalize_demands_the_lengths_fill_the_stream():
    assert normalize_copy_lengths(None, 32) == (32,)
    assert normalize_copy_lengths([16, 16], 32) == (16, 16)
    with pytest.raises(CopyLengthError):
        normalize_copy_lengths([16, 8], 32)


def test_config_derives_the_tuple_and_the_training_geometry():
    base = load_config(TASK)
    assert base.shapes.copy_lengths == (32,)
    assert base.shapes.train_action_tokens == base.shapes.action_tokens == 32
    assert base.shapes.train_joint_seq == base.shapes.joint_seq == 392

    four = load_config(TASK, overrides=["flow.tau_copies=4"])
    assert four.shapes.copy_lengths == (32, 32, 32, 32)
    assert four.shapes.train_action_tokens == 128
    assert four.shapes.train_joint_seq == 360 + 128
    # the DATA geometry and everything inference sizes off it must not move
    assert four.shapes.action_tokens == 32
    assert four.shapes.prefill_joint_seq == base.shapes.prefill_joint_seq

    mix = load_config(TASK, overrides=["flow.copy_lengths=[32,16,8,4]"])
    assert mix.shapes.copy_lengths == (32, 16, 8, 4)
    assert mix.shapes.train_action_tokens == 60


def test_a_bad_flow_block_fails_at_config_load():
    """Not inside a jitted step, twenty minutes into a queued slice."""
    with pytest.raises(ConfigError):
        load_config(TASK, overrides=["flow.copy_lengths=[64]"])
    with pytest.raises(ConfigError):
        load_config(TASK, overrides=["flow.tau_copies=0"])


def test_the_two_spellings_may_not_disagree():
    """`tau_copies=2` beside a three-element list is two answers to one question."""
    with pytest.raises(ConfigError):
        load_config(TASK, overrides=["flow.tau_copies=2", "flow.copy_lengths=[32,32,32]"])
    # ...but the agreeing spelling is allowed, so the guard is not just "never both"
    both = load_config(TASK, overrides=["flow.tau_copies=3", "flow.copy_lengths=[32,32,32]"])
    assert both.shapes.copy_lengths == (32, 32, 32)


# --------------------------------------------------------------------------- #
# the mask
# --------------------------------------------------------------------------- #
def legacy_mask(video_tokens: int, action_tokens: int, tokens_per_frame: int) -> np.ndarray:
    """The pre-copies mask, transcribed here so the default cannot drift silently."""
    total = video_tokens + action_tokens
    mask = np.zeros((total, total), dtype=bool)
    mask[:video_tokens, :video_tokens] = video_self_attention_mask_numpy(
        tokens_per_frame, video_tokens // tokens_per_frame
    )
    mask[video_tokens:, video_tokens:] = True
    mask[video_tokens:, : min(tokens_per_frame, video_tokens)] = True
    return mask


def test_single_copy_mask_is_the_old_mask_bit_for_bit():
    want = legacy_mask(VIDEO_TOKENS, ACTION_HORIZON, TOKENS_PER_FRAME)
    for lengths in (None, (ACTION_HORIZON,)):
        got = np.asarray(
            build_mot_attention_mask(VIDEO_TOKENS, ACTION_HORIZON, TOKENS_PER_FRAME, lengths)
        )
        assert np.array_equal(got, want)


def test_copies_are_mutually_invisible_and_keep_every_other_rule():
    lengths = (8, 4, 2)
    sa = sum(lengths)
    mask = np.asarray(build_mot_attention_mask(VIDEO_TOKENS, sa, TOKENS_PER_FRAME, lengths))

    # the video block is untouched by copies
    assert np.array_equal(
        mask[:VIDEO_TOKENS, :VIDEO_TOKENS],
        video_self_attention_mask_numpy(TOKENS_PER_FRAME, VIDEO_TOKENS // TOKENS_PER_FRAME),
    )
    assert not mask[:VIDEO_TOKENS, VIDEO_TOKENS:].any(), "video must never attend action"

    # every action row: exactly its own copy plus the first video frame, nothing else
    offset = VIDEO_TOKENS
    for length in lengths:
        rows = slice(offset, offset + length)
        expected = np.zeros(mask.shape[1], dtype=bool)
        expected[:TOKENS_PER_FRAME] = True
        expected[offset : offset + length] = True
        for row in range(offset, offset + length):
            assert np.array_equal(mask[row], expected), f"row {row} sees the wrong set"
        # and no key outside this copy is visible from it
        assert not mask[rows, VIDEO_TOKENS:offset].any()
        assert not mask[rows, offset + length :].any()
        offset += length


# --------------------------------------------------------------------------- #
# the RoPE table
# --------------------------------------------------------------------------- #
def test_rope_restarts_at_zero_for_every_copy():
    tower = build_fastwam().mot.action
    base_cos, base_sin = tower.rope_tables(ACTION_HORIZON)

    # one copy: unchanged
    for lengths in (None, (ACTION_HORIZON,)):
        cos, sin = tower.rope_tables(ACTION_HORIZON, lengths)
        assert np.array_equal(np.asarray(cos), np.asarray(base_cos))
        assert np.array_equal(np.asarray(sin), np.asarray(base_sin))

    lengths = (8, 4, 2)
    cos, sin = tower.rope_tables(sum(lengths), lengths)
    assert cos.shape[0] == sum(lengths)
    offset = 0
    for length in lengths:
        # copy i occupies positions 0..length-1, i.e. the head of the single-copy table
        assert np.array_equal(np.asarray(cos[offset : offset + length]), np.asarray(base_cos[:length]))
        assert np.array_equal(np.asarray(sin[offset : offset + length]), np.asarray(base_sin[:length]))
        offset += length


# --------------------------------------------------------------------------- #
# the forward
# --------------------------------------------------------------------------- #
def test_one_copy_cannot_see_another_perturbing_it_changes_nothing():
    """The decisive test. Exact equality, not a tolerance.

    Masked logits are replaced by `finfo.min` before the softmax, so a masked key's
    contribution is exactly 0.0 and the sum over k has the same length either way. Any
    leak — through the mask, through a shared RoPE position, through a t_mod row landing
    on the wrong copy — makes this differ in the last bits, which a tolerance would hide.
    """
    model = build_fastwam()
    batch = sample_batch()
    lengths = (ACTION_HORIZON, ACTION_HORIZON)
    sa = sum(lengths)

    key = jax.random.key(3)
    action = jnp.concatenate([batch["action"], batch["action"]], axis=1)
    t_video = jnp.array([137.0, 913.0])
    t_rows = jnp.concatenate(
        [jnp.full((BATCH, ACTION_HORIZON), 642.0), jnp.full((BATCH, ACTION_HORIZON), 41.0)], axis=1
    )
    context, context_mask = model.append_proprio(
        batch["context"], batch["context_mask"], batch["proprio"][:, 0]
    )

    def run(act, taus):
        return model.predict(
            batch["latents"], act, t_video, taus, context, context_mask, lengths
        )

    video_a, action_a = run(action, t_rows)

    # perturb copy 1 only: its action values AND its tau
    perturbed = action.at[:, ACTION_HORIZON:].add(jax.random.normal(key, (BATCH, ACTION_HORIZON, ACTION_DIM)))
    taus_b = t_rows.at[:, ACTION_HORIZON:].set(777.0)
    video_b, action_b = run(perturbed, taus_b)

    assert np.array_equal(np.asarray(action_a[:, :ACTION_HORIZON]), np.asarray(action_b[:, :ACTION_HORIZON]))
    assert np.array_equal(np.asarray(video_a), np.asarray(video_b))
    # control: copy 1 itself DID move, or the perturbation never landed
    assert not np.array_equal(
        np.asarray(action_a[:, ACTION_HORIZON:]), np.asarray(action_b[:, ACTION_HORIZON:])
    )
    assert sa == action.shape[1]


def test_a_copy_matches_running_that_chunk_on_its_own():
    """Copy i's output equals a single-copy forward of the same rows at the same tau.

    Only fp32 reassociation separates them: the packed row reduces over a longer k axis
    whose extra terms are exact zeros, which XLA may tree-sum differently.
    """
    model = build_fastwam()
    batch = sample_batch()
    lengths = (ACTION_HORIZON, 4)
    t_video = jnp.array([137.0, 913.0])
    per_copy_t = (642.0, 41.0)
    context, context_mask = model.append_proprio(
        batch["context"], batch["context_mask"], batch["proprio"][:, 0]
    )

    packed = jnp.concatenate([batch["action"][:, :length] for length in lengths], axis=1)
    t_rows = jnp.concatenate(
        [jnp.full((BATCH, length), t) for length, t in zip(lengths, per_copy_t)], axis=1
    )
    _, packed_action = model.predict(
        batch["latents"], packed, t_video, t_rows, context, context_mask, lengths
    )

    offset = 0
    for length, t in zip(lengths, per_copy_t):
        _, alone = model.predict(
            batch["latents"],
            batch["action"][:, :length],
            t_video,
            jnp.full((BATCH,), t),
            context,
            context_mask,
        )
        assert rel_error(packed_action[:, offset : offset + length], alone) < 1e-5
        offset += length


# --------------------------------------------------------------------------- #
# the loss
# --------------------------------------------------------------------------- #
def test_single_copy_training_loss_is_the_released_path_bit_for_bit():
    model = build_fastwam()
    batch = sample_batch()
    key = jax.random.key(11)
    base_loss, base_metrics = model.training_loss(batch, key)
    for lengths in (None, (ACTION_HORIZON,)):
        loss, metrics = model.training_loss(batch, key, copy_lengths=lengths)
        assert np.asarray(loss) == np.asarray(base_loss)
        for name in ("loss_video", "loss_action"):
            assert np.asarray(metrics[name]) == np.asarray(base_metrics[name])
        assert np.array_equal(np.asarray(metrics["t_action"]), np.asarray(base_metrics["t_action"]))


def test_copies_draw_one_independent_tau_each():
    model = build_fastwam()
    batch = sample_batch()
    _, metrics = model.training_loss(batch, jax.random.key(11), copy_lengths=(8, 4, 2))
    t_action = np.asarray(metrics["t_action"])
    assert t_action.shape == (BATCH, 3), "one flow time per copy, per sample"
    assert len(np.unique(t_action)) == t_action.size, "the copies' taus must be independent draws"


def test_unequal_copies_are_averaged_per_copy_not_per_row():
    """An 8-row copy must not outweigh a 2-row copy 4:1 — that is the whole point of a mix.

    Driven through the real loss: `timesteps` and `noise` are supplied, so the forward can
    be reproduced here exactly and the reported `loss_action` compared against BOTH
    candidate reductions. The two differ only when the lengths do, which is why a test at
    equal lengths would pass no matter which one the code implements.
    """
    model = build_fastwam()
    batch = dict(sample_batch())
    batch["action_is_pad"] = jnp.zeros((BATCH, ACTION_HORIZON), bool)
    lengths = (ACTION_HORIZON, 2)
    action_cfg = model.config.action_scheduler

    t_video = jnp.array([137.0, 913.0])
    t_action = jnp.array([[642.0, 41.0], [88.0, 501.0]])  # [B, M], one per copy
    packed = jnp.concatenate([batch["action"][:, :length] for length in lengths], axis=1)
    noise_video = jax.random.normal(jax.random.key(21), batch["latents"].shape)
    noise_action = jax.random.normal(jax.random.key(22), packed.shape)

    _, metrics = model.training_loss(
        batch,
        jax.random.key(5),
        timesteps=(t_video, t_action),
        noise=(noise_video, noise_action),
        copy_lengths=lengths,
    )

    # rebuild the same forward and score it both ways
    context, context_mask = model.append_proprio(
        batch["context"], batch["context_mask"], batch["proprio"][:, 0]
    )
    t_rows = model._copy_t_per_row(t_action, lengths)
    noisy = flow_match.add_noise(packed, noise_action, t_rows, action_cfg.num_train_timesteps)
    target = flow_match.training_target(packed, noise_action)
    # the video side exactly as training_loss builds it: noised, then frame 0 written back
    video_cfg = model.config.video_scheduler
    noisy_latents = flow_match.add_noise(
        batch["latents"], noise_video, t_video, video_cfg.num_train_timesteps
    ).at[:, :, 0:1].set(batch["latents"][:, :, 0:1])
    _, pred = model.predict(
        noisy_latents, noisy, t_video, t_rows, context, context_mask, lengths
    )
    weight = flow_match.training_weight(t_action, action_cfg.num_train_timesteps)

    by_copy, offset = [], 0
    for index, length in enumerate(lengths):
        rows = slice(offset, offset + length)
        by_copy.append(
            jnp.mean(model._action_loss_per_sample(pred[:, rows], target[:, rows], None) * weight[:, index])
        )
        offset += length
    per_copy_loss = float(sum(by_copy) / len(lengths))
    # the reduction the code must NOT use: one mean over all rows, with each row carrying
    # its own copy's weight, which lets the long copy dominate
    per_row_loss = float(
        jnp.mean(
            model._action_loss_per_sample(pred, target, None)
            * jnp.mean(model._copy_t_per_row(weight, lengths), axis=1)
        )
    )

    assert float(metrics["loss_action"]) == pytest.approx(per_copy_loss, rel=1e-6)
    assert abs(per_copy_loss - per_row_loss) > 1e-6, "the two reductions must actually differ here"
    assert float(metrics["loss_action"]) != pytest.approx(per_row_loss, rel=1e-6)


def test_the_pad_mask_follows_its_copy():
    """`action_is_pad` is tiled with the same prefixes, or a short copy masks the wrong rows."""
    model = build_fastwam()
    batch = dict(sample_batch())
    # rows 6,7 are padding; a length-4 copy therefore contains none of them
    lengths = (ACTION_HORIZON, 4)
    _, metrics = model.training_loss(batch, jax.random.key(5), copy_lengths=lengths)
    assert np.isfinite(np.asarray(metrics["loss_action"]))

    packed = model._pack_copies(batch["action_is_pad"], lengths)
    assert packed.shape == (BATCH, sum(lengths))
    assert np.array_equal(np.asarray(packed[:, :ACTION_HORIZON]), np.asarray(batch["action_is_pad"]))
    assert np.array_equal(np.asarray(packed[:, ACTION_HORIZON:]), np.asarray(batch["action_is_pad"][:, :4]))


def test_sample_training_t_keeps_the_single_copy_draw_identical():
    key = jax.random.key(2)
    one = flow_match.sample_training_t(key, BATCH, 5.0, 1000)
    also_one = flow_match.sample_training_t(key, BATCH, 5.0, 1000, copies=1)
    assert np.array_equal(np.asarray(one), np.asarray(also_one))
    assert one.shape == (BATCH,)
    assert flow_match.sample_training_t(key, BATCH, 5.0, 1000, copies=3).shape == (BATCH, 3)


def test_a_timestep_override_must_be_per_copy():
    model = build_fastwam()
    batch = sample_batch()
    key = jax.random.key(1)
    t_video = jnp.array([137.0, 913.0])
    lengths = (ACTION_HORIZON, 4)
    packed = jnp.concatenate([batch["action"][:, :length] for length in lengths], axis=1)
    good = dict(
        noise=(
            jax.random.normal(jax.random.key(21), batch["latents"].shape),
            jax.random.normal(jax.random.key(22), packed.shape),
        ),
        copy_lengths=lengths,
    )
    model.training_loss(batch, key, timesteps=(t_video, jnp.zeros((BATCH, 2))), **good)
    for bad in (jnp.zeros((BATCH,)), jnp.zeros((BATCH, 3)), jnp.zeros((BATCH, sum(lengths)))):
        with pytest.raises(Exception, match="per copy|one flow time per copy"):
            model.training_loss(batch, key, timesteps=(t_video, bad), **good)


def test_add_noise_still_rejects_a_timestep_that_does_not_line_up():
    x0 = jnp.zeros((BATCH, 5, 3))
    noise = jnp.ones_like(x0)
    flow_match.add_noise(x0, noise, jnp.zeros((BATCH,)), 1000)
    flow_match.add_noise(x0, noise, jnp.zeros((BATCH, 5)), 1000)
    for bad in (jnp.zeros((BATCH, 4)), jnp.zeros((BATCH + 1,)), jnp.zeros((BATCH, 5, 3))):
        with pytest.raises(flow_match.SchedulerError):
            flow_match.add_noise(x0, noise, bad, 1000)


# --------------------------------------------------------------------------- #
# MFU accounting
# --------------------------------------------------------------------------- #
def test_copies_are_charged_to_the_mfu_numerator():
    """Copies are real work. Reporting the data's 32 rows would flatter a copies run."""
    base = flops_per_sample(load_config(TASK))
    assert base / 1e12 == pytest.approx(10.93, abs=0.01)  # the figure docs/training.md quotes

    mix = flops_per_sample(load_config(TASK, overrides=["flow.copy_lengths=[32,16,8,4]"]))
    four = flops_per_sample(load_config(TASK, overrides=["flow.tau_copies=4"]))
    assert mix / base == pytest.approx(1.015, abs=0.005)
    assert four / base == pytest.approx(1.053, abs=0.005)
