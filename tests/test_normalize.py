"""Normalisation against the official constants.

The numbers below are transcribed from
`dataset_stats.json` from the public FastWAM RoboTwin dataset release
(sha256 7a02c46c...) and the expectations are computed by hand from
`utils/normalizer.py:100-133`, not by running the code under test.
"""

import json

import numpy as np
import pytest

from fastwam_tpu.data.normalize import Normalizer, load_official_stats

# First three dims of the real file, full precision as json stores them.
ACTION_MEAN = [-0.24357569217681885, 1.162607192993164, 0.8319377899169922]
ACTION_STD = [0.39944010972976685, 1.0112872123718262, 0.787507951259613]
STATE_MEAN = [-0.22513258457183838, 1.0872493982315063, 0.778487503528595]
STATE_STD = [0.3852057456970215, 1.0093021392822266, 0.7779697775840759]


def _stats_blob(dim=3, num_episodes=27500, num_transition=6075103, **overrides):
    def field(mean, std):
        return {
            "global_mean": mean[:dim],
            "global_std": std[:dim],
            "global_min": [-9.0] * dim,
            "global_max": [9.0] * dim,
            "stepwise_mean": [[0.0] * dim],  # present in the real file, never read
        }

    payload = {
        "action": {"default": field(ACTION_MEAN, ACTION_STD)},
        "state": {"default": field(STATE_MEAN, STATE_STD)},
        "num_episodes": num_episodes,
        "num_transition": num_transition,
    }
    payload.update(overrides)
    return json.dumps(payload).encode()


def test_scale_and_offset_are_the_official_formula():
    """`scale = 1/(std + 1e-8)`, `offset = -mean/(std + 1e-8)`, computed in float32
    because the official casts the json to float32 before deriving them."""
    norm = Normalizer(np.array(ACTION_MEAN), np.array(ACTION_STD))
    mean32 = np.array(ACTION_MEAN, np.float32)
    std32 = np.array(ACTION_STD, np.float32)
    np.testing.assert_array_equal(norm.scale, (np.float32(1.0) / (std32 + np.float32(1e-8))))
    np.testing.assert_array_equal(norm.offset, (-mean32 / (std32 + np.float32(1e-8))))


def test_forward_is_a_z_score_by_hand():
    norm = Normalizer(np.array(ACTION_MEAN), np.array(ACTION_STD))
    x = np.array([[0.0, 1.0, 0.5]], np.float32)
    expected = (x - np.array(ACTION_MEAN, np.float32)) / (np.array(ACTION_STD, np.float32) + 1e-8)
    np.testing.assert_allclose(norm.forward(x), expected, rtol=1e-6)
    # dim 0: (0 - (-0.2435757)) / 0.3994401 = 0.6098
    assert abs(float(norm.forward(x)[0, 0]) - 0.609790) < 1e-5


def test_clamp_is_forward_only():
    """`forward` clamps to +-5, `backward` does not (`normalizer.py:127-133`). A clamped
    sample is information the deployed inverse can never give back, which is exactly why
    both halves have to be asymmetric in the same direction."""
    norm = Normalizer(np.zeros(3), np.ones(3))
    x = np.array([[-100.0, 0.25, 100.0]], np.float32)
    fwd = norm.forward(x)
    np.testing.assert_allclose(fwd, [[-5.0, 0.25, 5.0]], rtol=1e-6)
    back = norm.backward(np.array([[-100.0, 0.25, 100.0]], np.float32))
    np.testing.assert_allclose(back, [[-100.0, 0.25, 100.0]], rtol=1e-6)


def test_round_trip_below_the_clamp():
    rng = np.random.default_rng(0)
    norm = Normalizer(np.array(STATE_MEAN), np.array(STATE_STD))
    x = (rng.standard_normal((64, 3)).astype(np.float32) * np.array(STATE_STD, np.float32)
         + np.array(STATE_MEAN, np.float32))
    fwd = norm.forward(x)
    assert np.abs(fwd).max() < 5.0  # nothing clamped, so the inverse is exact
    assert np.abs(norm.backward(fwd) - x).max() < 1e-5


def test_forward_writes_into_a_preallocated_buffer():
    norm = Normalizer(np.array(ACTION_MEAN), np.array(ACTION_STD))
    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = np.empty((2, 3), np.float32)
    got = norm.forward(x, out=out)
    assert got is out
    np.testing.assert_allclose(out, norm.forward(x), rtol=1e-6)


def test_action_and_state_read_their_own_constants():
    """Two different sets of constants for the same 14 physical dims: the official's
    means differ because they are taken over different row sets (`m4_data_design.md`
    §Q2). Mixing them up is a pure affine shift with no symptom."""
    stats = load_official_stats(_stats_blob(), source="test")
    assert not np.allclose(stats.action.mean, stats.proprio.mean)
    np.testing.assert_allclose(stats.action.mean, np.array(ACTION_MEAN, np.float32), rtol=1e-6)
    np.testing.assert_allclose(stats.proprio.mean, np.array(STATE_MEAN, np.float32), rtol=1e-6)


def test_counts_come_through_for_the_same_episodes_check():
    stats = load_official_stats(_stats_blob(), source="test")
    assert (stats.num_episodes, stats.num_transition) == (27500, 6075103)


def test_sha256_mismatch_raises():
    blob = _stats_blob()
    with pytest.raises(ValueError, match="does not match the configured"):
        load_official_stats(blob, source="test", expect_sha256="0" * 64)
    import hashlib

    load_official_stats(blob, source="test", expect_sha256=hashlib.sha256(blob).hexdigest())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d.pop("action"), "missing action"),
        (lambda d: d["action"]["default"].pop("global_std"), "no global_std"),
        (lambda d: d["action"]["default"].update(global_std=[0.0, 1.0, 1.0]), "non-positive"),
        (lambda d: d["action"]["default"].update(global_mean=[float("nan"), 1.0, 1.0]), "non-finite"),
        (lambda d: d.pop("num_transition"), "missing num_transition"),
    ],
)
def test_a_broken_stats_file_raises_rather_than_defaults(mutate, message):
    payload = json.loads(_stats_blob())
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        load_official_stats(json.dumps(payload).encode(), source="test")


def test_stepwise_stats_are_deliberately_not_read():
    """`use_stepwise_action_norm` is False in every shipped RoboTwin config, so the
    stepwise arrays in the 88 KB file are dead weight; reading them by accident would
    silently change the action space."""
    stats = load_official_stats(_stats_blob(), source="test")
    assert not hasattr(stats.action, "stepwise_mean")
