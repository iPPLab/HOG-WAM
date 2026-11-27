"""Window index arithmetic: the one-frame conventions that never raise when wrong.

Every expectation here is hand-computed from the official's own clamp
(`lerobot_dataset.py:660-673`) rather than from a second run of the code under test.
"""

import numpy as np
import pytest

from fastwam_tpu.data import windows

NUM_FRAMES = 33  # official obs_size
RATIO = 4  # action_video_freq_ratio -> 9 video frames, 32 actions
HORIZON = NUM_FRAMES - 1


def test_an_episode_contributes_one_window_per_lerobot_row():
    assert windows.num_windows(126) == 125
    assert windows.num_windows(2) == 1
    for bad in (1, 0, -3):
        with pytest.raises(ValueError):
            windows.num_windows(bad)


def test_full_dataset_window_count_matches_the_official_transition_count():
    """The identity that proves the two pipelines see the same episodes: our
    `sum(T-1)` over pixel_v3 equals the official `num_transition` of 6,075,103."""
    lengths = np.array([126, 75, 1085, 300], np.int64)
    assert sum(windows.num_windows(int(t)) for t in lengths) == int((lengths - 1).sum())


def test_interior_window_has_no_padding():
    idx, pad = windows.video_indices(0, 100, video_frames=9, video_stride=RATIO)
    assert idx.tolist() == [0, 4, 8, 12, 16, 20, 24, 28, 32]
    assert not pad.any()

    a_idx, a_pad = windows.action_indices(0, 100, action_horizon=HORIZON, action_shift=1)
    assert a_idx.tolist() == list(range(1, 33))
    assert not a_pad.any()

    p_idx, p_pad = windows.proprio_indices(0, 100, action_horizon=HORIZON)
    assert p_idx.tolist() == list(range(0, 32))
    assert not p_pad.any()


def test_tail_window_clamps_and_flags_by_hand():
    """T=6 raw frames -> 5 LeRobot rows (0..4) -> valid starts 0..4. From start 3:

        video  rows 3, 7, 11, ... -> clamped to 3, 4, 4, ...   pad F, T, T, ...
        action rows 3, 4, 5, ...  -> clamped to 3, 4, 4, ...   pad F, F, T, ...
               qpos = row + 1     -> 4, 5, 5, ...
        proprio rows as action, qpos = row -> 3, 4, 4, ...
    """
    idx, pad = windows.video_indices(3, 6, video_frames=9, video_stride=RATIO)
    assert idx.tolist() == [3] + [4] * 8
    assert pad.tolist() == [False] + [True] * 8

    a_idx, a_pad = windows.action_indices(3, 6, action_horizon=HORIZON, action_shift=1)
    assert a_idx.tolist() == [4, 5] + [5] * 30
    assert a_pad.tolist() == [False, False] + [True] * 30

    p_idx, p_pad = windows.proprio_indices(3, 6, action_horizon=HORIZON)
    assert p_idx.tolist() == [3, 4] + [4] * 30
    assert p_pad.tolist() == [False, False] + [True] * 30


def test_observations_never_reach_the_last_raw_frame():
    """The LeRobot table stops one row short, so the final image is only ever an action
    target. Clamping to T-1 instead of T-2 would show the model a frame the official
    model never saw in that position, and nothing downstream would notice."""
    n_frames = 40
    for start in range(windows.num_windows(n_frames)):
        v, _ = windows.video_indices(start, n_frames, video_frames=9, video_stride=RATIO)
        p, _ = windows.proprio_indices(start, n_frames, action_horizon=HORIZON)
        assert v.max() <= n_frames - 2
        assert p.max() <= n_frames - 2
        a, _ = windows.action_indices(start, n_frames, action_horizon=HORIZON, action_shift=1)
        assert a.max() <= n_frames - 1


def test_pad_marks_exactly_the_replicated_steps():
    """`is_pad` is computed before the clamp, so it must be True exactly where the index
    stopped moving because it ran off the end."""
    rng = np.random.default_rng(0)
    for n_frames in rng.integers(3, 60, size=25):
        n_frames = int(n_frames)
        for start in range(windows.num_windows(n_frames)):
            idx, pad = windows.video_indices(start, n_frames, video_frames=9, video_stride=RATIO)
            raw = start + np.arange(9) * RATIO
            assert pad.tolist() == (raw >= n_frames - 1).tolist()
            assert idx.tolist() == np.clip(raw, 0, n_frames - 2).tolist()

            a_idx, a_pad = windows.action_indices(start, n_frames, action_horizon=HORIZON)
            raw_a = start + np.arange(HORIZON)
            assert a_pad.tolist() == (raw_a >= n_frames - 1).tolist()
            assert a_idx.tolist() == (np.clip(raw_a, 0, n_frames - 2) + 1).tolist()


def test_action_shift_is_a_parameter_not_a_literal():
    """D5 is unresolved, so flipping the convention has to be a one-line change. With
    shift 0 the action chunk is the proprio chunk."""
    a_idx, a_pad = windows.action_indices(5, 60, action_horizon=HORIZON, action_shift=0)
    p_idx, p_pad = windows.proprio_indices(5, 60, action_horizon=HORIZON)
    assert a_idx.tolist() == p_idx.tolist()
    assert a_pad.tolist() == p_pad.tolist()
    assert windows.ACTION_SHIFT_DEFAULT == 1


def test_action_shift_that_would_read_past_the_episode_raises():
    with pytest.raises(ValueError, match="outside"):
        windows.action_indices(0, 34, action_horizon=HORIZON, action_shift=3)
