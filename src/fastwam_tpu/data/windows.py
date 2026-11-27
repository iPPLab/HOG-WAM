"""Window index arithmetic: one `.twp` episode + one start -> the indices of a sample.

This module is the single place that knows how a training window maps onto raw frames.
It is pure numpy with no IO, because the closed-loop harness (M3) must build its
observation window with *these* functions and not a second copy of the arithmetic — a
one-frame anchor slip cost 48 points of closed-loop success once already
(`project_anchor_gap_fix`), and nothing in the code detects it.

The official pipeline never sees `.twp`; it sees a LeRobot table. Everything below is
that table's index convention, re-derived on our storage:

`.twp` stores an episode as T raw frames: `T` JPEG triples and `qpos[T, 14]`.
The official LeRobot conversion of the same episode has **T - 1 rows** (proved, not
assumed: our independently computed `sum(T_i - 1) = 6,075,103` equals the official
`dataset_stats.json`'s `num_transition` digit for digit). Each row is one sample start,
so an episode contributes `T - 1` windows and the whole dataset contributes 6,075,103.

Row `r` of that table holds
    observation.images.* = raw frame r
    observation.state    = qpos[r]
    action               = qpos[r + ACTION_SHIFT]      <- see `action_shift` below

Given a start row `i`, `BaseLerobotDataset` asks for these deltas
(`base_lerobot_dataset.py:68-86`, `obs_size = num_frames = 33`, `action_size = 32`):

    images / state : rows i+0 .. i+32     (33 obs steps)
    action         : rows i+0 .. i+31     (32 action steps)

and `LeRobotDataset._get_query_indices` (`lerobot_dataset.py:660-673`) clamps every
one of them into the episode and reports what it clamped:

    row     = max(ep_start, min(ep_end - 1, i + delta))       # clamp into [0, L-1]
    is_pad  = (i + delta) < 0 or (i + delta) >= L             # BEFORE the clamp

with `L = ep_end - ep_start = T - 1`. `RobotVideoDataset._get` then keeps only the 9
video frames (`robot_video_dataset.py:63, 145-151`) and drops the last proprio row to
line proprio up with action (`:203`).

Two consequences worth spelling out, because both are silent when wrong:

1. **The index ceiling is `T - 2`, not `T - 1`.** The LeRobot table stops one row short
   of the raw frames, so the last raw *image* of an episode is never used as an
   observation — it exists only as the last action target. Clamping to `T - 1` would
   quietly show the model a frame the official model never saw at that position.
2. **`is_pad` is computed before clamping**, so it flags "this step ran off the end of
   the episode and got the last one replicated", which is exactly what the loss masks
   want (`fastwam.py:431-436` for video, `:550-556` for action).
"""

import numpy as np

__all__ = [
    "ACTION_SHIFT_DEFAULT",
    "action_indices",
    "num_windows",
    "proprio_indices",
    "video_indices",
]

# D5 (fastwam_tpu_plan.md M4): `action[r] = qpos[r + 1]`, i.e. the action commanded at
# row r is the NEXT joint pose. The evidence is strong but circumstantial:
#   * the official LeRobot table has T-1 rows, and dropping exactly one row per episode
#     is what `action[r] = qpos[r+1]` forces (`qpos[T-1]` has no successor);
#   * the official `state.global_min/max` and `action.global_min/max` agree in all 14
#     dims (same physical quantity) while their means differ (different row sets);
#   * closed loop, `take_action(qpos)` takes an absolute joint target, so the action
#     must lead the state or step one is a no-op.
# It is still a one-frame bet, so it is a PARAMETER and not a literal: the discriminating
# experiment (teacher-forced forward with the official ckpt, compare the predicted chunk
# against `qpos[i+1..i+32]` vs `qpos[i..i+31]`, lower error wins) is cheap and belongs to
# whoever holds the ckpt. Flipping the answer is a one-line config change.
ACTION_SHIFT_DEFAULT = 1


def num_windows(n_frames: int) -> int:
    """Sample starts in an episode of `n_frames` raw frames: the LeRobot row count."""
    if n_frames < 2:
        raise ValueError(f"an episode needs at least 2 frames to yield a window, got {n_frames}")
    return int(n_frames) - 1


def _clamped(start: int, deltas: np.ndarray, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """`(rows, is_pad)` for `start + deltas`, LeRobot's clamp-and-flag (`:660-673`)."""
    raw = int(start) + deltas
    rows = np.clip(raw, 0, num_windows(n_frames) - 1)
    return rows, raw >= num_windows(n_frames)


def video_indices(
    start: int, n_frames: int, *, video_frames: int = 9, video_stride: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """`(frame_index[9], image_is_pad[9])` for the window starting at row `start`.

    The 33 observation steps are subsampled by `action_video_freq_ratio = 4`, so the
    frames are `start, start+4, ..., start+32`. A raw frame index equals its row index.
    """
    deltas = np.arange(video_frames, dtype=np.int64) * int(video_stride)
    return _clamped(start, deltas, n_frames)


def action_indices(
    start: int, n_frames: int, *, action_horizon: int = 32, action_shift: int = ACTION_SHIFT_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """`(qpos_index[32], action_is_pad[32])` for the action chunk at row `start`.

    The shift is applied *after* the clamp, exactly as reading the LeRobot `action`
    column of a clamped row would: `qpos[min(start+j, T-2) + shift]`. With the default
    shift of 1 the largest index is `T-1`, i.e. the final raw frame, which is precisely
    the row the LeRobot table gives up in exchange (see the module docstring).
    """
    rows, is_pad = _clamped(start, np.arange(action_horizon, dtype=np.int64), n_frames)
    idx = rows + int(action_shift)
    if idx.min() < 0 or idx.max() > n_frames - 1:
        raise ValueError(
            f"action_shift={action_shift} sends qpos index {idx.min()}..{idx.max()} outside "
            f"[0, {n_frames - 1}] for start={start}"
        )
    return idx, is_pad


def proprio_indices(
    start: int, n_frames: int, *, action_horizon: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """`(qpos_index[32], is_pad[32])` for the proprio rows at `start`.

    The official builds 33 state rows and then throws the last one away to match the
    action length (`robot_video_dataset.py:203`), so this is `qpos[start .. start+31]`,
    clamped. Only index 0 ever reaches the model (`fastwam.py:361`); the rest is carried
    because the official tensor carries it. The official's own `proprio_is_pad` stays
    length 33 and is read by nobody, so we return the length-32 mask that lines up with
    the tensor and emit no `proprio_is_pad` at all.
    """
    return _clamped(start, np.arange(action_horizon, dtype=np.int64), n_frames)
