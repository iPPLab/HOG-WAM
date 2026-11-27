"""RoboTwin policy plugin. Compose a three-camera canvas for the JAX model and execute the first 24 actions of each 32-step prediction."""

import atexit
import os
import time
from collections import deque

import numpy as np
from PIL import Image

# deploy_policy.py:221-234 + configs/sim_robotwin.yaml. Constants, not env vars: every
# one of them changes what the model is being asked to do, so they belong with the code
# that was verified against the official source rather than with the launcher.
HEAD_SIZE_WH = (320, 256)
WRIST_SIZE_WH = (160, 128)
CANVAS_HW = (384, 320)
REPLAN_STEPS = 24
ACTION_HORIZON = 32
ACTION_DIM = 14

_QUEUE: deque = deque()
_LAST_CNT = -1
_NEED_TASK = True
_REPLANS = 0
_PROF = os.environ.get("TW_PROF", "1") == "1"
_TIMING = {"model": 0.0, "act": 0.0, "harness": 0.0, "n": 0, "t_last": None, "t_ep0": None}


def _profile_flush():
    if _TIMING["n"]:
        total = time.perf_counter() - _TIMING["t_ep0"]
        other = total - _TIMING["harness"] - _TIMING["model"] - _TIMING["act"]
        print(
            f"[fw_prof] steps={_TIMING['n']} replans={_REPLANS} total={total:.1f}s "
            f"harness(render)={_TIMING['harness']:.1f} model(rpc)={_TIMING['model']:.1f} "
            f"act(phys+video)={_TIMING['act']:.1f} other={other:.1f}",
            flush=True,
        )
    _TIMING.update(model=0.0, act=0.0, harness=0.0, n=0, t_last=None, t_ep0=None)


# The harness runs ONE episode per process, so the episode-boundary flush inside eval()
# never fires and the profile was never printed. Flush at interpreter exit instead.
atexit.register(_profile_flush)


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """`deploy_policy.py:130-133`, verbatim.

    PIL's BILINEAR is support-scaled — antialiased on a downscale — which is why it
    matches the training pipeline's `torchvision.Resize(..., antialias=True)` closely.
    The resize runs uint8 -> uint8 exactly as the official does; moving it to float here
    would be a different quantisation of the same picture and nothing downstream would
    notice.
    """
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return np.asarray(pil.resize(size_wh, resample=Image.BILINEAR), dtype=np.uint8)


def compose_canvas(observation: dict) -> np.ndarray:
    """Three camera frames -> the official uint8 [384, 320, 3] RGB canvas.

        head_camera  [240,320,3] -> [256,320]
        left_camera  [240,320,3] -> [128,160]
        right_camera [240,320,3] -> [128,160]
        bottom = concat([left, right], axis=1)   -> [128,320]
        canvas = concat([head, bottom], axis=0)  -> [384,320]

    The `* (2/255) - 1` scaling stays server-side, on the uint8 array, so the wire
    payload is 368 KB instead of 1.4 MB of float32.
    """
    cameras = observation["observation"]
    head = _resize_rgb(cameras["head_camera"]["rgb"], HEAD_SIZE_WH)
    left = _resize_rgb(cameras["left_camera"]["rgb"], WRIST_SIZE_WH)
    right = _resize_rgb(cameras["right_camera"]["rgb"], WRIST_SIZE_WH)
    canvas = np.concatenate([head, np.concatenate([left, right], axis=1)], axis=0)
    if canvas.shape != (*CANVAS_HW, 3):
        raise ValueError(
            f"composed canvas is {canvas.shape}, expected {(*CANVAS_HW, 3)}; the camera "
            "resolution is not what this recipe was verified against (task_config wants "
            "head_camera_type: D435 -> 320x240)"
        )
    return np.ascontiguousarray(canvas, dtype=np.uint8)


def get_model(usr_args=None):
    """Never called: the weights live in the model server process."""
    raise RuntimeError(
        "fastwam_policy runs client/server-split; start scripts/fastwam_model_server.py "
        "and point eval_policy_client.py at its port"
    )


def reset_model(model=None):
    """Clear the episode. The OFFICIAL client calls this at episode start
    (`eval_policy.py:365`, `reset_func(model)`); our forked `eval_policy_client.py`
    calls `model.call('reset_model')` instead, which only resets the SERVER.

    That difference matters as soon as `should_request_observation()` is honoured — see
    the note there — so the render-skip patch adds this call back.
    """
    global _LAST_CNT, _NEED_TASK, _REPLANS
    _QUEUE.clear()
    _LAST_CNT = -1
    _NEED_TASK = True
    _REPLANS = 0


def should_request_observation() -> bool:
    """`deploy_policy.py:262-263`: render only on a replan step.

    The official's `skip_get_obs_within_replan` skips 23 of every 24 three-camera
    renders, which is most of the wall clock. Honoured only if the client is patched to
    ask (`scripts/patches/skip_obs_in_replan.patch`).

    ORDERING TRAP, found by `scripts/check_adapter.py`: the client asks this BEFORE
    calling `eval()`, and the episode boundary is only visible inside `eval()` (via
    `TASK_ENV.take_action_cnt` resetting). An episode almost never ends on a window
    boundary, so at the first step of the NEXT episode the queue still holds the
    previous episode's leftover rows — this would answer False, the client would pass
    `observation=None`, and `eval()` would then clear the queue and have no frame to
    replan from. That is why the patch also calls `reset_model()` at episode start; with
    both halves applied the queue is empty before this is first consulted.

    Unpatched, this function is never called and the stale queue is harmless: `eval()`
    clears it and the client has rendered anyway.
    """
    return not _QUEUE


def eval(TASK_ENV, model, observation):
    """One env step. `observation` may be None when the client skipped the render."""
    global _LAST_CNT, _NEED_TASK, _REPLANS
    entered = time.perf_counter()

    count = TASK_ENV.take_action_cnt
    if count <= _LAST_CNT:  # the counter reset -> a new episode started
        _QUEUE.clear()
        _NEED_TASK = True
        _REPLANS = 0
        if _PROF:
            _profile_flush()
    if _PROF:
        if _TIMING["t_ep0"] is None:
            _TIMING["t_ep0"] = entered
        if _TIMING["t_last"] is not None:
            _TIMING["harness"] += entered - _TIMING["t_last"]
    _LAST_CNT = count

    if _NEED_TASK:
        # The instruction is drawn per episode from the `unseen` pool by the client
        # (`eval_policy_client.py:450-452`) and is a real sentence. It must reach the
        # server BEFORE the first get_action: the text is read during the one video
        # prefill, so a late set_task would condition the whole chunk on nothing.
        # A failure here is fatal, not a warning — an episode run on the wrong text
        # scores like a modelling problem and nothing in the logs would say otherwise.
        instruction = TASK_ENV.get_instruction()
        if not instruction or not str(instruction).strip():
            raise RuntimeError(
                "TASK_ENV.get_instruction() is empty. FastWAM conditions on the drawn "
                "instruction; check `instruction_type: unseen` in deploy_policy.yml."
            )
        model.call(func_name="set_task", obs={"instruction": str(instruction)})
        print(f"[fastwam_policy] set_task {str(instruction)!r}", flush=True)
        _NEED_TASK = False

    if not _QUEUE:
        if observation is None:
            raise ValueError(
                "the action queue is empty but no observation was provided — a replan needs "
                "the current frame. If TW_SKIP_OBS_IN_REPLAN=1, the client is asking "
                "should_request_observation() but is NOT calling this module's reset_model() "
                "at episode start, so the first step of an episode consulted a queue still "
                "holding the previous one's rows. Apply the whole of "
                "scripts/patches/skip_obs_in_replan.patch, not just the need_obs half."
            )
        canvas = compose_canvas(observation)
        condition = os.environ.get("HOGWAM_ROBOTWIN_CONDITION", "clean")
        if condition not in ("clean", "randomized"):
            from .rgb_perturbation import PAPER_PROFILES, perturb_rgb

            # The campaign starts a fresh process for each rollout.
            seed = 20260927 + 1_000_003 + _REPLANS
            canvas = perturb_rgb(canvas, seed=seed, profile=PAPER_PROFILES[condition])
        qpos = np.asarray(observation["joint_action"]["vector"], np.float32)
        if qpos.shape != (ACTION_DIM,):
            raise ValueError(f"joint_action vector is {qpos.shape}, expected ({ACTION_DIM},)")

        started = time.perf_counter()
        chunk = np.asarray(
            model.call(func_name="get_action", obs={"canvas": canvas, "qpos": qpos}), np.float32
        )
        if _PROF:
            _TIMING["model"] += time.perf_counter() - started
        if chunk.shape != (ACTION_HORIZON, ACTION_DIM):
            raise ValueError(
                f"server returned {chunk.shape}, expected ({ACTION_HORIZON}, {ACTION_DIM})"
            )
        if not np.isfinite(chunk).all():
            raise ValueError("server returned non-finite actions")
        # Execute the first 24 of 32 rows, from row 0. Receding horizon: the tail 8 rows
        # are the least-conditioned, so they are dropped and re-predicted from the next
        # fresh observation. No row is skipped at the front — the chunk is absolute qpos,
        # not a delta from an anchor.
        _QUEUE.extend(chunk[:REPLAN_STEPS])
        _REPLANS += 1

    started = time.perf_counter()
    TASK_ENV.take_action(np.asarray(_QUEUE.popleft(), np.float32), action_type="qpos")
    if _PROF:
        _TIMING["act"] += time.perf_counter() - started
        _TIMING["n"] += 1
        _TIMING["t_last"] = time.perf_counter()
