"""Loop check for policy/fastwam_policy: replan cadence, row order, env-var immunity.

Drives the real `fastwam_policy.eval()` against a fake TASK_ENV and a fake model client,
so what is checked is the code the sim will run. No sapien, no GPU, no weights.

The three things it pins, in order of how expensive they are to get wrong:

  1. **replan = 24, executed from row 0.** The official predicts 32 and executes
     `chunk[0:24]` (deploy_policy.py:288-291). Our own v3 line drops row 0 under
     `TW_ANCHOR_PREV` and the shared harness *forces* `TW_CHUNK_EXEC=32`; honouring
     either would execute the wrong rows of every chunk. Both variables are set to their
     TinyWAM values below and the adapter must ignore them.
  2. **set_task once per episode, with a real sentence.** The instruction is drawn from
     the `unseen` pool per episode; it must reach the server before the first
     `get_action`, because the text is consumed by the one video prefill.
  3. **The rows reach take_action in order, unmodified.** No integration, no anchor, no
     re-scaling.

    python scripts/check_adapter.py
"""

import os
import sys

import numpy as np

# Set the TinyWAM conventions in the environment BEFORE importing the adapter: the whole
# point is that they must have no effect. TW_CHUNK_EXEC=32 is what eval_harness/worker.sh
# forces for its own campaigns.
os.environ["TW_CHUNK_EXEC"] = "32"
os.environ["TW_ANCHOR_PREV"] = "1"
os.environ["TW_PROF"] = "0"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policies"))
import fastwam_policy as adapter  # noqa: E402

HORIZON, ACTION_DIM, REPLAN = 32, 14, 24
INSTRUCTION = "Pick up the shoe with lace holes and move it to the mat"


class FakeEnv:
    def __init__(self, instruction=INSTRUCTION):
        self.take_action_cnt = 0
        self.executed = []
        self._instruction = instruction

    def get_instruction(self, instruction=None):
        return self._instruction

    def take_action(self, action, action_type=None):
        assert action_type == "qpos", f"action_type must be 'qpos', got {action_type!r}"
        assert action.shape == (ACTION_DIM,), action.shape
        self.executed.append(np.asarray(action, np.float32).copy())
        self.take_action_cnt += 1


class FakeModel:
    """Returns a chunk whose row i is filled with the value `replan*100 + i`, so the
    executed sequence says exactly which chunk and which row each step came from."""

    def __init__(self):
        self.calls = []
        self.replans = 0

    def call(self, func_name, obs=None):
        self.calls.append((func_name, obs))
        if func_name == "set_task":
            assert isinstance(obs["instruction"], str) and obs["instruction"].strip()
            return "ok"
        if func_name == "get_action":
            assert obs["canvas"].shape == (384, 320, 3) and obs["canvas"].dtype == np.uint8
            assert obs["qpos"].shape == (ACTION_DIM,)
            chunk = np.zeros((HORIZON, ACTION_DIM), np.float32)
            for row in range(HORIZON):
                chunk[row, :] = self.replans * 100 + row
            self.replans += 1
            return chunk
        raise AssertionError(f"unexpected call {func_name}")


def observation(seed=0):
    rng = np.random.default_rng(seed)
    return {
        "observation": {
            cam: {"rgb": rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)}
            for cam in ("head_camera", "left_camera", "right_camera")
        },
        "joint_action": {"vector": np.linspace(-1, 1, ACTION_DIM).astype(np.float32)},
    }


def main():
    env, model = FakeEnv(), FakeModel()
    adapter.reset_model(model)

    steps = 60  # 2.5 replan windows
    for step in range(steps):
        # `should_request_observation()` and the queue must agree: when it says False the
        # client passes None, and the adapter must not need a frame.
        obs = observation(step) if adapter.should_request_observation() else None
        adapter.eval(env, model, obs)

    set_tasks = [c for c in model.calls if c[0] == "set_task"]
    get_actions = [c for c in model.calls if c[0] == "get_action"]

    assert len(set_tasks) == 1, f"set_task called {len(set_tasks)} times, expected once per episode"
    assert set_tasks[0][1]["instruction"] == INSTRUCTION
    print(f"ok   set_task once per episode, with the drawn sentence ({INSTRUCTION[:40]!r}...)")

    want_replans = -(-steps // REPLAN)  # ceil
    assert len(get_actions) == want_replans, (
        f"{len(get_actions)} replans over {steps} steps; expected {want_replans} "
        f"(one per {REPLAN}). TW_CHUNK_EXEC=32 was set in the environment and must be ignored."
    )
    print(f"ok   {len(get_actions)} replans over {steps} steps = one every {REPLAN}")

    assert len(env.executed) == steps
    for step, action in enumerate(env.executed):
        chunk_index, row = divmod(step, REPLAN)
        want = chunk_index * 100 + row
        assert np.allclose(action, want), (
            f"step {step}: executed {action[0]}, expected chunk {chunk_index} row {row} "
            f"({want}). A value of {chunk_index * 100 + row + 1} means row 0 was dropped "
            f"(TW_ANCHOR_PREV leaked in)."
        )
    print(f"ok   rows executed in order from row 0 of each chunk ({steps} steps verified)")

    # Only the replan steps needed a frame: 60 steps, 3 replans -> 3 observations used.
    rendered = sum(1 for step in range(steps) if step % REPLAN == 0)
    assert rendered == want_replans
    print(f"ok   should_request_observation() true on {rendered}/{steps} steps "
          f"(saves {steps - rendered} three-camera renders when the client honours it)")

    # --- a second episode in the same process, both client spellings --------------
    #
    # 60 steps is 2.5 windows, so episode 1 ends with 12 rows still queued — which is the
    # normal case, since episodes end on success or step_lim, never on a window boundary.
    #
    # (a) UNPATCHED client: renders every step, never asks should_request_observation().
    #     The stale queue is harmless; `eval` clears it on the take_action_cnt reset.
    leftover = len(adapter._QUEUE)
    assert leftover > 0, "the fixture no longer leaves a partial window; the check has no teeth"
    env2 = FakeEnv("Lift the pot and place it on the stove")
    for step in range(REPLAN + 1):
        adapter.eval(env2, model, observation(100 + step))
    set_tasks = [c for c in model.calls if c[0] == "set_task"]
    assert len(set_tasks) == 2 and set_tasks[1][1]["instruction"] == env2._instruction
    assert np.allclose(env2.executed[0], 300), (
        f"episode 2 started at {env2.executed[0][0]}, not a fresh chunk's row 0 — the queue "
        f"carried over from the previous episode ({leftover} rows were pending)"
    )
    print(f"ok   unpatched client: new episode re-sets the task and starts a fresh chunk "
          f"({leftover} stale rows discarded)")

    # (b) PATCHED client: consults should_request_observation() BEFORE eval(), so the
    #     episode boundary is invisible to it — which is exactly why the patch also calls
    #     the module's reset_model(). Without that call this loop raises on step 0, and
    #     that is a real closed-loop failure, not a test artifact.
    env3 = FakeEnv("Beat the block with the hammer")
    first_chunk = model.replans  # whatever chunk index this episode will be handed
    adapter.reset_model(model)  # <- what scripts/patches/skip_obs_in_replan.patch adds
    for step in range(REPLAN + 1):
        obs = observation(200 + step) if adapter.should_request_observation() else None
        adapter.eval(env3, model, obs)
    assert np.allclose(env3.executed[0], first_chunk * 100), (
        f"episode 3 step 0 executed {env3.executed[0][0]}, expected chunk {first_chunk} row 0"
    )
    print("ok   patched client: reset_model() at episode start makes the render skip safe")

    print("\nPASS: replan=24 from row 0, one set_task per episode, TW_CHUNK_EXEC/"
          "TW_ANCHOR_PREV ignored, episode boundaries clean in both client spellings")


if __name__ == "__main__":
    main()
