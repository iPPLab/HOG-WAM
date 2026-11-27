"""Is `--no_expert_check` usable, and is it worth using?

Two questions, one sim setup each, answered per task:

1. **How expensive is the expert pre-roll?** `eval_policy()` runs `play_once()` — the
   scripted expert — before every model episode, purely to prove the seed is solvable.
   With a canonical seed list that fact is already known, so the pre-roll is pure waste.
   This times it.

2. **Can we skip it without breaking the instruction?** We cannot just pass
   `--no_expert_check True`: `episode_info` is bound only inside the `if expert_check:`
   branch (`eval_policy_client.py:417`) and is then read unconditionally at :449 to build
   the instruction pool, so the flag as shipped raises NameError. Nothing in the archived
   logs ever printed `[canonical] st_seed=`, which is consistent with the path never
   having been run.

   The map it wants — `{"{A}": "060_kitchenpot/base0", "{a}": "left", ...}` — is written
   by the LAST statement of each task's `play_once()`, but its VALUES come from
   `load_actors()` (model ids, chosen during `setup_demo`) and from arm tags computed
   from the initial geometry. So the map is a function of the scene, not of the motion,
   and should be recoverable without moving anything.

   Two ways to get `play_once()` to fall through to that final statement:
     * `noop_move`  — replace `move()` with a no-op. The `grasp_actor(...)` arguments are
       still evaluated.
     * `plan_false` — set `plan_success = False`. Both `move()` (:908) and
       `grasp_actor()` (:1176) short-circuit on it, so the body costs nothing.

   Both are tested against the real map. A strategy is only usable if it reproduces the
   map EXACTLY for every task: a mismatch means the drawn instruction describes the wrong
   object, and FastWAM conditions on that sentence, so it would be a silent scoring bug
   rather than a crash.

Usage: prefer the guarded wrapper, which waits for VRAM headroom first —
    bash eval/scripts/fastwam_eval/run_probe_guarded.sh

Directly (cwd must be the RoboTwin root, sim venv), from the repo root:
    cd eval/third_party/RoboTwin
    CUDA_VISIBLE_DEVICES=1 ../../.venv-robotwin/bin/python \
        ../../scripts/fastwam_eval/probe_seed_info.py \
        --seedlist ../../seeds/canonical_seeds_v1.json \
        --tasks ~/fastwam_eval/tasks.txt
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback

sys.path.append("./")
sys.path.append("./policy")
sys.path.append("./description/utils")
sys.path.append("./script")

import yaml  # noqa: E402
from envs import CONFIGS_PATH  # noqa: E402


def build_args(task_name, task_config):
    """The `args` dict exactly as eval_policy_client.main() assembles it (:262-301),
    minus the video/save_dir side effects, which the probe does not want."""
    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "probe"
    args["policy_name"] = "probe"
    args["eval_video_log"] = False

    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiments = yaml.load(f.read(), Loader=yaml.FullLoader)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        cameras = yaml.load(f.read(), Loader=yaml.FullLoader)

    head = args["camera"]["head_camera_type"]
    args["head_camera_h"] = cameras[head]["h"]
    args["head_camera_w"] = cameras[head]["w"]

    def robot_file(name):
        path = embodiments[name]["file_path"]
        if path is None:
            raise ValueError(f"no embodiment file for {name}")
        return path

    et = args.get("embodiment")
    if len(et) == 1:
        args["left_robot_file"] = args["right_robot_file"] = robot_file(et[0])
        args["dual_arm_embodied"] = True
    elif len(et) == 3:
        args["left_robot_file"] = robot_file(et[0])
        args["right_robot_file"] = robot_file(et[1])
        args["embodiment_dis"] = et[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    def embodiment_config(path):
        with open(os.path.join(path, "config.yml"), "r", encoding="utf-8") as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    args["left_embodiment_config"] = embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = embodiment_config(args["right_robot_file"])
    args["eval_mode"] = True
    args["render_freq"] = 0
    return args


def roll(env, args, seed, strategy):
    """One setup_demo + play_once. Returns (info_map, seconds, error).

    `seconds` covers play_once only — setup_demo is paid by the model episode too, so it
    is not part of what skipping the expert saves.
    """
    env.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
    original = env.move
    if strategy == "noop_move":
        env.move = lambda *a, **k: False
    elif strategy == "plan_false":
        env.plan_success = False

    info, err = None, None
    t0 = time.time()
    try:
        info = dict(env.play_once().get("info", {}))
    except Exception:
        err = traceback.format_exc(limit=3)
    elapsed = time.time() - t0

    extra = {}
    if strategy is None:
        # Only meaningful for the real roll: did the expert actually solve this seed?
        try:
            extra = {"plan_success": bool(env.plan_success), "check_success": bool(env.check_success())}
        except Exception:
            extra = {"plan_success": None, "check_success": None}

    env.move = original
    try:
        env.close_env()
    except Exception:
        pass
    return info, elapsed, err, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seedlist", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--task-config", default="demo_clean")
    ap.add_argument("--n-seeds", type=int, default=1, help="canonical seeds per task to probe")
    # `noop_move` measured 24.5 s against `plan_false`'s 0.0 s on adjust_bottle, for the
    # same map: the planning lives in the `grasp_actor(...)` arguments, which a no-op
    # `move` still evaluates. Keep it available to re-check that reading, but the default
    # is the cheap strategy so a 10-task sweep costs minutes rather than half an hour.
    ap.add_argument("--strategies", default="plan_false",
                    help="comma-separated: plan_false,noop_move")
    args_cli = ap.parse_args()
    strategies = [s for s in args_cli.strategies.split(",") if s]

    seeds = json.load(open(args_cli.seedlist))["seeds"]
    tasks = [t.strip() for t in open(args_cli.tasks) if t.strip() and not t.startswith("#")]

    def class_decorator(task_name):  # eval_policy_client.py:96
        module = importlib.import_module(f"envs.{task_name}")
        return getattr(module, task_name)()

    for task in tasks:
        if task not in seeds:
            print(json.dumps({"task": task, "error": "not in seedlist"}), flush=True)
            continue
        for seed in seeds[task][: args_cli.n_seeds]:
            row = {"task": task, "seed": seed}
            try:
                args = build_args(task, args_cli.task_config)
                env = class_decorator(task)
                real, t_real, err, extra = roll(env, args, seed, None)
                row.update(expert_s=round(t_real, 1), real=real, err=err, **extra)
                for strategy in strategies:
                    env = class_decorator(task)
                    got, t_s, e_s, _ = roll(env, args, seed, strategy)
                    row[strategy] = {
                        "s": round(t_s, 2),
                        "match": got == real,
                        "info": got,
                        "err": e_s,
                    }
            except Exception:
                row["fatal"] = traceback.format_exc(limit=4)
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
