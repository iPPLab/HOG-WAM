# RoboTwin evaluation client adapted for the HOG-WAM policy adapter.
# Launch from the pinned RoboTwin root so environment, policy, and asset imports
# resolve there. Setup links the script-relative task_config directory.
# This client supports skipping redundant rendering and collecting episode info
# when an explicitly selected scene seed does not require an expert pre-roll.
import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
# Living outside the shared script/ dir costs us the one path python gives a script for
# free: its own directory. `from test_render import Sapien_TEST` (and anything else in
# script/) resolves through this. cwd is the shared RoboTwin root, so ./script IS the
# directory this file was copied out of.
sys.path.append("./script")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *


import sys
import os
import subprocess
import socket
import json
import threading
import time
import random
import traceback
import yaml
from datetime import datetime
import importlib
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import json
from typing import Any

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

import numpy as np
import json
from typing import Any
import base64

class NumpyEncoder(json.JSONEncoder):
    """Enhanced json encoder for numpy types with array reconstruction info"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float32:
                dtype = 'float32'
            elif obj.dtype == np.float64:
                dtype = 'float64'
            elif obj.dtype == np.int32:
                dtype = 'int32'
            elif obj.dtype == np.int64:
                dtype = 'int64'
            else:
                dtype = str(obj.dtype)

            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'dtype': dtype,
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_to_json(data: Any) -> str:
    """Convert numpy-containing data to JSON string with reconstruction info"""
    return json.dumps(data, cls=NumpyEncoder)

def json_to_numpy(json_str: str) -> Any:
    """Convert JSON string back to Python objects with numpy arrays"""
    def object_hook(dct):
        if '__numpy_array__' in dct:
            data = base64.b64decode(dct['data'])
            return np.frombuffer(data, dtype=dct['dtype']).reshape(dct['shape'])
        return dct

    return json.loads(json_str, object_hook=object_hook)

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name, conda_env=None):
    # conda_env is abandoned
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def _harvest_episode_info(TASK_ENV):
    """The object-identity map (`{"{A}": "060_kitchenpot/base0", "{a}": "left"}`) without
    running the expert.

    Each task's `play_once()` records the map as its last statement, but the values come
    from `load_actors()` -- model ids drawn during `setup_demo` -- and from arm tags
    computed off the initial geometry. None of it depends on the motion, so the map is
    reachable by letting `play_once()` fall straight through.

    `plan_success = False` is what makes it fall through: `move()` (_base_task.py:908)
    and `grasp_actor()` (:1176) both return immediately on it, so nothing is planned and
    nothing is stepped. The flag is restored before returning -- the caller checks it.

    An empty map is NOT treated as failure here: `handover_block.play_once()` never
    writes one, so `{}` is what the expert produces for it too, and its templates carry
    no placeholders to fill. What actually has to hold is that the drawn sentence comes
    out fully substituted, and that is asserted where the sentence exists (see the
    `"{"`-in-instruction check below) rather than guessed at from the map.
    """
    saved = TASK_ENV.plan_success
    TASK_ENV.plan_success = False
    try:
        return TASK_ENV.play_once() or {"info": {}}
    finally:
        TASK_ENV.plan_success = saved


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

class ModelClient:
    def __init__(self, host='localhost', port=9999, timeout=None):
        """`TW_RPC_TIMEOUT` seconds per RPC; 300 if unset, which is the upstream value.

        On a cache miss, `set_task` waits for a CPU UMT5-XXL encode. If a cold
        cache and CPU contention cause timeouts, increase TW_RPC_TIMEOUT.
        """
        self.host = host
        self.port = port
        if timeout is None:
            timeout = float(os.environ.get("TW_RPC_TIMEOUT", 300))
        self.timeout = timeout
        self.sock = None
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = 1000
        retry_delay = 5

        while attempts < max_attempts:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                attempts += 1
                if self.sock:
                    self.sock.close()
                if attempts < max_attempts:
                    print(f"⚠️ Connection attempt {attempts} failed: {str(e)}")
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to server after {max_attempts} attempts: {str(e)}"
                    )

    def _send_recv(self, data):
        """Send request and receive response with numpy array support"""
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode('utf-8')

            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, 'big'))
            self.sock.sendall(json_data)

            # Receive and deserialize response
            response = self._recv_response()
            return response

        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _recv_response(self):
        """Receive response with numpy array reconstruction"""
        # Read response length
        len_data = self.sock.recv(4)
        if not len_data:
            raise ConnectionError("Connection closed by server")

        size = int.from_bytes(len_data, 'big')

        # Read complete response
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)

        # Deserialize with numpy reconstruction
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        return response['res']

    def close(self):
        """Close the connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
                print("🔌 Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    port = usr_args["port"]
    save_dir = None
    video_save_dir = None
    video_size = None

    policy_conda_env = usr_args.get("policy_conda_env", None)

    get_model = eval_function_decorator(policy_name, "get_model", conda_env=policy_conda_env)

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    # --- canonical seed mode (absolute seed controls) -------------------------
    # 上游默认: st_seed = 100000*(1+seed) 只是个"起点", 之后 expert_check 会一直
    # 往后滑到第一个专家能解出来的场景为止(:391-421 的 `now_seed += 1; continue`)。
    # 但专家用的是 mplib/curobo 的随机采样规划器, 同一场景这次能解、下次可能解不了
    # —— 实测 13 轮归档日志, 两轮之间 seed 列表平均只有 91.9% 重叠(25 集里 23 集相同,
    # 最差 16 集)。也就是说"同一起始 seed = 同一批 episode"这个假设是错的,
    # 光这一项就贡献 ±4pp、最差 ±16pp 的分数波动, 而它与模型无关。
    #
    # abs_seed: 直接指定要跑的那个场景, 绕开 100000*(1+seed) 这层映射。
    # no_expert_check: 跳过专家预跑。配合一份"参考轮里被专家解出过"的 canonical
    # 列表使用 —— 场景可解是已知事实, 这一次专家做不出来是规划器抽风而非场景变难,
    # 用它把一道好题作废没有道理。副作用: 每集从跑两遍(专家+模型)变成一遍。
    #
    # 两个键都不给 = 上游原样行为。
    abs_seed = usr_args.get("abs_seed", None)
    st_seed = int(abs_seed) if abs_seed is not None else 100000 * (1 + seed)
    expert_check = not bool(usr_args.get("no_expert_check", False))
    if abs_seed is not None or not expert_check:
        print(f"\033[36m[canonical] st_seed={st_seed} expert_check={expert_check}\033[0m", flush=True)
    # -------------------------------------------------------------------------
    suc_nums = []
    test_num = int(usr_args.get("test_num", 100))
    topk = 1

    # model = get_model(usr_args)
    model = ModelClient(port=port)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   policy_conda_env=policy_conda_env,
                                   expert_check=expert_check)
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                policy_conda_env=None,
                expert_check=True):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    # expert_check 上游是硬编码 True(:370)。改成参数, 默认仍是 True = 上游行为。
    # False 时 :391 的预跑整段被跳过, :415 的条件短路为真 -> now_seed 不再滑动,
    # 跑的就是传进来的那个 st_seed 本身。
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval", conda_env=policy_conda_env)

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue
        else:
            # `--no_expert_check` cannot simply skip the block above: `episode_info` is
            # bound ONLY inside it, and :466 reads it unconditionally to build the
            # instruction pool. As shipped, the flag raises NameError on the first
            # episode -- no archived log has ever printed `[canonical] st_seed=`, which
            # is consistent with the path never having been exercised.
            #
            # Harvest the same map without paying for the expert. Same shape as the
            # branch above (setup -> use -> close) so the model episode below still
            # starts from its own fresh scene and inherits nothing from this one.
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info = _harvest_episode_info(TASK_ENV)
            TASK_ENV.close_env()

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
        instruction = np.random.choice(results[0][instruction_type])
        # The one invariant that makes skipping the expert safe. `generate_episode_
        # descriptions` fills `{A}`/`{a}` placeholders from the object-identity map; hand
        # it a map that is missing an entry and it returns the template with the brace
        # still in it, no error. FastWAM is conditioned on this sentence, so that would
        # be a silent scoring bug on every episode. Fail loudly instead.
        if "{" in str(instruction):
            raise RuntimeError(
                f"instruction still has an unfilled placeholder: {instruction!r}. The "
                "object-identity map did not cover this task's template."
            )
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False
        model.call(func_name='reset_model')
        # Gated render skip (FastWAM M3). Off unless TW_SKIP_OBS_IN_REPLAN=1, so every
        # other policy and every running campaign behaves exactly as before. The lookup
        # is hoisted out of the loop: eval_function_decorator imports the policy module.
        _skip_obs = os.environ.get("TW_SKIP_OBS_IN_REPLAN") == "1"
        _should_obs = None
        if _skip_obs:
            try:
                # BOTH halves are required, and this one is the easy one to forget.
                # The line above resets the SERVER; the official client also resets the
                # POLICY MODULE (eval_policy.py:365, `reset_func(model)`), and this fork
                # dropped that. Without it the policy's action queue still holds the
                # PREVIOUS episode's leftover rows when should_request_observation() is
                # first consulted, so step 0 of every episode after the first answers
                # "no render needed" and then has no frame to replan from.
                eval_function_decorator(
                    policy_name, "reset_model", conda_env=policy_conda_env)(model)
                _should_obs = eval_function_decorator(
                    policy_name, "should_request_observation", conda_env=policy_conda_env)
            except Exception as _exc:
                print(f"[skip-obs] {policy_name} does not support the render skip "
                      f"({_exc!r}); rendering every step", flush=True)
                _should_obs = None
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            need_obs = True
            if _should_obs is not None:
                need_obs = bool(_should_obs())
            observation = TASK_ENV.get_obs() if need_obs else None
            eval_func(TASK_ENV, model, observation)
            if TASK_ENV.eval_success:
                succ = True
                break
        # task_total_reward += TASK_ENV.episode_score
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config['port'] = args.port

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
