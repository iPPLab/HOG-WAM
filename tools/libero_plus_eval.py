"""Official LIBERO Plus variants: one paired, resumable trial per task and model."""
import argparse
from collections import deque
import contextlib
import hashlib
import io
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np

SUITES = ['libero_spatial', 'libero_object', 'libero_goal', 'libero_10']
LIMITS = dict(zip(SUITES, [220, 280, 300, 520]))


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def prepare(root, training_index):
    # Import from the explicitly supplied Plus PYTHONPATH, not the original LIBERO.
    from libero.libero.benchmark import task_maps
    repo = root / 'LIBERO-plus'
    categories = json.loads((repo / 'libero/libero/benchmark/task_classification.json').read_text())
    training = json.loads(training_index.read_text())['episodes']
    canonical = {}
    for episode in training:
        suite, name = episode['task'].split('/', 1)
        canonical.setdefault(suite, {})[name.removesuffix('_demo')] = episode['instruction']
    jobs = []
    for suite in SUITES:
        mapping = {item['name']: item for item in categories[suite]}
        for task_id, task in enumerate(task_maps[suite].values()):
            c = mapping[task.name]
            # Plus's filename-to-language helper appends table/view/initstate/noise
            # numbers to non-language instructions. Those are benchmark metadata,
            # not commands. Hold language fixed except in the language category.
            matches = [name for name in canonical[suite] if task.name.startswith(name)]
            if not matches:
                raise ValueError('Cannot associate Plus variant with a trained task: ' + task.name)
            base_name = max(matches, key=len)
            language = task.language if c['category'] == 'Language Instructions' else canonical[suite][base_name]
            jobs.append({'suite': suite, 'task_id': task_id, 'name': task.name,
                'language': language, 'benchmark_language_raw': task.language,
                'base_task': base_name, 'category': c['category'],
                'difficulty': c['difficulty_level'], 'bddl_file': task.bddl_file,
                'max_steps': LIMITS[suite], 'seed': 42 + len(jobs), 'trial': 0})
    if len(jobs) != 10030:
        raise ValueError(f'Unexpected official Plus size: {len(jobs)}')
    atomic_json(root / 'data/plus_tasks.json', jobs)
    print('MANIFEST_COMPLETE', len(jobs), 'unique_instructions', len({j['language'] for j in jobs}), flush=True)


def evaluate(args):
    import torch
    import robosuite.macros as macros
    from robosuite.utils.transform_utils import quat2axisangle
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    # Ensure cleanup uses the owning EGL context. This is a defensive local
    # compatibility shim, not proof that all native driver faults originate here.
    # Do not alter other users' shared simulator packages.
    from robosuite.utils.binding_utils import MjRenderContext
    def release_render_context(self):
        con, gl = getattr(self, 'con', None), getattr(self, 'gl_ctx', None)
        if gl is not None:
            gl.make_current()
        if con is not None:
            con.free()
            self.con = None
        if gl is not None:
            gl.free()
            self.gl_ctx = None
    MjRenderContext.__del__ = release_render_context
    # Training HDF5 uses raw OpenGL orientation. Pin this instead of relying on a
    # robosuite global modified by another application's screenshot convention.
    macros.IMAGE_CONVENTION = 'opengl'
    jobs = json.loads((args.root / 'data/plus_tasks.json').read_text())
    if args.smoke:
        # A real standard task from Plus, exercising camera/robot/noise decoding.
        jobs = [next(j for j in jobs if j['category'] == 'Camera Viewpoints')]
    else:
        jobs = [j for i, j in enumerate(jobs) if i % args.shards == args.shard]
    if args.task_limit:
        jobs = jobs[:args.task_limit]
    results = args.root / args.evaluation_dir / (args.result_subdir or ('smoke' if args.smoke else 'results')) / args.model
    benchmarks = {}
    def benchmark(suite):
        if suite not in benchmarks:
            with contextlib.redirect_stdout(io.StringIO()):
                benchmarks[suite] = get_benchmark(suite)(task_order_index=0)
        return benchmarks[suite]
    connection = Client(args.socket, family='AF_UNIX')
    def request(payload):
        connection.send(payload)
        result = connection.recv()
        if not result['ok']:
            raise RuntimeError(result['error'])
        return result
    completed = successes = 0
    for job in jobs:
        path = results / job['suite'] / f"{job['task_id']:05d}.json"
        if path.exists() and json.loads(path.read_text()).get('status') == 'complete':
            continue
        start = time.time()
        env = None
        try:
            random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
            suite = benchmark(job['suite'])
            task = suite.get_task(job['task_id'])
            assert task.name == job['name'] and task.language == job['benchmark_language_raw']
            env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(job['task_id']),
                camera_heights=128, camera_widths=128, render_gpu_device_id=0)
            env.seed(job['seed'])
            env.reset()
            states = suite.get_task_init_states(job['task_id'])
            obs = env.set_init_state(states[0])
            # Same 10-step settling and native -1 (open) gripper for both models.
            for _ in range(10):
                obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
            request({'op': 'reset'})
            queue = deque()
            inference_seconds = []
            success = False
            for step in range(job['max_steps']):
                if not queue:
                    state = np.concatenate([obs['robot0_eef_pos'],
                        quat2axisangle(obs['robot0_eef_quat']), obs['robot0_gripper_qpos']]).astype(np.float32)
                    if state.shape != (8,):
                        raise ValueError(f'Bad proprio {state.shape}')
                    if step == 0:
                        diagnostic = results / 'inputs' / f"{job['suite']}_{job['task_id']:05d}.npz"
                        diagnostic.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(diagnostic, agent=obs['agentview_image'],
                            wrist=obs['robot0_eye_in_hand_image'], state=state)
                    reply = request({'op': 'infer', 'agent': obs['agentview_image'],
                        'wrist': obs['robot0_eye_in_hand_image'], 'state': state, 'language': job['language']})
                    chunk = np.asarray(reply['action'], np.float32)
                    assert chunk.shape == (32, 7) and np.isfinite(chunk).all()
                    # Native robosuite OSC action space, no OpenVLA gripper flip,
                    # no qpos integration. Controller accepts normalized [-1,1].
                    chunk = np.clip(chunk, -1, 1)
                    chunk[:, 6] = np.where(chunk[:, 6] >= 0, 1., -1.)
                    queue.extend(chunk[:args.replan])
                    inference_seconds.append(reply['inference_seconds'])
                    if len(inference_seconds) == 1:
                        print('FIRST_ACTION', args.model, job['suite'], job['task_id'], chunk[0].tolist(), flush=True)
                obs, reward, done, info = env.step(queue.popleft().tolist())
                success = bool(env.check_success())
                if success or done or (args.smoke and step >= 15):
                    break
            result = dict(job, status='complete', success=success, steps=step + 1,
                seconds=time.time() - start, replans=len(inference_seconds),
                inference_seconds=sum(inference_seconds), model=args.model,
                checkpoint_step=13200, replan_steps=args.replan,
                visual_input_mode='rgb_to_hog' if args.model == 'rgb_to_hog' else 'rgb',
                mujoco_version=__import__('mujoco').__version__,
                render_gpu=os.environ.get('MUJOCO_EGL_DEVICE_ID'),
                timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
            atomic_json(path, result)
            completed += 1; successes += success
            print('EPISODE_RESULT', json.dumps(result), flush=True)
        except Exception:
            atomic_json(path.with_suffix('.error.json'), dict(job, status='error', traceback=traceback.format_exc(), seconds=time.time()-start))
            traceback.print_exc()
            # Infrastructure exceptions are never counted as policy failures.
            raise
        finally:
            if env is not None:
                env.close()
                env = None
                import gc
                gc.collect()
    connection.close()
    print('SHARD_COMPLETE', args.model, args.shard, completed, successes, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--training-index', type=Path, help='Prepared LIBERO index.json for --prepare')
    p.add_argument('--model', choices=['baseline', 'hog', 'rgb_to_hog'])
    p.add_argument('--socket')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--replan', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--task-limit', type=int, default=0)
    p.add_argument('--result-subdir')
    p.add_argument('--evaluation-dir', default='eval_plus')
    args = p.parse_args()
    if args.prepare:
        prepare(args.root, args.training_index or args.root / 'data/raw/index.json')
    else:
        if not args.model or not args.socket:
            p.error('evaluation requires --model and --socket')
        if not 0 <= args.shard < args.shards:
            p.error('require 0 <= --shard < --shards')
        evaluate(args)
