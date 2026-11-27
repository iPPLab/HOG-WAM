"""Local-only JAX LIBERO policy service; the simulator runs in its own Python env."""
import argparse
import dataclasses
import json
from multiprocessing.connection import Listener
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from fastwam_tpu.config import load_config
from fastwam_tpu.data.libero import compose_cameras, normalize, denormalize
from fastwam_tpu.inference.load import load_fastwam
from fastwam_tpu.inference.policy import FastWAMPolicy
from fastwam_tpu.inference.text import TextEmbeddingCache, build_prompt
from fastwam_tpu.models.wan.vae import load_vae


class MinMaxDimension:
    def __init__(self, stats):
        self.stats, self.dim = stats, len(stats['min'])

    def forward(self, values):
        return normalize(values, self.stats)

    def backward(self, values):
        return denormalize(values, self.stats)


class LiberoPolicy(FastWAMPolicy):
    def canvas_to_tensor(self, cameras):
        # Use precisely the training float32 antialiased resize, without uint8
        # re-quantization, flip, rotation, or RoboTwin's three-camera composition.
        agent, wrist = cameras
        image = compose_cameras(np.asarray(agent)[None], np.asarray(wrist)[None])
        return jnp.asarray(image[:, 0][None])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model', choices=['baseline', 'hog', 'rgb_to_hog'], required=True)
    p.add_argument('--training-config', type=Path, required=True,
                   help='Explicit training recipe for a new, isolated checkpoint')
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='Explicit committed final BF16 inference export')
    p.add_argument('--socket', required=True)
    p.add_argument('--ready-file', type=Path)
    a = p.parse_args()
    cfg = load_config(a.training_config.resolve())
    training_hog_init = cfg.model.hog_dit_init
    towers = 'rgb_hog_action' if a.model == 'hog' else 'rgb_action'
    visual = 'rgb_to_hog' if a.model == 'rgb_to_hog' else 'rgb'
    if (cfg.data.format, cfg.model.training_towers, cfg.model.visual_input_mode) != (
        'libero_hdf5', towers, visual
    ):
        p.error('--model and --training-config do not describe the same LIBERO policy')
    # Both deployment snapshots deliberately omit the removable HOG tower.
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, training_towers='rgb_action'))
    stats_blob = Path(cfg.data.l3.stats_path).read_bytes()
    stats = json.loads(stats_blob)
    from types import SimpleNamespace
    dimensions = SimpleNamespace(action=MinMaxDimension(stats['action']),
                                 proprio=MinMaxDimension(stats['state']))
    ckpt = a.checkpoint.resolve()
    if not (ckpt / 'commit_success.txt').exists():
        raise RuntimeError('Refuse incomplete final checkpoint: ' + str(ckpt))
    if json.loads((ckpt / 'commit_success.txt').read_text())['step'] != 13200:
        p.error('LIBERO Plus paper evaluation requires the step-13200 export')
    print('LOADING', a.model, ckpt, flush=True)
    vae = load_vae(cfg.model.weights.vae_orbax, dtype=jnp.bfloat16, weights_dtype=jnp.bfloat16)
    model = load_fastwam(str(ckpt), cfg, dtype=jnp.bfloat16)
    policy = LiberoPolicy(model=model, vae=vae, stats=dimensions,
        action_horizon=32, canvas_hw=(224, 448), latent_compat=False)
    assert policy.visual_input_mode == ('rgb_to_hog' if a.model == 'rgb_to_hog' else 'rgb')
    print('DEPLOYMENT_CONTRACT', json.dumps({'model': a.model, 'checkpoint': str(ckpt),
        'visual_input_mode': policy.visual_input_mode, 'training_towers': cfg.model.training_towers,
        'training_hog_dit_init': training_hog_init,
        'training_hog_condition_mode': cfg.model.hog_condition_mode,
        'proprio_dim': 8, 'action_dim': 7, 'predict_steps': 32, 'execute_steps': 8,
        'denoise_steps': 10, 'latent_compat': False}), flush=True)
    cache = TextEmbeddingCache(str(a.root / 'data/plus_text_cache'))
    # Warm the exact camera/proprio/text shapes before advertising READY.
    entries = sorted((a.root / 'data/plus_text_cache').glob('*.npz'))
    with np.load(entries[0], allow_pickle=False) as item:
        prompt = json.loads(str(item['meta']))['prompt']
    context = cache.get(prompt)
    agent = np.zeros((128, 128, 3), np.uint8)
    state = (np.asarray(stats['state']['min']) + np.asarray(stats['state']['max'])) / 2
    t0 = time.monotonic()
    action = policy.infer_action((agent, agent), state, context, np.ones(128, bool))
    assert action.shape == (32, 7) and np.isfinite(action).all()
    print('WARMUP_COMPLETE', time.monotonic() - t0, 'action_range', action.min(), action.max(), flush=True)
    policy.reset()
    socket = Path(a.socket)
    if socket.exists():
        raise RuntimeError('Socket already exists: ' + str(socket))
    with Listener(str(socket), family='AF_UNIX') as listener:
        socket.chmod(0o600)
        if a.ready_file:
            a.ready_file.write_text(json.dumps({'pid': __import__('os').getpid(),
                'model': a.model, 'checkpoint': str(ckpt), 'visual_input_mode': policy.visual_input_mode,
                'training_hog_dit_init': training_hog_init,
                'training_hog_condition_mode': cfg.model.hog_condition_mode,
                'socket': str(socket), 'warmup_seconds': time.monotonic() - t0,
                'action_shape': list(action.shape), 'all_actions_finite': True}, indent=2))
        print('POLICY_READY', a.model, a.socket, flush=True)
        while True:
            with listener.accept() as connection:
                while True:
                    try:
                        request = connection.recv()
                    except EOFError:
                        break
                    try:
                        if request['op'] == 'reset':
                            policy.reset()
                            response = {'ok': True}
                        elif request['op'] == 'infer':
                            context = cache.get(build_prompt(request['language']))
                            if context is None:
                                raise ValueError('Missing real UMT5 embedding: ' + request['language'])
                            t0 = time.monotonic()
                            action = policy.infer_action((request['agent'], request['wrist']),
                                request['state'], context, np.ones(128, bool))
                            if not np.isfinite(action).all():
                                raise ValueError('Non-finite model action')
                            response = {'ok': True, 'action': action,
                                        'inference_seconds': time.monotonic() - t0}
                        else:
                            raise ValueError('Unknown request')
                    except Exception as exc:
                        response = {'ok': False, 'error': repr(exc)}
                    try:
                        connection.send(response)
                    except (BrokenPipeError, ConnectionResetError, EOFError):
                        # A simulator can die in native EGL code. Keep the loaded
                        # policy alive for its replacement, not just this client.
                        break


if __name__ == '__main__':
    main()
