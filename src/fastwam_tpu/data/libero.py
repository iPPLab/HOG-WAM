"""Native LIBERO HDF5 training and deployment contract (no RoboTwin conversion).

Two RGB cameras are concatenated horizontally at 224x448. The release's OpenGL
image orientation is retained verbatim; deployment must pass the same raw camera
arrays without a second flip. Actions are native seven-dimensional OSC commands,
not joint positions. State is [eef xyz, axis-angle xyz, two gripper positions].
"""

import concurrent.futures
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .canvas import resize_bilinear_aa
from ..inference.text import TextEmbeddingCache, build_prompt


SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def retained_indices(actions, threshold=1e-4):
    """Drop motionless actions only when the gripper command also stays unchanged.

    Frame zero is retained. A gripper-only open/close command must never disappear.
    The same retained indices select observations, actions and proprioception.
    """
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all():
        raise ValueError('LIBERO actions must be finite [T,7]')
    keep = np.ones(len(a), bool)
    if len(a) > 1:
        keep[1:] = (np.linalg.norm(a[1:, :6], axis=1) >= threshold) | (a[1:, 6] != a[:-1, 6])
    return np.flatnonzero(keep).astype(np.int64)


def normalize(values, stats):
    lo, hi = np.asarray(stats['min'], np.float32), np.asarray(stats['max'], np.float32)
    span = hi - lo
    # Constant dimensions map to zero, not to an arbitrary end of [-1,1].
    return np.where(span > 1e-6,
                    np.clip(2 * (values - lo) / np.maximum(span, 1e-6) - 1, -1, 1), 0).astype(np.float32)


def denormalize(values, stats):
    lo, hi = np.asarray(stats['min'], np.float32), np.asarray(stats['max'], np.float32)
    return ((np.asarray(values, np.float32) + 1) * (hi - lo) / 2 + lo).astype(np.float32)


def compose_cameras(agentview, wrist):
    if agentview.shape != wrist.shape or agentview.ndim != 4 or agentview.shape[-1] != 3:
        raise ValueError('LIBERO needs matching [T,H,W,3] RGB camera streams')
    cameras = [resize_bilinear_aa(x.astype(np.float32) / 255, 224, 224)
               for x in (agentview, wrist)]
    return (np.concatenate(cameras, axis=2) * 2 - 1).transpose(3, 0, 1, 2).copy()


class LiberoHdf5Dataset:
    """Implements the same sample/seek/data_spec interface as RobotwinTwpDataset.

    Numeric arrays and valid frame indices are prepared once. Only nine RGB frames
    per camera/window are read from the immutable HDF5 demonstrations. Random draws
    depend solely on the trainer-supplied generator, including after resume.
    """

    def __init__(self, prefix, *, stats_blob, stats_source, text_dir, task_filter=None,
                 num_frames=33, action_video_freq_ratio=4, video_size=(224, 448),
                 split='train', val_set_proportion=0, n_hosts=1, decode_threads=8, **unused):
        if split != 'train' or val_set_proportion != 0 or n_hosts != 1:
            raise ValueError('Initial LIBERO interface supports single-host, full training split only')
        if num_frames != 33 or action_video_freq_ratio != 4 or tuple(video_size) != (224, 448):
            raise ValueError('LIBERO contract: 32 actions, 9 stride-four frames, 224x448 canvas')
        self.root = Path(prefix)
        manifest_bytes = (self.root / 'index.json').read_bytes()
        self.manifest = json.loads(manifest_bytes)
        self.stats = json.loads(stats_blob)
        self.fingerprint = hashlib.sha256(manifest_bytes).hexdigest()
        if self.stats['index_sha256'] != self.fingerprint:
            raise ValueError('LIBERO statistics and episode index have different fingerprints')
        self.episodes = [e for e in self.manifest['episodes']
                         if task_filter is None or e['task'] in task_filter]
        if not self.episodes:
            raise ValueError('Empty LIBERO task selection')
        if task_filter is not None and set(task_filter) != {e['task'] for e in self.episodes}:
            raise ValueError('Unknown LIBERO task in task_filter')
        self._numeric = np.load(self.root / 'numeric.npz')
        self._actions = self._numeric['actions']
        self._states = self._numeric['states']
        self._raw_indices = self._numeric['raw_indices']
        counts = np.asarray([e['length'] for e in self.episodes], np.int64)
        self.index = SimpleNamespace(num_windows=int(counts.sum()))
        self._probabilities = counts / counts.sum()
        cache = TextEmbeddingCache(str(text_dir))
        self._contexts = {}
        for e in self.episodes:
            task = e['task']
            if task not in self._contexts:
                context = cache.get(build_prompt(e['instruction']))
                if context is None:
                    raise ValueError('Missing real UMT5 embedding for ' + task)
                self._contexts[task] = context
        self._threads = concurrent.futures.ThreadPoolExecutor(max_workers=decode_threads)
        self.stats_source = stats_source
        self._closed = False
        print('[data] LIBERO ' + json.dumps(self.data_spec()), flush=True)

    def seek(self, step):
        self._step = int(step)
        return self

    def start(self):
        return self

    def _sample(self, selection):
        import h5py
        episode_id, start = selection
        e = self.episodes[episode_id]
        n, offset = e['length'], e['offset']
        action_steps = start + np.arange(32)
        video_steps = start + np.arange(9) * 4
        ai = offset + np.minimum(action_steps, n - 1)
        vi = self._raw_indices[offset + np.minimum(video_steps, n - 1)]
        unique, inverse = np.unique(vi, return_inverse=True)
        with h5py.File(self.root / e['file'], 'r') as f:
            obs = f['data'][e['demo']]['obs']
            agentview = obs['agentview_rgb'][unique][inverse]
            wrist = obs['eye_in_hand_rgb'][unique][inverse]
        return {
            'video': compose_cameras(agentview, wrist),
            'action': normalize(self._actions[ai], self.stats['action']),
            'proprio': normalize(self._states[ai], self.stats['state']),
            'context': self._contexts[e['task']],
            'context_mask': np.ones(128, bool),
            'image_is_pad': video_steps >= n,
            'action_is_pad': action_steps >= n,
        }

    def sample_batch(self, rng, batch_size):
        ids = rng.choice(len(self.episodes), size=batch_size, p=self._probabilities)
        selections = [(int(i), int(rng.integers(self.episodes[i]['length']))) for i in ids]
        samples = list(self._threads.map(self._sample, selections))
        return {k: np.stack([s[k] for s in samples]) for k in samples[0]}

    def data_spec(self):
        return {
            'dataset': 'LIBERO official HDF5 four suites', 'source': self.manifest['source'],
            'index_sha256': self.fingerprint, 'episodes': len(self.episodes),
            'tasks': len(self._contexts), 'windows': self.index.num_windows,
            'canvas': [224, 448], 'cameras': ['agentview_rgb', 'eye_in_hand_rgb'],
            'image_orientation': 'as-stored OpenGL; NO flip/rotation at training or deployment',
            'source_color_order': 'rgb', 'video_frames': 9, 'video_stride': 4,
            'action_horizon': 32, 'action_dim': 7, 'proprio_dim': 8,
            'action_contract': 'OSC_POSE delta xyz/axis-angle, native gripper -1/+1; no temporal shift',
            'proprio_contract': 'eef xyz + axis-angle xyz + gripper qpos[2]',
            'normalization': 'per-dimension dataset min/max to [-1,1]; constants -> 0',
            'noop_policy': 'drop zero six-axis motion only if gripper unchanged; retain first row',
            'stats_source': self.stats_source, 'validation': 'none; evaluation uses held-out resets/Plus perturbations',
        }

    def close(self):
        if not self._closed:
            self._threads.shutdown(wait=True)
            self._numeric.close()
            self._closed = True
