"""Index official LIBERO HDF5 and compute native action/state statistics.

Creates small metadata/numeric files only; images stay in the original release.
Text encoding is a separate --encode-text stage, run in the existing Torch env.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fastwam_tpu.data.libero import SUITES, retained_indices
from fastwam_tpu.inference.text import TextEmbeddingCache, Umt5Encoder, build_prompt


def prepare(root):
    import h5py
    marker = root.parent / 'DOWNLOAD_COMPLETE.json'
    source = json.loads(marker.read_text())
    files = [p for suite in SUITES for p in sorted((root / suite).glob('*.hdf5'))]
    if len(files) != 40:
        raise ValueError('Expected exactly 40 official task files, found ' + str(len(files)))
    episodes, all_actions, all_states, all_indices = [], [], [], []
    offset = removed = 0
    for task_id, path in enumerate(files):
        with h5py.File(path, 'r') as f:
            group = f['data']
            language = json.loads(group.attrs['problem_info'])['language_instruction']
            for demo in sorted(group, key=lambda x: int(x.split('_')[-1])):
                d = group[demo]
                actions = np.asarray(d['actions'], np.float32)
                state = np.concatenate([d['obs']['ee_states'][:], d['obs']['gripper_states'][:]], axis=1)
                if state.shape != (len(actions), 8) or not np.isfinite(state).all():
                    raise ValueError('Invalid aligned LIBERO state: ' + str(path) + '/' + demo)
                if any(len(d['obs'][cam]) != len(actions) for cam in ['agentview_rgb', 'eye_in_hand_rgb']):
                    raise ValueError('Camera/action alignment mismatch')
                keep = retained_indices(actions)
                removed += len(actions) - len(keep)
                if not len(keep):
                    raise ValueError('Empty episode')
                episodes.append({'file': str(path.relative_to(root)), 'task': path.parent.name + '/' + path.stem,
                                 'instruction': language, 'demo': demo, 'offset': offset, 'length': len(keep)})
                all_actions.append(actions[keep]); all_states.append(state[keep].astype(np.float32))
                all_indices.append(keep); offset += len(keep)
        print('INDEX', task_id + 1, '/', 40, path.name, flush=True)
    actions, states, indices = np.concatenate(all_actions), np.concatenate(all_states), np.concatenate(all_indices)
    payload = {'source': source, 'episodes': episodes, 'removed_noops': removed}
    blob = json.dumps(payload, sort_keys=True).encode()
    stats = {'dataset': 'libero', 'index_sha256': hashlib.sha256(blob).hexdigest(),
             'episodes': len(episodes), 'transitions': len(actions)}
    for name, values in [('action', actions), ('state', states)]:
        stats[name] = {'min': values.min(axis=0).tolist(), 'max': values.max(axis=0).tolist()}
    np.savez(root / 'numeric.npz', actions=actions, states=states, raw_indices=indices)
    (root / 'index.json').write_bytes(blob)
    (root / 'dataset_stats.json').write_text(json.dumps(stats, indent=2))
    print('PREPARED', json.dumps({k: v for k, v in stats.items() if k not in ['action', 'state']}), flush=True)


def encode_text(root, cache_dir, device='cuda'):
    manifest = json.loads((root / 'index.json').read_text())
    prompts = sorted({build_prompt(e['instruction']) for e in manifest['episodes']})
    cache = TextEmbeddingCache(str(cache_dir))
    missing = [p for p in prompts if cache.get(p) is None]
    if missing:
        encoder = Umt5Encoder(device=device, dtype='float32')
        for i in range(0, len(missing), 4):
            group = missing[i:i + 4]
            for prompt, (rows, length) in zip(group, encoder.encode(group), strict=True):
                cache.put(prompt, rows, length)
            print('TEXT', min(i + 4, len(missing)), '/', len(missing), flush=True)
    (cache_dir / 'LIBERO_TEXT_COMPLETE.json').write_text(json.dumps({
        'prompts': len(prompts), 'encoder': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers',
        'dtype': 'float32', 'template': build_prompt('{instruction}'),
    }, indent=2))
    print('TEXT_COMPLETE', len(prompts), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--encode-text', type=Path)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.encode_text is not None:
        encode_text(a.root, a.encode_text, a.device)
    else:
        prepare(a.root)
