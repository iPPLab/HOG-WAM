"""Real UMT5 cache for every Plus language variant, using the training encoder."""
import argparse
import json
from pathlib import Path
import shutil

from fastwam_tpu.inference.text import TextEmbeddingCache, Umt5Encoder, build_prompt

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--training-text-cache', type=Path,
                   help='Reuse matching embeddings from the LIBERO training text cache')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    target = args.root / 'data/plus_text_cache'
    target.mkdir(exist_ok=True)
    training_cache = args.training_text_cache or args.root / 'data/text_cache'
    for source in training_cache.glob('*.npz'):
        if not (target / source.name).exists():
            shutil.copy2(source, target / source.name)
    tasks = json.loads((args.root / 'data/plus_tasks.json').read_text())
    prompts = sorted({build_prompt(j['language']) for j in tasks})
    cache = TextEmbeddingCache(str(target))
    missing = [prompt for prompt in prompts if cache.get(prompt) is None]
    print('TEXT_REQUIREMENTS', len(prompts), 'missing', len(missing), flush=True)
    if missing:
        encoder = Umt5Encoder(device=args.device, dtype='float32')
        for i in range(0, len(missing), 4):
            group = missing[i:i+4]
            for prompt, (rows, length) in zip(group, encoder.encode(group), strict=True):
                cache.put(prompt, rows, length)
            print('PLUS_TEXT', min(i+4, len(missing)), '/', len(missing), flush=True)
    if any(cache.get(prompt) is None for prompt in prompts):
        raise RuntimeError('Incomplete text cache')
    (target / 'PLUS_TEXT_COMPLETE.json').write_text(json.dumps({
        'unique_prompts': len(prompts), 'dtype': 'float32', 'storage': 'float16',
        'repo': 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers', 'template': build_prompt('{instruction}')}, indent=2))
    print('PLUS_TEXT_COMPLETE', len(prompts), flush=True)
