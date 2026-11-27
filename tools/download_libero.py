"""Download the immutable official four-suite LIBERO demonstration release."""

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


REVISION = 'f13aa24a3da8c43c7225569f28c562979fa0e35a'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def download(root):
    root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id='yifengzhu-hf/LIBERO-datasets', repo_type='dataset', revision=REVISION,
        local_dir=str(root), allow_patterns=['README.md'] + [s + '/*.hdf5' for s in SUITES],
        max_workers=4,
    )
    counts = {s: len(list((root / s).glob('*.hdf5'))) for s in SUITES}
    if any(n != 10 for n in counts.values()):
        raise RuntimeError('Incomplete official release: ' + str(counts))
    report = {'source': 'yifengzhu-hf/LIBERO-datasets', 'revision': REVISION,
              'suites': counts, 'files': 40}
    (root.parent / 'DOWNLOAD_COMPLETE.json').write_text(json.dumps(report, indent=2))
    print('DOWNLOAD_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='HDF5 destination, e.g. /data/libero/raw')
    download(parser.parse_args().root)
