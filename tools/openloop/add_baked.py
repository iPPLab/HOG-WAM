"""Add the .twp's own BAKED text row to the probe set.

If the 30k run had actually trained on `text_mode=baked` (bare instruction, Wan2.1)
rather than the templated side table, the model would prefer THIS embedding. That is a
decisive local test of which conditioning it was trained under.
"""
import os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fastwam_tpu.data import twp

TMP = Path(os.environ.get("FASTWAM_OPENLOOP_DIR", ROOT / "outputs" / "openloop"))
z = dict(np.load(TMP / "probe_set.npz", allow_pickle=True))
ep_ids = z["episode_id"]

cache = {}
for e in sorted(set(int(i) for i in ep_ids)):
    buf = open(TMP / "eps" / f"episode_{e:06d}.twp", "rb").read()
    epi = twp.Episode(buf, expect_id=e)
    te = epi.text_emb
    print(f"ep {e}: text_emb {te.shape} dtype={te.dtype}")
    if te.shape[0] == 0:
        raise SystemExit("this L3 carries no baked text; nothing to test")
    row = np.asarray(te[0], np.float32)
    print(f"   |x|max={np.abs(row).max():.3f} nonzero rows={int((np.abs(row).sum(1) > 0).sum())}")
    cache[e] = row

z["baked_ctx"] = np.stack([cache[int(i)] for i in ep_ids])
tr = z["train_ctx"]
bk = z["baked_ctx"]
print(f"\nsidetable ctx: |x|max={np.abs(tr).max():.3f} mean|x|={np.abs(tr).mean():.4f}")
print(f"baked     ctx: |x|max={np.abs(bk).max():.3f} mean|x|={np.abs(bk).mean():.4f}")
print(f"they differ by L2 {np.sqrt(((tr - bk) ** 2).mean()):.4f}")
np.savez_compressed(TMP / "probe_set.npz", **z)
print(f"\nrewrote {TMP / 'probe_set.npz'} with baked_ctx")
