import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TMP = Path(os.environ.get("FASTWAM_OPENLOOP_DIR", ROOT / "outputs" / "openloop"))
tag = sys.argv[1] if len(sys.argv) > 1 else "ours"
z = np.load(TMP / f"openloop_{tag}.npz")
p = np.load(TMP / "probe_set.npz", allow_pickle=True)
meta = json.loads(str(p["meta"]))
gt1, qpos = z["gt1"], z["qpos"]
arms = [k for k in z.files if k not in ("qpos", "gt1", "gt0")]
n = gt1.shape[0]


def l2(a, b, ax=None):
    return np.sqrt(((a - b) ** 2).mean(axis=ax))


print(f"=== {tag}: per-window L2 vs gt (shift1) ===")
hold = np.repeat(qpos[:, None, :], gt1.shape[1], axis=1)
hdr = f"{'win':<4}{'ep':>4}{'start':>6}  " + "".join(f"{a[:11]:>12}" for a in arms) + f"{'HOLD':>12}"
print(hdr)
for i in range(n):
    row = "".join(f"{l2(z[a][i], gt1[i]):>12.4f}" for a in arms)
    print(f"{i:<4}{meta[i]['episode_id']:>4}{meta[i]['start']:>6}  {row}"
          f"{l2(hold[i], gt1[i]):>12.4f}")

print(f"\n=== {tag}: how much the arm actually moves in this window ===")
print(f"{'win':<4}{'gt path len':>13}{'gt |r31-r0|':>13}{'pred path':>12}{'pred |r31-r0|':>15}")
for i in range(n):
    print(f"{i:<4}{np.abs(np.diff(gt1[i], axis=0)).sum():>13.4f}"
          f"{np.abs(gt1[i, -1] - gt1[i, 0]).sum():>13.4f}"
          f"{np.abs(np.diff(z['gt'][i], axis=0)).sum():>12.4f}"
          f"{np.abs(z['gt'][i, -1] - z['gt'][i, 0]).sum():>15.4f}")

print(f"\n=== {tag}: skill score vs the hold-still baseline (1 = perfect, 0 = no better) ===")
for a in arms:
    e, h = l2(z[a], gt1), l2(hold, gt1)
    print(f"  {a:<12} L2={e:.4f}   skill = 1 - L2/HOLD = {1 - e / h:+.3f}")

print(f"\n=== {tag}: right-arm dims (7..13) — these episodes never move the right arm ===")
print(f"  gt   |x| on dims 7..13 (excluding gripper col 13): "
      f"{np.abs(gt1[:, :, 7:13]).mean():.4f}")
print(f"  pred |x| on dims 7..13 (excluding gripper col 13): "
      f"{np.abs(z['gt'][:, :, 7:13]).mean():.4f}")
