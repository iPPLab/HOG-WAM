"""Score one checkpoint open-loop against real training windows, with ablations.

Usage:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python run_openloop.py --ckpt <root> --tag ours
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax.numpy as jnp
from fastwam_tpu import load_config
from fastwam_tpu.inference.load import build_policy

OUTPUT_DIR = Path(os.environ.get("FASTWAM_OPENLOOP_DIR", ROOT / "outputs" / "openloop"))

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--vae", required=True)
ap.add_argument("--stats", required=True)
ap.add_argument("--probe", default=str(OUTPUT_DIR / "probe_set.npz"))
ap.add_argument("--config", required=True)
args = ap.parse_args()

z = np.load(args.probe, allow_pickle=True)
canvas, qpos = z["canvas"], z["qpos"]
gt1, gt0, ctx = z["gt_shift1"], z["gt_shift0"], z["train_ctx"]
meta = json.loads(str(z["meta"]))
n = canvas.shape[0]
print(f"[{args.tag}] probe set: {n} windows, canvas {canvas.shape}, ctx {ctx.shape}", flush=True)

cfg = load_config(args.config)
policy = build_policy(ckpt_root=args.ckpt, vae_ckpt=args.vae, stats_path=args.stats,
                      config=cfg, dtype=jnp.bfloat16)
print(f"[{args.tag}] policy ready\n{policy.describe()}", flush=True)

mask = np.ones(ctx.shape[1], bool)
rng = np.random.default_rng(0)
noise_canvas = rng.integers(0, 256, canvas.shape[1:], dtype=np.uint8)

ARMS = {
    "gt":          lambda i: (canvas[i], qpos[i], ctx[i]),
    "zero_frame":  lambda i: (np.full_like(canvas[i], 128), qpos[i], ctx[i]),
    "noise_frame": lambda i: (noise_canvas, qpos[i], ctx[i]),
    "wrong_frame": lambda i: (canvas[(i + n // 2) % n], qpos[i], ctx[i]),
    "empty_text":  lambda i: (canvas[i], qpos[i], np.zeros_like(ctx[i])),
    "wrong_text":  lambda i: (canvas[i], qpos[i], ctx[(i + n // 2) % n]),
    "zero_qpos":   lambda i: (canvas[i], np.zeros_like(qpos[i]), ctx[i]),
}
if "baked_ctx" in z:
    baked = z["baked_ctx"]
    ARMS["baked_text"] = lambda i: (canvas[i], qpos[i], baked[i])
if "alt_ctx" in z:
    alt = z["alt_ctx"]
    ARMS["alt_table"] = lambda i: (canvas[i], qpos[i], alt[i])
# Half the text embedding's magnitude, and double it: if the model is merely mis-scaled
# relative to what it trained on, one of these moves in the right direction.
ARMS["text_half"] = lambda i: (canvas[i], qpos[i], ctx[i] * 0.5)
ARMS["text_double"] = lambda i: (canvas[i], qpos[i], ctx[i] * 2.0)

out = {}
for arm, make in ARMS.items():
    preds = []
    for i in range(n):
        c, q, t = make(i)
        policy.reset()
        preds.append(policy.infer_action(np.ascontiguousarray(c), q, t, mask))
    out[arm] = np.stack(preds)
    print(f"[{args.tag}] {arm} done", flush=True)

pred = out["gt"]


def l2(a, b):
    return float(np.sqrt(((a - b) ** 2).mean()))


print(f"\n================ {args.tag} ================")
print(f"{'arm':<12} {'L2 vs gt(shift1)':>17} {'L2 vs gt(shift0)':>17} {'vs gt-arm':>10}")
for arm, p in out.items():
    print(f"{arm:<12} {l2(p, gt1):>17.4f} {l2(p, gt0):>17.4f} {l2(p, pred):>10.4f}")

hold = np.repeat(qpos[:, None, :], gt1.shape[1], axis=1)
print(f"\nbaselines: hold-current-qpos vs gt1 = {l2(hold, gt1):.4f}   "
      f"vs gt0 = {l2(hold, gt0):.4f}")
print(f"           gt1 vs gt0 (one frame of motion) = {l2(gt1, gt0):.4f}")
mean_a = gt1.mean(axis=(0, 1))
print(f"           predict-dataset-mean vs gt1 = {l2(np.broadcast_to(mean_a, gt1.shape), gt1):.4f}")

print("\n--- chunk shape (gt arm) ---")
print(f"row0 - current qpos : {np.abs(pred[:, 0] - qpos).mean():.4f} "
      f"(gt shift1: {np.abs(gt1[:, 0] - qpos).mean():.4f}, "
      f"gt shift0: {np.abs(gt0[:, 0] - qpos).mean():.4f})")
print(f"total path length   : pred {np.abs(np.diff(pred, axis=1)).sum(1).mean():.4f}  "
      f"gt {np.abs(np.diff(gt1, axis=1)).sum(1).mean():.4f}")
print(f"row31 - row0        : pred {np.abs(pred[:, -1] - pred[:, 0]).mean():.4f}  "
      f"gt {np.abs(gt1[:, -1] - gt1[:, 0]).mean():.4f}")

for i in range(min(2, n)):
    print(f"\nwindow {i} (ep {meta[i]['episode_id']} start {meta[i]['start']}):")
    print(f"  qpos    {np.round(qpos[i], 3)}")
    print(f"  pred[0] {np.round(pred[i, 0], 3)}")
    print(f"  gt1[0]  {np.round(gt1[i, 0], 3)}")
    print(f"  pred[31]{np.round(pred[i, -1], 3)}")
    print(f"  gt1[31] {np.round(gt1[i, -1], 3)}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
target = OUTPUT_DIR / f"openloop_{args.tag}.npz"
np.savez_compressed(target, **out, qpos=qpos, gt1=gt1, gt0=gt0)
print(f"\nwrote {target}")
