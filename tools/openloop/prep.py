"""Build a fixed open-loop probe set from the TRAINING data.

For a handful of real training windows we save exactly what the deployed policy is
handed at a replan (uint8 canvas + raw qpos + text embedding) plus the GROUND TRUTH
future qpos the loader would have used as the target. Then any checkpoint can be scored
against it offline, on this box, with no sim.
"""
import argparse, glob, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fastwam_tpu import load_config
from fastwam_tpu.data import twp, windows

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("task", nargs="?", default="adjust_bottle")
ap.add_argument("--episodes", type=int, default=3)
ap.add_argument("--manifest", required=True)
ap.add_argument("--text-cache", required=True)
ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "openloop"))
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = load_config(args.config)

OUTPUT_DIR = Path(args.output_dir)
EP_DIR = OUTPUT_DIR / "eps"
PREFIX = cfg.data.l3.prefix
SIDETABLE = cfg.data.l3.text_dir
HEAD_WH, WRIST_WH = (320, 256), (160, 128)
EP_DIR.mkdir(parents=True, exist_ok=True)


def gs(args, **kw):
    return subprocess.run(["gsutil"] + args, capture_output=True, check=True, **kw).stdout


def eval_style_canvas(head, left, right):
    """The eval client's pixel path (policies/fastwam_policy/__init__.py), uint8 PIL."""
    def rs(img, wh):
        return np.asarray(Image.fromarray(np.asarray(img, np.uint8), "RGB").resize(
            wh, resample=Image.BILINEAR), dtype=np.uint8)
    top = rs(head, HEAD_WH)
    bottom = np.concatenate([rs(left, WRIST_WH), rs(right, WRIST_WH)], axis=1)
    return np.ascontiguousarray(np.concatenate([top, bottom], axis=0), np.uint8)


# ---------------------------------------------------------------- pick episodes
spec = json.loads(gs(["cat", f"{PREFIX}/spec.json"]))
roles = spec.get("camera_roles") or spec.get("cameras")
print("spec camera_roles:", roles)
col = {r: list(roles).index(r) for r in ("head", "left_wrist", "right_wrist")}
print("column map:", col)

manifest = json.load(open(args.manifest))
print("manifest type:", type(manifest).__name__,
      len(manifest) if hasattr(manifest, "__len__") else "")
if isinstance(manifest, dict):
    print("manifest top keys:", list(manifest)[:8])

WANT = args.task
N_EPS = args.episodes

# The manifest maps episodes to their source task; fall back to scanning if the shape
# is not what we expect (printed above, so a wrong guess is visible not silent).
candidates = []
entries = manifest["episodes"] if isinstance(manifest, dict) and "episodes" in manifest else manifest
if isinstance(entries, list) and entries and isinstance(entries[0], dict):
    print("manifest entry sample:", json.dumps(entries[0])[:300])
    key = next((k for k in ("task", "task_name", "source_task", "name") if k in entries[0]), None)
    print("task key:", key)
    if key:
        for e in entries:
            if str(e[key]) == WANT:
                candidates.append(int(e.get("episode_id", e.get("id", len(candidates)))))
print(f"episodes for task {WANT!r}: {len(candidates)} (first: {candidates[:6]})")
chosen = candidates[:N_EPS]
if not chosen:
    print("!! no task match; falling back to the first episodes in the set")
    chosen = list(range(N_EPS))

# ---------------------------------------------------------------- text side table
meta = json.loads(gs(["cat", f"{SIDETABLE}/meta.json"]))
print("sidetable:", {k: meta[k] for k in ("max_len", "d_model", "n_rows", "encoder", "granularity")})
rows = np.frombuffer(gs(["cat", f"{SIDETABLE}/rows.i32"]), np.int32)
texts = json.loads(gs(["cat", f"{SIDETABLE}/texts.json"]))
row_bytes = int(meta["max_len"]) * int(meta["d_model"]) * 2

# ---------------------------------------------------------------- eval text cache
cache = {}
for path in glob.glob(str(Path(args.text_cache) / "*.npz")):
    z = np.load(path, allow_pickle=True)
    m = json.loads(str(z["meta"])) if "meta" in z else {}
    if "prompt" in m:
        arr = z["emb"] if "emb" in z else z[list(z)[0]]
        cache[m["prompt"]] = (path, np.asarray(arr, np.float32), m)
print(f"eval text cache: {len(cache)} prompts, e.g. {list(cache)[:2]}")

samples = []
for ep_id in chosen:
    name = f"episode_{ep_id:06d}.twp"
    local = str(EP_DIR / name)
    if not os.path.exists(local):
        gs(["cp", f"{PREFIX}/{name}", local])
    buf = open(local, "rb").read()
    ep = twp.Episode(buf, expect_id=ep_id)
    qpos = np.asarray(ep.actions, np.float32)
    seen, unseen = ep.seen, ep.unseen
    print(f"\nep {ep_id}: T={ep.n_frames} cams={ep.n_cams} dim={ep.action_dim} "
          f"seen[{len(seen)}] unseen[{len(unseen)}]")
    print(f"   seen[0]  : {seen[0]!r}")
    print(f"   unseen[0]: {unseen[0]!r}")

    row = int(rows[ep_id])
    blob = gs(["cat", "-r", f"{row*row_bytes}-{row*row_bytes+row_bytes-1}", f"{SIDETABLE}/emb.f16"])
    train_ctx = np.frombuffer(blob, np.float16).reshape(int(meta["max_len"]), int(meta["d_model"]))
    train_ctx = np.asarray(train_ctx, np.float32)
    print(f"   sidetable row {row}: {texts[row]!r}")
    print(f"   train ctx: |x|max={np.abs(train_ctx).max():.3f} nonzero rows="
          f"{int((np.abs(train_ctx).sum(1) > 0).sum())}")

    n_w = windows.num_windows(ep.n_frames)
    for start in [int(n_w * f) for f in (0.1, 0.35, 0.6, 0.85)]:
        a_idx, a_pad = windows.action_indices(start, ep.n_frames, action_shift=1)
        a_idx0, _ = windows.action_indices(start, ep.n_frames, action_shift=0)
        p_idx, _ = windows.proprio_indices(start, ep.n_frames)
        streams = {r: np.asarray(Image.open(__import__("io").BytesIO(
            bytes(ep.jpeg(start, c)))).convert("RGB"), np.uint8) for r, c in col.items()}
        # L3 JPEGs are cv2-baked BGR; the loader reverses them (data/canvas.py).
        streams = {r: v[..., ::-1] for r, v in streams.items()}
        canvas = eval_style_canvas(streams["head"], streams["left_wrist"], streams["right_wrist"])
        samples.append(dict(
            episode_id=ep_id, start=start,
            canvas=canvas, qpos=qpos[start],
            gt_shift1=qpos[a_idx], gt_shift0=qpos[a_idx0], action_is_pad=a_pad,
            proprio_rows=qpos[p_idx], train_ctx=train_ctx,
            seen0=seen[0], unseen0=unseen[0], sidetable_text=texts[row],
        ))

np.savez_compressed(
    OUTPUT_DIR / "probe_set.npz",
    canvas=np.stack([s["canvas"] for s in samples]),
    qpos=np.stack([s["qpos"] for s in samples]),
    gt_shift1=np.stack([s["gt_shift1"] for s in samples]),
    gt_shift0=np.stack([s["gt_shift0"] for s in samples]),
    action_is_pad=np.stack([s["action_is_pad"] for s in samples]),
    train_ctx=np.stack([s["train_ctx"] for s in samples]),
    episode_id=np.array([s["episode_id"] for s in samples]),
    start=np.array([s["start"] for s in samples]),
    meta=json.dumps([{k: s[k] for k in ("episode_id", "start", "seen0", "unseen0",
                                        "sidetable_text")} for s in samples]),
)
print(f"\nwrote {OUTPUT_DIR / 'probe_set.npz'} with {len(samples)} windows")
