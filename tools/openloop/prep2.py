"""Cross-task probe set.

The first probe set drew all its windows from one task, which made the `wrong_text` /
`wrong_frame` arms compare a hammer episode against another hammer episode — too weak to
say anything. This one spans several tasks, so swapping the text or the frame really does
ask the model a different question.
"""
import argparse, io, json, os, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fastwam_tpu import load_config
from fastwam_tpu.data import twp, windows

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--manifest", required=True)
ap.add_argument("--output-dir", default=str(ROOT / "outputs" / "openloop"))
ap.add_argument("--config", required=True)
args = ap.parse_args()
cfg = load_config(args.config)

OUTPUT_DIR = Path(args.output_dir)
EP_DIR = OUTPUT_DIR / "eps"
PREFIX = cfg.data.l3.prefix
SIDETABLE = cfg.data.l3.text_dir
HEAD_WH, WRIST_WH = (320, 256), (160, 128)
TASKS = ["adjust_bottle", "beat_block_hammer", "lift_pot", "place_container_plate"]
PER_TASK = 1
STARTS = (0.15, 0.45, 0.75)
EP_DIR.mkdir(parents=True, exist_ok=True)


def gs(a):
    return subprocess.run(["gsutil"] + a, capture_output=True, check=True).stdout


def canvas_of(streams):
    def rs(img, wh):
        return np.asarray(Image.fromarray(np.asarray(img, np.uint8), "RGB").resize(
            wh, resample=Image.BILINEAR), np.uint8)
    bottom = np.concatenate([rs(streams["left_wrist"], WRIST_WH),
                             rs(streams["right_wrist"], WRIST_WH)], axis=1)
    return np.ascontiguousarray(
        np.concatenate([rs(streams["head"], HEAD_WH), bottom], axis=0), np.uint8)


spec = json.loads(gs(["cat", f"{PREFIX}/spec.json"]))
roles = spec.get("camera_roles") or spec.get("cameras")
col = {r: list(roles).index(r) for r in ("head", "left_wrist", "right_wrist")}

entries = json.load(open(args.manifest))["episodes"]
by_task = {}
for e in entries:
    by_task.setdefault(e["task_key"], []).append(int(e["episode_id"]))
print("tasks in the set:", len(by_task))
chosen = []
for t in TASKS:
    if t not in by_task:
        print(f"!! task {t!r} absent; have e.g. {sorted(by_task)[:8]}")
        continue
    chosen += [(t, e) for e in by_task[t][:PER_TASK]]
print("chosen:", chosen)

meta = json.loads(gs(["cat", f"{SIDETABLE}/meta.json"]))
rows = np.frombuffer(gs(["cat", f"{SIDETABLE}/rows.i32"]), np.int32)
texts = json.loads(gs(["cat", f"{SIDETABLE}/texts.json"]))
rb = int(meta["max_len"]) * int(meta["d_model"]) * 2

S = []
for task, ep_id in chosen:
    local = str(EP_DIR / f"episode_{ep_id:06d}.twp")
    if not os.path.exists(local):
        gs(["cp", f"{PREFIX}/episode_{ep_id:06d}.twp", local])
    ep = twp.Episode(open(local, "rb").read(), expect_id=ep_id)
    qpos = np.asarray(ep.actions, np.float32)
    r = int(rows[ep_id])
    ctx = np.asarray(np.frombuffer(
        gs(["cat", "-r", f"{r*rb}-{r*rb+rb-1}", f"{SIDETABLE}/emb.f16"]), np.float16
    ).reshape(int(meta["max_len"]), int(meta["d_model"])), np.float32)
    baked = np.asarray(ep.text_emb[0], np.float32)
    print(f"\n[{task}] ep {ep_id} T={ep.n_frames}")
    print(f"   sidetable: {texts[r]!r}")
    n_w = windows.num_windows(ep.n_frames)
    for f in STARTS:
        st = int(n_w * f)
        a1, pad = windows.action_indices(st, ep.n_frames, action_shift=1)
        a0, _ = windows.action_indices(st, ep.n_frames, action_shift=0)
        p, _ = windows.proprio_indices(st, ep.n_frames)
        streams = {ro: np.asarray(Image.open(io.BytesIO(bytes(ep.jpeg(st, c)))).convert("RGB"),
                                  np.uint8)[..., ::-1] for ro, c in col.items()}
        S.append(dict(task=task, episode_id=ep_id, start=st, canvas=canvas_of(streams),
                      qpos=qpos[st], gt1=qpos[a1], gt0=qpos[a0], pad=pad,
                      ctx=ctx, baked=baked, text=texts[r]))

np.savez_compressed(
    OUTPUT_DIR / "probe_cross.npz",
    canvas=np.stack([s["canvas"] for s in S]), qpos=np.stack([s["qpos"] for s in S]),
    gt_shift1=np.stack([s["gt1"] for s in S]), gt_shift0=np.stack([s["gt0"] for s in S]),
    action_is_pad=np.stack([s["pad"] for s in S]),
    train_ctx=np.stack([s["ctx"] for s in S]), baked_ctx=np.stack([s["baked"] for s in S]),
    episode_id=np.array([s["episode_id"] for s in S]),
    start=np.array([s["start"] for s in S]),
    meta=json.dumps([{k: s[k] for k in ("task", "episode_id", "start", "text")} for s in S]),
)
print(f"\nwrote {OUTPUT_DIR / 'probe_cross.npz'} with {len(S)} windows across "
      f"{len(set(s['task'] for s in S))} tasks")
