#!/usr/bin/env python3
"""Pack the Fast-WAM RoboTwin LeRobot v2.1 release into aligned TWP3 episodes."""

import argparse
import json
from pathlib import Path

import numpy as np

from fastwam_tpu.data import canvas, twp

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ROLES = ("head", "left_wrist", "right_wrist")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_jpegs(path: Path):
    import av
    import simplejpeg

    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            pixels = frame.to_ndarray(format="rgb24")
            # Match the upstream per-camera training resize before composing the canvas.
            scaled = canvas.resize_bilinear_aa(pixels[None].astype(np.float32), 240, 320)[0]
            pixels = np.ascontiguousarray(np.rint(scaled).clip(0, 255).astype(np.uint8))
            yield simplejpeg.encode_jpeg(pixels, quality=95, colorspace="RGB", colorsubsampling="444")


def convert(source: Path, output: Path, task_map: dict[str, str]) -> dict:
    import pyarrow.parquet as pq

    info = json.loads((source / "meta/info.json").read_text())
    if info["codebase_version"] != "v2.1":
        raise ValueError(f"expected LeRobot v2.1, got {info['codebase_version']!r}")

    cameras = tuple(f"observation.images.{name}" for name in CAMERAS)
    tasks = {row["task_index"]: row["task"] for row in read_jsonl(source / "meta/tasks.jsonl")}
    episodes = sorted(read_jsonl(source / "meta/episodes.jsonl"), key=lambda row: row["episode_index"])
    ids = [row["episode_index"] for row in episodes]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("source episode indices must be non-empty and unique")
    if task_map and set(task_map) != {str(eid) for eid in ids}:
        raise ValueError("task map must identify every source episode exactly once")

    output.mkdir(parents=True, exist_ok=False)
    records = []
    for episode in episodes:
        eid = episode["episode_index"]
        count = episode["length"]
        args = {"episode_index": eid, "episode_chunk": eid // info["chunks_size"]}
        table = pq.read_table(
            source / info["data_path"].format(**args),
            columns=["episode_index", "frame_index", "action", "observation.state", "task_index"],
        )
        if (
            count < 1
            or table["frame_index"].to_pylist() != list(range(count))
            or set(table["episode_index"].to_pylist()) != {eid}
        ):
            raise ValueError(f"episode {eid}: inconsistent frame indices or count")

        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        if actions.shape != (count, 14) or states.shape != actions.shape:
            raise ValueError(f"episode {eid}: expected 14 aligned action and state values per frame")
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError(f"episode {eid}: non-finite action or state")

        instruction_ids = set(table["task_index"].to_pylist())
        if len(instruction_ids) != 1:
            raise ValueError(f"episode {eid}: expected one training instruction throughout the episode")
        instruction = tasks[next(iter(instruction_ids))]
        task_key = task_map.get(str(eid), episode.get("task_key", instruction))
        if not isinstance(task_key, str) or not task_key.strip():
            raise ValueError(f"episode {eid}: invalid task key {task_key!r}")

        streams = [read_jpegs(source / info["video_path"].format(video_key=cam, **args)) for cam in cameras]
        try:
            blobs = [jpeg for frames in zip(*streams, strict=True) for jpeg in frames]
        finally:
            for stream in streams:
                stream.close()
        if len(blobs) != count * len(cameras):
            raise ValueError(f"episode {eid}: video and parquet frame counts differ")

        # A TWP episode exposes n_frames - 1 sample starts. LeRobot already drops
        # the terminal raw frame, so reserve an unsampled slot after its last row.
        blobs.extend(blobs[-len(cameras):])
        actions = np.concatenate((actions, actions[-1:]))
        states = np.concatenate((states, states[-1:]))
        payload = twp.build(
            eid, [len(jpeg) for jpeg in blobs], b"".join(blobs),
            actions, [instruction], [], n_cams=len(cameras), states=states,
        )
        path = output / f"episode_{eid:06d}.twp"
        path.write_bytes(payload)
        records.append({
            "episode_id": eid, "task_key": task_key, "n_frames": count + 1,
            "payload_end": twp.Episode(payload, expect_id=eid).payload_end(),
        })
        if len(records) % 100 == 0:
            print(f"packed {len(records)}/{len(episodes)} episodes", flush=True)

    spec = {
        "format": "TWP3", "version": twp.VERSION, "cameras": cameras, "camera_roles": ROLES,
        "color_order": "rgb", "action_shift": 0, "action_dim": 14, "state_dim": 14,
    }
    (output / "spec.json").write_text(json.dumps(spec, indent=2) + "\n")
    manifest = {
        "format": "TWP3", "n_episodes": len(records),
        "n_transitions": sum(record["n_frames"] - 1 for record in records),
        "episodes": records,
    }
    # The loader opens the manifest first. Publish it only after all episodes and the spec exist.
    temporary = output / "MANIFEST.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "MANIFEST.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="extracted LeRobot v2.1 dataset root")
    parser.add_argument("--output", type=Path, required=True, help="new directory for TWP3")
    parser.add_argument("--task-map", type=Path, help="JSON object mapping source episode IDs to task keys")
    args = parser.parse_args()
    task_map = {} if args.task_map is None else json.loads(args.task_map.read_text())
    if not isinstance(task_map, dict):
        parser.error("--task-map must contain a JSON object")
    manifest = convert(args.source, args.output, task_map)
    print(f"packed {manifest['n_episodes']:,} episodes and {manifest['n_transitions']:,} transitions")


if __name__ == "__main__":
    main()
