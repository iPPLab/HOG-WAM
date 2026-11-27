"""Inspect (and optionally copy) the official RoboTwin `dataset_stats.json`.

The B line trains on the released checkpoint's normalisation constants rather than its
own (`m4_data_design.md` §Q3 / plan D3). Two reasons the file is copied rather than
recomputed: it is what the released checkpoint was trained with, and an honest
recomputation would *not* reproduce it — the official means are per-episode
equal-weight, the action stream is replicated to a 32-step window before the statistics
are taken, and `global_q01` is a min of per-episode percentiles. Any of those differences
shows up as a constant offset in the action space and never as an error.

The file already sits in-region next to the checkpoint download, so **the default is to
read it where it is** and simply prove it is the file the config expects:

    uv run python tools/fetch_official_stats.py verify
    uv run python tools/fetch_official_stats.py verify --path /tmp/dataset_stats.json

`--publish` copies it next to the run's data prefix with a provenance sidecar. That
writes to GCS, so it is opt-in and never happens as a side effect of looking at the file.

Credentials: `gs://` reads go through gcsfs, which needs APPLICATION-DEFAULT
credentials — a working `gsutil` is not the same thing. If it 401s, the message says to
`gsutil cp` the file down and pass `--path`.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastwam_tpu import load_config  # noqa: E402
from fastwam_tpu.data.normalize import load_official_stats  # noqa: E402

def read_blob(path: str) -> bytes:
    if not path.startswith("gs://"):
        return Path(path).read_bytes()
    try:
        import gcsfs

        return gcsfs.GCSFileSystem().cat_file(path.removeprefix("gs://"))
    except Exception as exc:  # noqa: BLE001 - the message is the point
        raise SystemExit(
            f"could not read {path}: {exc!r}\n"
            f"gcsfs needs application-default credentials (a working gsutil is not enough).\n"
            f"  gsutil cp {path} /tmp/dataset_stats.json\n"
            f"  uv run python tools/fetch_official_stats.py verify --path /tmp/dataset_stats.json"
        ) from exc


def describe(blob: bytes, source: str, expect_sha256: str | None) -> None:
    digest = hashlib.sha256(blob).hexdigest()
    print(f"path        {source}")
    print(f"bytes       {len(blob):,}")
    print(f"sha256      {digest}")
    if expect_sha256:
        print(f"config      {expect_sha256}  {'MATCH' if digest == expect_sha256 else 'MISMATCH'}")
    stats = load_official_stats(blob, source=source)
    print(f"episodes    {stats.num_episodes:,}")
    print(f"transitions {stats.num_transition:,}")
    for name, norm in (("action", stats.action), ("proprio(state)", stats.proprio)):
        print(
            f"{name:<14} dim={norm.dim} "
            f"mean[0:3]={np.array2string(norm.mean[:3], precision=6)} "
            f"std[0:3]={np.array2string(norm.std[:3], precision=6)}"
        )
    same_range = np.array_equal(
        np.asarray(json.loads(blob)["action"]["default"]["global_min"]),
        np.asarray(json.loads(blob)["state"]["default"]["global_min"]),
    )
    print(f"action/state share global_min: {same_range}  (they are the same physical quantity)")
    if digest != (expect_sha256 or digest):
        raise SystemExit("sha256 does not match the configured value; refusing to call this verified")


def _cmd_verify(args) -> None:
    cfg = load_config(args.config)
    path = args.path or cfg.data.l3.stats_path
    describe(read_blob(path), path, cfg.data.l3.stats_sha256)
    if args.publish:
        publish(read_blob(path), path, args.publish)


def publish(blob: bytes, source: str, dst: str) -> None:
    """Copy the file plus a provenance sidecar. Writes to GCS — opt in only."""
    import subprocess
    import tempfile

    digest = hashlib.sha256(blob).hexdigest()
    provenance = {
        "source": source,
        "sha256": digest,
        "bytes": len(blob),
        "consumed_fields": ["action.default.global_mean", "action.default.global_std",
                            "state.default.global_mean", "state.default.global_std"],
        "note": (
            "Copied verbatim, never recomputed. The official means are per-episode "
            "equal-weight and the action stream is replicated to a 32-step window before "
            "the statistics are taken, so a recomputation would legitimately disagree."
        ),
    }
    with tempfile.TemporaryDirectory() as tmp:
        stats_path = Path(tmp) / "dataset_stats.json"
        prov_path = Path(tmp) / "dataset_stats.provenance.json"
        stats_path.write_bytes(blob)
        prov_path.write_text(json.dumps(provenance, indent=2))
        cmd = ["gsutil", "cp", str(stats_path), str(prov_path), dst.rstrip("/") + "/"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="read the stats file and check it against the config")
    v.add_argument("--config", required=True)
    v.add_argument("--path", default=None, help="override the configured stats_path (local or gs://)")
    v.add_argument("--publish", default=None, metavar="GS_DIR",
                   help="also copy it (plus provenance) to this prefix — writes to GCS")
    v.set_defaults(func=_cmd_verify)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
