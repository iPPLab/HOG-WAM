#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
condition=${1:?usage: run_robotwin.sh CONDITION TAG [WORKERS] [GPU_IDS]}
tag=${2:?set a unique run tag}
case "$condition" in
    randomized) export TW_TASK_CONFIG=demo_randomized ;;
    clean|mixed|warm|underexposure|contrast) export TW_TASK_CONFIG=demo_clean ;;
    *) echo "Unknown RoboTwin condition: $condition" >&2; exit 2 ;;
esac
[[ "$tag" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "Invalid tag" >&2; exit 2; }
export HOGWAM_ROBOTWIN_CONDITION=$condition
: "${FW_TRAINING_CONFIG:?Set the selected training recipe}"
: "${FW_CKPT:?Set the step-29365 BF16 export}"
: "${FW_VAE:?Set the Wan2.2 VAE Orbax directory}"
: "${FW_STATS:?Set the RoboTwin dataset_stats.json path}"
: "${FW_TEXT_CACHE:?Set a writable inference text-cache directory}"
# The paper uses the native expert-solvability check, not a fixed seed list.
unset TW_SEEDLIST FW_SERVER_EXTRA
export TW_SEEDS=100 FW_TASKS="$repo/eval/scripts/fastwam_eval/tasks/all.txt"
export FW_ROOT=${FW_ROOT:-$repo/outputs/eval/$tag}
python3 - "$FW_ROOT/campaign.json" "$condition" "$tag" "$FW_CKPT" \
    "$FW_TRAINING_CONFIG" "$FW_VAE" "$FW_STATS" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

manifest, condition, tag, ckpt, config, vae, stats = sys.argv[1:]
path, checkpoint = Path(manifest), Path(ckpt).resolve()
marker = checkpoint / "commit_success.txt"
if not marker.is_file() or json.loads(marker.read_text()).get("step") != 29365:
    raise SystemExit(f"Expected a committed step-29365 BF16 export: {checkpoint}")
if not Path(vae).is_dir():
    raise SystemExit(f"VAE directory not found: {vae}")
spec = {
    "condition": condition, "tag": tag, "checkpoint": str(checkpoint),
    "recipe_sha256": hashlib.sha256(Path(config).read_bytes()).hexdigest(),
    "vae": str(Path(vae).resolve()), "stats_sha256": hashlib.sha256(Path(stats).read_bytes()).hexdigest(),
    "noise_seed_base": 20260927, "episodes_per_task": 100,
}
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists() and json.loads(path.read_text()) != spec:
    raise SystemExit(f"campaign manifest mismatch: {path}")
if not path.exists():
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".campaign.")
    with os.fdopen(fd, "w") as f:
        json.dump(spec, f, indent=2)
    os.replace(name, path)
PY
exec bash "$repo/eval/scripts/fastwam_eval/run.sh" "$tag" "${3:-1}" "${4:-0}"
