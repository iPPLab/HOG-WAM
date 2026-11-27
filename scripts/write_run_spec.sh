#!/usr/bin/env bash
# Record what a run was actually launched with, next to the run's own checkpoints.
#
#   ./scripts/write_run_spec.sh <run_name> [override ...]
#
# Written because the 29,365-step `v6e-official-30k` run has no record of its first 5,200
# steps: the log was truncated on every relaunch, and nothing outside the log said which
# code package or which overrides produced those checkpoints. When that run turned out to
# have trained in a doubled latent space, answering "which bytes did this?" took an
# archaeology session against GCS timestamps.
#
# Two files, because they answer two different questions:
#
#   RUN_SPEC.json   what this run IS. Written once, on the first top-level launch, and never
#                   overwritten -- a spec that changes under you is not a spec. A relaunch
#                   that would change it is a fact worth seeing, so it is reported rather
#                   than silently applied.
#   launches.jsonl  one record per top-level launch_training.sh invocation. Internal Spot
#                   relaunches stay in supervisor and worker logs because they do not change
#                   the run specification.
#
# Read-modify-write on launches.jsonl assumes one control-host launcher owns a run at a time.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=_deployment_env.sh
source "$REPO/scripts/_deployment_env.sh"

RUN=${1:?usage: write_run_spec.sh <run_name> [overrides...]}
shift
OVERRIDES="$*"

BUCKET=${BUCKET:-$GCS_BUCKET}
CODE_PREFIX=${CODE_PREFIX:?launch_training.sh must resolve CODE_PREFIX}
RUN_PREFIX=${RUN_PREFIX:?launch_training.sh must resolve RUN_PREFIX}
TPU=${TPU:?launch_training.sh must export TPU}
ZONE=${ZONE:?launch_training.sh must export ZONE}
ACC=${ACC:?launch_training.sh must export ACC}
HASH=$(cat "$REPO/.last_sync_hash" 2>/dev/null || echo unknown)
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DEST="gs://$BUCKET/$RUN_PREFIX/$RUN"

# The exact argv the training process receives on every host, so the spec can be replayed
# by hand without reconstructing it from three scripts.
CMD=".venv/bin/python -m fastwam_tpu.training.train $TRAINING_CONFIG tpu.run_name=$RUN tpu.resume=true $OVERRIDES"

spec_json() {
    python3 - "$RUN" "$HASH" "$NOW" "$CMD" "$OVERRIDES" "$TPU" "$ZONE" "$ACC" "$BUCKET" "$CODE_PREFIX" "$TRAINING_CONFIG" <<'PY'
import json, sys
run, pkg, now, cmd, ov, tpu, zone, acc, bucket, code_prefix, training = sys.argv[1:12]
print(json.dumps({
    "run_name": run,
    "training_config": training,
    "code_package_md5": pkg,
    "code_package_uri": f"gs://{bucket}/{code_prefix}/fastwam_tpu_{pkg}.tgz",
    "first_launch_utc": now,
    "command": cmd,
    "overrides": ov.split() if ov else [],
    "slice": {"tpu": tpu, "zone": zone, "accelerator": acc, "spot": True},
}, indent=2))
PY
}

if gcloud storage ls "$DEST/RUN_SPEC.json" >/dev/null 2>&1; then
    echo "RUN_SPEC.json already exists for $RUN -- not overwriting."
    existing_hash=$(gcloud storage cat "$DEST/RUN_SPEC.json" 2>/dev/null \
        | python3 -c "import sys,json;print(json.load(sys.stdin).get('code_package_md5',''))" 2>/dev/null)
    if [ -n "$existing_hash" ] && [ "$existing_hash" != "$HASH" ]; then
        echo "!! WARNING: this run was specified with package $existing_hash, but the"
        echo "!! working tree is now $HASH. The checkpoints under $RUN were NOT all"
        echo "!! produced by the same code. Decide deliberately before continuing."
    fi
else
    spec_json | gcloud storage cp - "$DEST/RUN_SPEC.json" || exit 1
    echo "wrote $DEST/RUN_SPEC.json (package $HASH)"
fi

# Append this top-level launch invocation. Spot relaunches within the same babysitter
# session are operational events recorded by the supervisor and worker logs; they do not
# change the run specification.
# `gcloud storage cat` on a missing object is an error, not empty output, so the miss is
# swallowed deliberately: the first launch has no history to read.
LINE=$(python3 -c "
import json,sys
print(json.dumps({'utc': sys.argv[1], 'code_package_md5': sys.argv[2], 'command': sys.argv[3]}))
" "$NOW" "$HASH" "$CMD")
PREV=$(gcloud storage cat "$DEST/launches.jsonl" 2>/dev/null)
printf '%s\n' "${PREV:+$PREV}" "$LINE" | grep -v '^$' \
    | gcloud storage cp - "$DEST/launches.jsonl" \
    && echo "appended launch record to $DEST/launches.jsonl"
