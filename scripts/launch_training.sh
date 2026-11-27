#!/usr/bin/env bash
# Public entry point for any Linux control host with gcloud, ssh and tmux/screen.
#
# ./scripts/launch_training.sh configs/training/my_experiment.yaml RUN_NAME
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPTS_DIR/.." && pwd)"

CONFIG=${1:?usage: launch_training.sh CONFIG_YAML RUN_NAME [--print-plan] [-- overrides...]}
RUN=${2:?usage: launch_training.sh CONFIG_YAML RUN_NAME [--print-plan] [-- overrides...]}
shift 2
PRINT_PLAN=0
OVERRIDES=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        --print-plan) PRINT_PLAN=1; shift ;;
        --) shift; OVERRIDES+=("$@"); break ;;
        *)
            echo "FATAL: unknown option $1; put dotted Python overrides after --" >&2
            exit 2
            ;;
    esac
done

export FASTWAM_TRAINING_CONFIG=$CONFIG FASTWAM_RUN_NAME=$RUN
# shellcheck source=_deployment_env.sh
source "$SCRIPTS_DIR/_deployment_env.sh"
export BUCKET=$GCS_BUCKET
export OPS="$SCRIPTS_DIR/tpu_ops.sh"
export REUSE_STAGED=${REUSE_STAGED:-1}

if [ "$PRINT_PLAN" = 1 ]; then
    cat <<EOF
control_host=$(hostname)
run_name=$RUN
training_config=$TRAINING_CONFIG
tpu_resource=$TPU
tpu_accelerator=$ACC
tpu_zone=$ZONE
tpu_runtime=$RT
tpu_workers=auto-discovered-after-READY
control_logs=$CONTROL_LOG_DIR
gcs_logs=$GCS_LOG_DIR
tpu_temporary_files=/tmp/fastwam
fixed_model_spec=src/fastwam_tpu/specs/fastwam.yaml
fixed_data_spec=src/fastwam_tpu/specs/robotwin.yaml
EOF
    printf 'override=%s\n' "${OVERRIDES[@]}"
    exit 0
fi

mkdir -p "$CONTROL_LOG_DIR" "$LAUNCH_LOG_DIR"

# Staging is part of the one public launch workflow, not a prerequisite users memorize.
"$SCRIPTS_DIR/stage_code.sh"

record_run_spec() {
    for _ in $(seq 1 240); do
        if [ -s "$REPO/.last_sync_hash" ]; then
            "$SCRIPTS_DIR/write_run_spec.sh" "$RUN" "${OVERRIDES[@]}"
            return
        fi
        sleep 30
    done
    echo "RUN_SPEC watcher timed out waiting for .last_sync_hash" >&2
    return 1
}

record_run_spec > "$CONTROL_LOG_DIR/run_spec_${RUN}.log" 2>&1 &
exec "$SCRIPTS_DIR/babysit.sh" "$RUN" "${OVERRIDES[@]}"
