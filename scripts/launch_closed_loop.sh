#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--" ]]; then
    echo "Usage: launch_closed_loop.sh -- evaluator command [args...]" >&2
    exit 2
fi
CHECKPOINT=${FW_CHECKPOINT:?FW_CHECKPOINT is required}
RESULTS=${FW_RESULTS_DIR:?FW_RESULTS_DIR is required}
OIDN=${FW_OIDN_VERSION:-2.3.3}
CAMERAS=${FW_CAMERA_COUNT:-3}
EPISODES=${FW_EPISODES_PER_TASK:-25}
[[ "$OIDN" == "2.3.3" ]] || { echo "FATAL: expected OIDN 2.3.3, got $OIDN" >&2; exit 2; }
[[ "$CAMERAS" == "3" ]] || { echo "FATAL: expected 3 cameras, got $CAMERAS" >&2; exit 2; }
[[ -d "$CHECKPOINT" || -f "$CHECKPOINT" ]] || { echo "FATAL: checkpoint does not exist: $CHECKPOINT" >&2; exit 2; }
[[ -z "${FW_TASKS_FILE:-}" || -f "$FW_TASKS_FILE" ]] || { echo "FATAL: task file does not exist: $FW_TASKS_FILE" >&2; exit 2; }
if [[ -e "$RESULTS" ]] && [[ -n "$(find "$RESULTS" -mindepth 1 -maxdepth 1 -print -quit)" ]] && [[ "${FW_ALLOW_DIRTY:-0}" != 1 ]]; then
    echo "FATAL: result directory is non-empty: $RESULTS (set FW_ALLOW_DIRTY=1 deliberately)" >&2
    exit 2
fi
mkdir -p "$RESULTS"
shift
[[ "$#" -gt 0 ]] || { echo "FATAL: evaluator command is missing" >&2; exit 2; }
printf '%s\n' "closed_loop_plan" "checkpoint=$CHECKPOINT" "results=$RESULTS" "oidn_version=$OIDN" "camera_count=$CAMERAS" "episodes_per_task=$EPISODES" "task_file=${FW_TASKS_FILE:-unset}" "command=$*"
exec "$@"
