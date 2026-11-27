#!/bin/bash
# End-to-end check of the canonical-seed path before it is let near the campaign.
#
# Runs ONE unit twice on the same task, through the patched client, on the same server:
#   A. as the campaign runs today   -- sliding seed, expert pre-roll on
#   B. canonical seed               -- --abs_seed <s> --no_expert_check True
#
# What has to be true for B to be usable:
#   * rc = 0 and a Success rate line (i.e. `_harvest_episode_info` really did stand in
#     for the expert -- the shared client raises NameError here)
#   * the instruction is a fully substituted sentence, no `{A}` left. The patched client
#     raises on a leftover brace, so a clean run already proves it, but it is printed
#     for eyeballing because this is the failure that would otherwise be silent.
#   * B is meaningfully faster than A -- that is the whole point.
#
# B's `[canonical] st_seed=<n> expert_check=False` line is the marker that the path was
# actually taken; its absence means the flags did not reach the client.
set -u
H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/_paths.sh"
SEEDLIST=${TW_SEEDLIST:?set TW_SEEDLIST to a RoboTwin seed table}
TASK=${1:-lift_pot}
IDX=${2:-0}
PORT=${3:-6033}
OUT=$FW_ROOT/logs/fwopt
mkdir -p "$OUT"

ABS=$(python3 -c "import json;print(json.load(open('$SEEDLIST'))['seeds']['$TASK'][$IDX])")
echo "task=$TASK unit=$IDX canonical_seed=$ABS"
cd "$EV/third_party/RoboTwin" || exit 1

run() {  # $1=label  $2...=extra client args
  local label=$1; shift
  local lg="$OUT/seedlist_${TASK}_${label}.log" t0 t1 rc
  t0=$(date +%s)
  timeout -k 60 3600 env CUDA_VISIBLE_DEVICES=1 TASK_CONFIG="$TW_TASK_CONFIG" PYTHONUNBUFFERED=1 \
    TW_SKIP_OBS_IN_REPLAN=1 \
    "$SIM_PY" "$TW_CLIENT_PY" \
      --port "$PORT" --config policy/fastwam_policy/deploy_policy.yml \
      --overrides --task_name "$TASK" --test_num 1 --task_config "$TW_TASK_CONFIG" \
      --seed "$IDX" --ckpt_setting "fwseed_${label}" "$@" \
      < /dev/null > "$lg" 2>&1
  rc=$?; t1=$(date +%s)
  local strip="s/\x1b\[[0-9;]*m//g"
  echo "[$label] rc=$rc wall=$((t1-t0))s rate=$(grep -a 'Success rate' "$lg" | tail -1 | sed -E "$strip" | grep -oE '[0-9]+/[0-9]+ => [0-9.]+%')"
  echo "[$label] canonical_line=$(grep -a '\[canonical\]' "$lg" | tail -1 | sed -E "$strip")"
  echo "[$label] instruction=$(grep -a 'set_task' "$lg" | tail -1 | sed -E "$strip")"
  echo "[$label] $(grep -a 'fw_prof' "$lg" | tail -1)"
  grep -aq 'Traceback' "$lg" && echo "[$label] TRACEBACK: $(grep -a -A3 'Traceback' "$lg" | tail -2 | sed -E "$strip")"
}

# The baseline arm is the "how long does it take today" reference. Skip it with
# SKIP_BASELINE=1 when the campaign is stopped and every minute of the window is idle
# throughput -- the canonical arm alone still proves the path works, which is what the
# pre-rollout gate actually asks for.
[ "${SKIP_BASELINE:-0}" = 1 ] || run baseline
run canonical --abs_seed "$ABS" --no_expert_check True
