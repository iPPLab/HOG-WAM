#!/bin/bash
# Evaluate one task/seed per simulator invocation and write a per-unit result.
# Usage: worker.sh <worker_id> <gpu_id> <rpc_port> <run_tag>
set -u
WID=${1:?worker id}
GPU=${2:?gpu index}
PORT=${3:?server port}
TAG=${4:?run tag}

H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/_paths.sh"
ROOT=$FW_ROOT
RES=$ROOT/results/$TAG
LOGD=$ROOT/logs/$TAG
TASKS=${FW_TASKS:-$H/tasks/all.txt}
STOP=$ROOT/stop-$WID
SEEDS=${TW_SEEDS:-100}
MAXTRY=${TW_MAXTRY:-3}
SRVWAIT=${TW_SRVWAIT:-180}
# Which driver to run. Defaults to this repo's copy (eval/rt_client/script/), NOT the
# checkout's: only ours carries `_harvest_episode_info`, without which --no_expert_check
# dies on `UnboundLocalError: episode_info`, and only ours honours TW_RPC_TIMEOUT and the
# render skip (TW_SKIP_OBS_IN_REPLAN=1). It is launched with cwd set to the RoboTwin
# root, so envs/, policy/ and description/utils/ still resolve there.
CLIENT_PY=$TW_CLIENT_PY
mkdir -p "$RES" "$LOGD"

log(){ echo "[$(date '+%m-%d %H:%M:%S')][w$WID] $*"; }

claim() {  # $1=task $2=seed -> 0 if claimed
  local d="$RES/$1/$2.lock"
  mkdir -p "$RES/$1"
  if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; return 0; fi
  local p; p=$(cat "$d/pid" 2>/dev/null)
  # A lock is alive iff its pid is alive. mtime-based reclaim would reclaim a LIVE slow
  # unit (episode duration is heavy-tailed) and put a second sim on the card.
  if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then
    if mv "$d" "$d.stale.$$" 2>/dev/null; then
      rm -rf "$d.stale.$$"
      if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; log "reclaimed stale $1/$2 (dead pid $p)"; return 0; fi
    fi
  fi
  return 1
}

release(){ rm -rf "$RES/$1/$2.lock"; }

# Probe the SOCKET, not the log: server logs are appended across incarnations, so a stale
# "listening" line would wave us through against a dead port.
wait_server() {
  local w=0
  while ! ss -ltnH 2>/dev/null | grep -q ":$PORT "; do
    [ -f "$STOP" ] && return 1
    w=$((w+1))
    [ "$w" -eq 1 ] && log "server port $PORT down; waiting"
    [ "$w" -ge "$SRVWAIT" ] && { log "server still down after $((SRVWAIT*10))s -> release unit"; return 1; }
    sleep 10
  done
  [ "$w" -gt 0 ] && log "server back after $((w*10))s"
  return 0
}

write_result() {  # $1=task $2=seed $3=success $4=rate_line
  local f="$RES/$1/$2.json" t="$RES/$1/.$2.tmp.$$"
  local as; as=$(grep -a 'current seed' "$LOGD/${TAG}_${1}_s${2}.log" 2>/dev/null \
                 | sed -E 's/\x1b\[[0-9;]*m//g' | tail -1 | grep -oE '[0-9]+$')
  printf '{"task":"%s","seed":%s,"actual_seed":%s,"success":%s,"rate_line":"%s","worker":%s,"ts":"%s"}\n' \
    "$1" "$2" "${as:-null}" "$3" "$4" "$WID" "$(date -Is)" > "$t"
  mv -f "$t" "$f"
}

seed_of() {
  [ -z "${TW_SEEDLIST:-}" ] && return 0
  python3 - "$TW_SEEDLIST" "$1" "$2" <<'PY2'
import json
import sys

with open(sys.argv[1]) as handle:
    seeds = json.load(handle)["seeds"]
try:
    print(seeds[sys.argv[2]][int(sys.argv[3])])
except (KeyError, IndexError):
    pass
PY2
}

run_unit() {  # $1=task $2=seed -> echoes ok|fail|infra
  local task=$1 seed=$2
  local lg="$LOGD/${TAG}_${task}_s${seed}.log"
  local abs; abs=$(seed_of "$task" "$seed")
  if [ -n "${TW_SEEDLIST:-}" ] && [ -z "$abs" ]; then
    log "seedlist has no $task[$seed] -> refuse (never silently fall back to sliding seeds)"
    echo infra; return
  fi
  local extra=()
  # TW_EXPERT_TASKS: comma-list of tasks that must run the REAL expert pre-roll.
  # `_harvest_episode_info`'s plan_success=False short-circuit crashes inside these
  # envs' play_once() (put_bottles_dustbin: IndexError on right_action[1][0];
  # open_microwave: target_pose None assert) -- the fast path was only ever
  # validated on the t10 tasks. Canonical seeds are expert-feasible by
  # construction, so the expert pass reproduces the same scene, it just costs
  # the pre-roll (~90s/unit).
  if [ -n "$abs" ]; then
    case ",${TW_EXPERT_TASKS:-}," in
      *,"$task",*) extra=(--abs_seed "$abs") ;;
      *)           extra=(--abs_seed "$abs" --no_expert_check True) ;;
    esac
  fi
  ( cd "$RT_ROOT" && \
    timeout -k 60 "${TW_UNIT_TIMEOUT:-7200}" env CUDA_VISIBLE_DEVICES=$GPU \
    TASK_CONFIG="$TW_TASK_CONFIG" PYTHONUNBUFFERED=1 \
    HOGWAM_TRIAL_SEED="$seed" \
    TW_SKIP_OBS_IN_REPLAN="${TW_SKIP_OBS_IN_REPLAN:-0}" \
    "$SIM_PY" "$CLIENT_PY" --port "$PORT" \
      --config policy/fastwam_policy/deploy_policy.yml \
      --overrides --task_name "$task" --test_num 1 --task_config "$TW_TASK_CONFIG" --seed "$seed" \
      --ckpt_setting "${TAG}_s${seed}" "${extra[@]}" \
      < /dev/null > "$lg" 2>&1 )
  local rc=$?
  local line; line=$(grep -a 'Success rate' "$lg" 2>/dev/null | tail -1 | sed -E 's/\x1b\[[0-9;]*m//g' | grep -oE '[0-9]+/[0-9]+ => [0-9.]+%')
  # A non-zero exit or a missing rate line is infra, never success:false. Recording an
  # outage as a failed episode poisons the statistics, which is worse than losing the unit.
  if [ $rc -ne 0 ] || [ -z "$line" ]; then echo infra; return; fi
  case "$line" in 1/1*) echo ok ;; 0/1*) echo fail ;; *) echo infra ;; esac
}

log "start (gpu=$GPU port=$PORT tag=$TAG seeds=0..$((SEEDS-1)) root=$ROOT)"
idle=0
while [ ! -f "$STOP" ]; do
  got=0
  while IFS= read -r task; do
    [ -z "$task" ] && continue
    case "$task" in \#*) continue ;; esac
    for s in $(seq 0 $((SEEDS-1))); do
      [ -f "$STOP" ] && break 2
      [ -f "$RES/$task/$s.json" ] && continue
      [ -f "$RES/$task/$s.giveup" ] && continue
      claim "$task" "$s" || continue
      got=1; idle=0
      log "claim $task/$s"
      tries=0; st=infra
      while [ $tries -lt $MAXTRY ]; do
        wait_server || { st=down; break; }
        st=$(run_unit "$task" "$s")
        # An empty status means run_unit's own subshell died; without this it would fall
        # through and be WRITTEN as a real success:false. Never let unknown become data.
        [ -z "$st" ] && st=infra
        [ "$st" != infra ] && break
        tries=$((tries+1)); log "infra fail $task/$s (try $tries/$MAXTRY)"; sleep 20
      done
      if [ "$st" = down ]; then
        log "release $task/$s (server unavailable)"
        release "$task" "$s"
      elif [ "$st" = infra ]; then
        log "give up $task/$s after $MAXTRY infra failures -> mark .giveup"
        : > "$RES/$task/$s.giveup"
        release "$task" "$s"
      else
        write_result "$task" "$s" "$([ "$st" = ok ] && echo true || echo false)" \
                     "$(grep -a 'Success rate' "$LOGD/${TAG}_${task}_s${s}.log" | tail -1 | sed -E 's/\x1b\[[0-9;]*m//g' | grep -oE '[0-9]+/[0-9]+ => [0-9.]+%')"
        release "$task" "$s"
        log "done $task/$s -> $st"
      fi
      break 2
    done
  done < "$TASKS"
  if [ $got -eq 0 ]; then
    idle=$((idle+1))
    [ $idle -ge 3 ] && { log "no work left; exiting"; break; }
    sleep 30
  fi
done
log "exit"
