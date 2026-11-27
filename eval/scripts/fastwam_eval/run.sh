#!/bin/bash
# Run a RoboTwin campaign with one or more local CUDA model servers.
# Usage: run.sh <run_tag> [workers=1] [gpu_ids=0]
# Required: FW_CKPT, FW_VAE, FW_STATS and FW_TEXT_CACHE.
set -u
TAG=${1:?run tag}
NW=${2:-1}
GPUS=${3:-0}

H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/_paths.sh"
ROOT=$FW_ROOT
TEXT_PORT=${FW_TEXT_PORT:-6135}
BASE_PORT=${FW_BASE_PORT:-6033}
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

: "${FW_CKPT:?set FW_CKPT}"; : "${FW_VAE:?set FW_VAE}"; : "${FW_STATS:?set FW_STATS}"
: "${FW_TEXT_CACHE:?set FW_TEXT_CACHE}"
mkdir -p "$ROOT/logs/$TAG" "$ROOT/results/$TAG" "$FW_TEXT_CACHE"
export FW_TASKS=${FW_TASKS:-$H/tasks/all.txt}
[ -f "$FW_TASKS" ] || { echo "missing task list $FW_TASKS"; exit 1; }

# Check for background textures required by the randomized scene configuration.
CFG=$RT_ROOT/task_config/$TW_TASK_CONFIG.yml
[ -f "$CFG" ] || { echo "no task_config $TW_TASK_CONFIG.yml in $RT_ROOT/task_config"; exit 1; }
if grep -qE '^\s*random_background:\s*true' "$CFG"; then
  [ -d "$RT_ROOT/assets/background_texture/unseen" ] || {
    echo "$TW_TASK_CONFIG sets random_background: true but assets/background_texture is missing."
    echo "Install it (11 GB):  bash $EV/setup.sh assets"
    exit 1; }
  log "task_config $TW_TASK_CONFIG (randomized; background textures present)"
else
  log "task_config $TW_TASK_CONFIG"
fi

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NSRV=${#GPU_LIST[@]}
if [ "$TEXT_PORT" -ge "$BASE_PORT" ] && [ "$TEXT_PORT" -lt "$((BASE_PORT + NSRV))" ]; then
  echo "FW_TEXT_PORT overlaps the model-server port range; choose a separate port." >&2
  exit 2
fi

# Reserve host memory for the text service and simulator workers.
AVAIL=$(awk '/^MemAvailable:/{print int($2/1048576)}' /proc/meminfo)
NEED=$((24 + NSRV * 3 + NW * 2))
[ "$AVAIL" -ge "$NEED" ] || {
  echo "host RAM too low: ${AVAIL}G available < ${NEED}G needed (UMT5 sidecar 22G fp32 + ${NSRV} servers + ${NW} sims)"
  echo "who is holding it: ps -eo rss,etimes,cmd -ww --no-headers --sort=-rss | head"
  exit 1; }
SWU=$(awk '/^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{if(t>0) print int(100*(t-f)/t); else print 0}' /proc/meminfo)
[ "$SWU" -lt 50 ] || { echo "swap is ${SWU}% used; the machine is thrashing. Clean up first."; exit 1; }

# Reserve GPU memory for new simulator workers and any model server started by this run.
for i in "${!GPU_LIST[@]}"; do
  g=${GPU_LIST[$i]}
  port=$((BASE_PORT + i))
  used=$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits)
  total=$(nvidia-smi --id="$g" --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$((total - used))
  # Estimate the peak memory of resident simulator workers in MiB.
  sims=$(( 4100 * ( (NW + NSRV - 1) / NSRV ) ))
  if ss -ltnH 2>/dev/null | grep -q ":$port " \
     && ps -eo args -ww --no-headers | grep -v 'bash -c' \
        | grep '[f]astwam_model_server\.py' | grep -q -- "--port $port "; then
    want=$sims
    note="server already resident, budgeting sims only"
  else
    want=$((15000 + sims))
    note="cold start, budgeting server + sims"
  fi
  [ "$free" -ge "$want" ] || {
    echo "gpu$g has ${free} MiB free, needs ~${want} MiB ($note)."
    echo "If that shortfall is someone else's campaign, wait for it to seal or use fewer workers."
    exit 1; }
  log "gpu$g: ${free} MiB free (want ${want}; $note)"
done

# --- our own leftovers -------------------------------------------------------------
n=$(ps -eo cmd -ww --no-headers | grep -v 'bash -c' | grep -cE '[f]astwam_eval/worker\.sh')
[ "$n" -eq 0 ] || { echo "unclean: $n FastWAM worker(s) still running; stop them first"; exit 1; }
rm -f "$ROOT"/stop-*

# --- text sidecar ------------------------------------------------------------------
if ss -ltnH 2>/dev/null | grep -q ":$TEXT_PORT "; then
  log "text sidecar already listening on $TEXT_PORT; reusing"
else
  log "starting UMT5 text sidecar on $TEXT_PORT (~22 GB RAM fp32, loads in ~8 min)"
  setsid nohup "$TEXT_PY" "$H/../fastwam_text_server.py" \
    --port "$TEXT_PORT" --cache "$FW_TEXT_CACHE" \
    > "$ROOT/logs/$TAG/text_${TEXT_PORT}.log" 2>&1 < /dev/null & disown -a
  for i in $(seq 1 80); do
    ss -ltnH 2>/dev/null | grep -q ":$TEXT_PORT " && { log "text sidecar READY"; break; }
    sleep 15
  done
  ss -ltnH 2>/dev/null | grep -q ":$TEXT_PORT " || {
    echo "text sidecar never came up; see $ROOT/logs/$TAG/text_${TEXT_PORT}.log"; exit 1; }
fi
export FW_TEXT_SERVER="127.0.0.1:$TEXT_PORT"

# --- model servers -----------------------------------------------------------------
for i in "${!GPU_LIST[@]}"; do
  port=$((BASE_PORT + i))
  gpu=${GPU_LIST[$i]}
  if ss -ltnH 2>/dev/null | grep -q ":$port "; then
    # Reusing a live server is only safe if it is OUR server with OUR checkpoint.
    # Without this, reusing a tag with a different ckpt silently evaluates the whole
    # campaign on the old weights: wrong data, no error, no way to tell afterwards.
    argv=$(ps -u "$(id -u)" -o args -ww --no-headers | grep -v 'bash -c' | grep '[f]astwam_model_server\.py' | grep -- "--port $port " | head -1)
    [ -n "$argv" ] || { echo "port $port is taken by something that is not our server"; exit 1; }
    for flag_value in "--ckpt $FW_CKPT" "--config $FW_TRAINING_CONFIG" "--vae $FW_VAE" "--stats $FW_STATS"; do
      grep -Fq -- "$flag_value " <<< "$argv " || {
        echo "live server on $port does not match $flag_value; stop it before launching." >&2
        exit 1
      }
    done
    log "server on $port already running with the right ckpt; reusing"
    continue
  fi
  log "starting server babysitter port $port -> gpu$gpu"
  setsid nohup bash "$H/server_babysitter.sh" "$port" "$gpu" "$TAG" \
    > "$ROOT/logs/$TAG/babysitter_${port}.log" 2>&1 < /dev/null & disown -a
done

log "waiting for model servers ..."
for i in "${!GPU_LIST[@]}"; do
  port=$((BASE_PORT + i))
  for _ in $(seq 1 180); do ss -ltnH 2>/dev/null | grep -q ":$port " && break; sleep 15; done
  ss -ltnH 2>/dev/null | grep -q ":$port " || {
    touch "$ROOT/stop-server-$port"
    echo "server on $port never became ready; see $ROOT/logs/$TAG/server_${port}.log"; exit 1; }
  log "server $port READY"
done

# --- workers -----------------------------------------------------------------------
for i in $(seq 1 "$NW"); do
  idx=$(( (i - 1) % NSRV ))
  gpu=${GPU_LIST[$idx]}
  port=$((BASE_PORT + idx))
  setsid nohup bash "$H/worker.sh" "$i" "$gpu" "$port" "$TAG" \
    > "$ROOT/logs/$TAG/worker$i.log" 2>&1 < /dev/null & disown -a
  log "worker $i -> gpu$gpu port$port"
  sleep 3
done
log "launched $NW workers. progress: bash $H/status.sh $TAG"
