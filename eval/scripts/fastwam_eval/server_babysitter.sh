#!/bin/bash
# Keep exactly one FastWAM model server alive on a fixed port, independent of any task's
# lifetime. Workers connect and disconnect; the server outlives every unit.
#
# Probe the listening socket rather than an append-only log. Restoring weights
# and compiling the first inference can take longer than later requests.
#
# Usage: server_babysitter.sh <port> <gpu> <run_tag>
# Env:   FW_CKPT FW_VAE FW_STATS FW_TEXT_CACHE FW_TEXT_SERVER (see run.sh)
set -u
PORT=${1:?port}
GPU=${2:?gpu index}
TAG=${3:?run tag}

H=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$H/_paths.sh"
SCRIPTS=$(cd "$H/.." && pwd)
ROOT=$FW_ROOT
MF=${FW_MEM_FRACTION:-0.62}
LOGD=$ROOT/logs/$TAG
STOP=$ROOT/stop-server-$PORT
mkdir -p "$LOGD"
log(){ echo "[$(date '+%m-%d %H:%M:%S')][srv:$PORT] $*"; }

: "${FW_CKPT:?set FW_CKPT to the converted official Orbax root}"
: "${FW_VAE:?set FW_VAE to the Wan2.2 VAE Orbax dir}"
: "${FW_STATS:?set FW_STATS to the official dataset_stats.json}"
: "${FW_TEXT_CACHE:?set FW_TEXT_CACHE to the shared sha256(prompt) cache dir}"

while [ ! -f "$STOP" ]; do
  log "starting server (gpu$GPU, mem_fraction=$MF)"
  CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_MEM_FRACTION=$MF \
    "$PY" "$SCRIPTS/fastwam_model_server.py" \
      --port "$PORT" --ckpt "$FW_CKPT" --vae "$FW_VAE" --stats "$FW_STATS" \
      --config "$FW_TRAINING_CONFIG" \
      --text-cache "$FW_TEXT_CACHE" \
      ${FW_TEXT_SERVER:+--text-server "$FW_TEXT_SERVER"} \
      --max-clients "${FW_MAX_CLIENTS:-2}" ${FW_SERVER_EXTRA:-} \
      >> "$LOGD/server_${PORT}.log" 2>&1 &
  SP=$!
  # 160 x 15 s = 40 min. Loading 12 GB of weights is ~1 min; the rest is the XLA compile
  # of the warm-up call, which is paid once here rather than inside a real episode (the
  # client's socket timeout is 300 s).
  #
  # The budget is deliberately generous. Compiling 30 UNROLLED DiT layers plus a 10-step
  # action loop is a big graph and it was measured in the tens of minutes on this 3090,
  # so a tight budget does not merely mis-report — the loop falls through, the outer
  # `while` restarts a server that was seconds from listening, and the campaign sits in
  # a restart loop that never converges. Over-waiting costs nothing: the workers block on
  # the socket and only give a unit back after TW_SRVWAIT.
  for i in $(seq 1 160); do
    ss -ltnH 2>/dev/null | grep -q ":$PORT " && { log "READY (pid $SP, ${i}5s)"; break; }
    kill -0 "$SP" 2>/dev/null || break
    sleep 15
  done
  while kill -0 "$SP" 2>/dev/null; do
    [ -f "$STOP" ] && { log "stop file -> kill server"; kill "$SP" 2>/dev/null; exit 0; }
    sleep 30
  done
  log "server died (pid $SP); restarting in 30s -- tail $LOGD/server_${PORT}.log"
  sleep 30
done
log "stopped"
