#!/usr/bin/env bash
# Internal multi-host TPU operations: sync, launch, status, logs, stop and devices.
#
# launch_training.sh resolves one selected recipe before calling this file. It
# intentionally contains no personal path, run preset or hard-coded worker count.
#
# Run this from any Linux control host. TPU VMs never need gcloud: downloads use the
# metadata-service token, while resource discovery and GCS administration stay here.
#
#   ./scripts/tpu_ops.sh sync
#   ./scripts/tpu_ops.sh launch <run> [k.p=v ...]
#   ./scripts/tpu_ops.sh status <run>
#   ./scripts/tpu_ops.sh logs [worker] [lines]
#   ./scripts/tpu_ops.sh stop
#
# Rules baked in, each paid for by an earlier campaign (GC_TPU/tpu_ops_pitfalls.md):
#   * writing a script and executing it are SEPARATE ssh calls; a heredoc in the same argv
#     as a pgrep/pkill pattern makes the pattern match the wrapper.
#   * per-worker checks are issued one worker at a time; `--worker=all` swallows output in
#     a no-tty context and sometimes does not run at all.
#   * the code package is named by its content hash and never overwritten, so "which bytes
#     did that run use" stays answerable.
#   * key files are md5-checked at EXECUTION time, not at sync time: this repo is a live
#     worktree and has changed under a sync before.
#   * "launched" is not "running": every launch is followed by a pid check.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=_deployment_env.sh
source "$REPO/scripts/_deployment_env.sh"

# A direct TPU VM may deliberately use a resource name which differs from the
# conventional run-derived name. The deployment recipe still validates every
# other setting; only this transport-level resource identifier is overridden by
# the dedicated direct-VM launcher.
if [ -n "${FASTWAM_TPU_NAME_OVERRIDE:-}" ]; then
    export TPU="$FASTWAM_TPU_NAME_OVERRIDE"
fi

TPU=${TPU:?launch_training.sh must resolve TPU from YAML}
ZONE=${ZONE:?launch_training.sh must resolve ZONE from YAML}
WORKERS=${WORKERS:-}
BUCKET=${BUCKET:?launch_training.sh must resolve BUCKET from YAML}
CODE_PREFIX=${CODE_PREFIX:?launch_training.sh must resolve CODE_PREFIX}
RUN_PREFIX=${RUN_PREFIX:?launch_training.sh must resolve RUN_PREFIX}
TPU_USER=${TPU_USER:?launch_training.sh must resolve TPU_USER}
TPU_TMP=/tmp/fastwam
REMOTE_DIR=/tmp/fastwam/repo

# `gcloud compute tpus tpu-vm ssh` costs ~2.5 MINUTES per invocation on this workstation
# (an API describe plus key-propagation checks every time, on a box that is also running
# other work), which turned a four-worker sync into twenty minutes. Direct ssh with a
# multiplexed control socket is 3 s for the first call and 0.4 s after — the same 8-call
# sync drops from 20 minutes to under 10 seconds. gcloud is still what discovers the IPs;
# they are cached for the session, because the addresses of an on-demand TPU do not move.
SSH_OPTS=(
    -i "$HOME/.ssh/google_compute_engine"
    -o StrictHostKeyChecking=no
    -o UserKnownHostsFile=/dev/null
    -o LogLevel=ERROR
    -o ControlMaster=auto
    -o ControlPath="/tmp/fastwam-tpu-cm-%h"
    -o ControlPersist=900
    -o ConnectTimeout=15
    -o ServerAliveInterval=15
)
IP_CACHE="${TMPDIR:-/tmp}/tpu-ops-ips-$TPU"

worker_ips() {
    if [ ! -s "$IP_CACHE" ]; then
        timeout 120 gcloud compute tpus tpu-vm describe "$TPU" --zone="$ZONE" \
            --format="value(networkEndpoints[].accessConfig.externalIp)" \
            | tr ';' '\n' | grep -E '^[0-9]' > "$IP_CACHE" || return 1
    fi
    cat "$IP_CACHE"
}

ensure_workers() {
    [ -n "$WORKERS" ] && return 0
    local count
    count=$(worker_ips | awk 'NF {n++} END {print n+0}')
    [ "$count" -gt 0 ] || { echo "FATAL: describe returned no TPU worker endpoints" >&2; return 1; }
    WORKERS=$(seq 0 $((count - 1)) | paste -sd' ' -)
    export WORKERS
}

cmd_workers() {
    ensure_workers || return 1
    printf '%s\n' "$WORKERS"
}

worker_ip() {  # worker_ip <n>
    worker_ips | sed -n "$(( $1 + 1 ))p"
}

ssh_do() {  # ssh_do <worker> <command>  -- one worker, one thing
    local ip
    ip=$(worker_ip "$1")
    [ -n "$ip" ] || { echo "FATAL: no IP for worker $1"; return 1; }
    timeout 300 ssh "${SSH_OPTS[@]}" "$TPU_USER@$ip" "$2" 2>&1
}

pack() {  # sets PKG and HASH. Called directly, never in $(...) -- a command substitution
          # runs it in a subshell and the assignments would be lost (with `set -u`, loudly).
    local tmp
    tmp=$(mktemp -d)
    # Logs and the sync marker are excluded because the package is named by its content
    # hash: the babysitter writes its log inside the repo, so without this every sync it
    # runs produces a "new" package that is byte-identical in code, forcing a pointless
    # re-extract and re-install on all 16 hosts after every preemption.
    tar czf "$tmp/pkg.tgz" --exclude=.venv --exclude=.git --exclude=__pycache__ \
        --exclude=.pytest_cache --exclude='*.pyc' --exclude=traces \
        --exclude='*.log' --exclude=.last_sync_hash --exclude=logs \
        -C "$REPO" .
    HASH=$(md5sum "$tmp/pkg.tgz" | cut -c1-12)
    PKG="$tmp/fastwam_tpu_$HASH.tgz"
    mv "$tmp/pkg.tgz" "$PKG"
}

cmd_sync() {
    ensure_workers || exit 1
    local hash
    if [ "${REUSE_STAGED:-0}" = "1" ] && [ -s "$REPO/.last_sync_hash" ]; then
        hash=$(cat "$REPO/.last_sync_hash")
        gcloud storage ls "gs://$BUCKET/$CODE_PREFIX/fastwam_tpu_$hash.tgz" >/dev/null \
            || { echo "FATAL: staged package $hash is missing under $CODE_PREFIX"; exit 1; }
        echo "reusing staged package gs://$BUCKET/$CODE_PREFIX/fastwam_tpu_$hash.tgz"
    else
        pack
        hash=$HASH
        echo "package $PKG ($hash)"
        gcloud storage cp "$PKG" "gs://$BUCKET/$CODE_PREFIX/fastwam_tpu_$hash.tgz" || exit 1
        echo "$hash" > "$REPO/.last_sync_hash"
    fi

    # The bootstrap script is written first and executed second, on purpose.
    local boot="$REPO/scripts/_bootstrap.sh"
    [ -f "$boot" ] || { echo "FATAL: $boot is missing"; exit 1; }
    local b64
    b64=$(base64 -w0 "$boot")
    for w in $WORKERS; do
        echo "-- writing the bootstrap to worker $w"
        ssh_do "$w" "mkdir -p $TPU_TMP && echo '$b64' | base64 -d > $TPU_TMP/bootstrap.sh && md5sum $TPU_TMP/bootstrap.sh" || exit 1
    done
    for w in $WORKERS; do
        echo "-- bootstrapping worker $w"
        ssh_do "$w" "setsid nohup bash $TPU_TMP/bootstrap.sh $hash $BUCKET $CODE_PREFIX > $TPU_TMP/bootstrap.log 2>&1 < /dev/null & echo LAUNCHED_\$!"
    done
    echo "installing; poll with: $0 syncstatus $hash"
    echo "$hash" > "$REPO/.last_sync_hash"
}

cmd_syncstatus() {
    ensure_workers || exit 1
    local hash=${1:-$(cat "$REPO/.last_sync_hash" 2>/dev/null)}
    local ok=0
    for w in $WORKERS; do
        local out
        out=$(ssh_do "$w" "tail -8 $TPU_TMP/bootstrap.log 2>/dev/null || echo NO_LOG")
        echo "=== worker $w"
        echo "$out"
        grep -q "BOOT_OK $hash" <<<"$out" && ok=$((ok + 1))
    done
    echo "--- $ok/$(wc -w <<<"$WORKERS") workers report BOOT_OK $hash"
}

cmd_launch() {
    ensure_workers || exit 1
    local run=${1:?usage: launch <run_name> [overrides...]}
    shift
    local overrides="$*"
    local hash
    hash=$(cat "$REPO/.last_sync_hash" 2>/dev/null || echo unknown)
    local multihost=false
    [ "$(wc -w <<<"$WORKERS")" -gt 1 ] && multihost=true

    # The runner is a file so that no pattern in it lands in an ssh argv, and so the pid is
    # recorded for `stop` instead of being hunted for with pkill.
    local runner
    runner=$(mktemp)
    # LAUNCH_ENV is exported into the runner verbatim (e.g. XLA/libtpu tuning flags, which
    # are read from the environment by the TPU runtime and cannot be config overrides). It
    # is echoed into the log for the same reason the package hash is: a run tuned by
    # invisible environment is a run whose numbers nobody can reproduce later.
    cat > "$runner" <<EOS
#!/bin/bash
cd $REMOTE_DIR || exit 1
# Prove, at EXECUTION time, that the code about to run is the code that was synced.
# sharding.py and mot.py are on this list because that is where the ZeRO placement and the
# activation pins live: a run that silently used the pre-fix bytes would look completely
# normal and just report the baseline s/step, which is the exact shape of the 7.6 h that
# the tar-prefix incident burned.
md5sum src/fastwam_tpu/training/train.py src/fastwam_tpu/models/fastwam.py \\
       src/fastwam_tpu/training/sharding.py src/fastwam_tpu/models/mot.py \\
       src/fastwam_tpu/data/loader.py src/fastwam_tpu/config.py \\
       src/fastwam_tpu/specs/fastwam.yaml src/fastwam_tpu/specs/robotwin.yaml \\
       $TRAINING_CONFIG
echo "code package: $hash"
${LAUNCH_ENV:-}
# Report what is actually set, not the string we meant to set: flag values contain quotes,
# and echoing the raw assignment back mangles them exactly when they matter most.
echo "launch env: LIBTPU_INIT_ARGS=[\${LIBTPU_INIT_ARGS:-<unset>}] XLA_FLAGS=[\${XLA_FLAGS:-<unset>}]"
exec .venv/bin/python -m fastwam_tpu.training.train \\
  $TRAINING_CONFIG \\
  tpu.run_name=$run tpu.multihost=$multihost $overrides
EOS
    local b64
    b64=$(base64 -w0 "$runner")
    for w in $WORKERS; do
        echo "-- writing the runner to worker $w"
        ssh_do "$w" "mkdir -p $TPU_TMP && echo '$b64' | base64 -d > $TPU_TMP/run.sh && md5sum $TPU_TMP/run.sh"
    done
    for w in $WORKERS; do
        echo "-- starting worker $w"
        # APPEND, with a banner marking each launch. `>` truncated the log on every start,
        # which on a supervised spot run means the surviving log describes only the LAST
        # attempt. That cost the final loss of the 29,365-step run: eleven no-op relaunches
        # after completion overwrote the log of the launch that actually finished it.
        ssh_do "$w" "mkdir -p $TPU_TMP; printf '\\n===== launch %s pkg=%s =====\\n' \"\$(date -u +%Y-%m-%dT%H:%M:%SZ)\" '$hash' >> $TPU_TMP/train.log; setsid nohup bash $TPU_TMP/run.sh >> $TPU_TMP/train.log 2>&1 < /dev/null & echo \$! > $TPU_TMP/train.pid; cat $TPU_TMP/train.pid"
    done
    echo "started; confirm with: $0 status $run    (launched is not running)"
}

cmd_status() {
    ensure_workers || exit 1
    local run=${1:-}
    for w in $WORKERS; do
        echo "=== worker $w"
        ssh_do "$w" "PID=\$(cat $TPU_TMP/train.pid 2>/dev/null); if [ -n \"\$PID\" ] && kill -0 \$PID 2>/dev/null; then echo \"ALIVE pid=\$PID \$(ps -o etimes= -p \$PID)s\"; else echo \"NOT RUNNING (pid=\${PID:-none})\"; fi; tail -3 $TPU_TMP/train.log 2>/dev/null"
    done
    if [ -n "$run" ]; then
        echo "=== durable checkpoints"
        # The one signal that is immune to process-table pollution: does the step advance.
        gcloud storage ls "gs://$BUCKET/$RUN_PREFIX/$run/full/" 2>/dev/null | tail -5 \
            || echo "(nothing finalized yet)"
    fi
}

cmd_logs() {
    ensure_workers || exit 1
    local worker=${1:-0} lines=${2:-60}
    ssh_do "$worker" "tail -$lines $TPU_TMP/train.log"
}

cmd_collect_logs() {
    ensure_workers || exit 1
    local dest=${1:?usage: collect-logs LOCAL_DIRECTORY} failed=0 w tmp
    mkdir -p "$dest"
    for w in $WORKERS; do
        tmp="$dest/.worker_${w}.tmp.$$"
        if ssh_do "$w" "cat $TPU_TMP/train.log 2>/dev/null" > "$tmp"; then
            mv -f "$tmp" "$dest/worker_${w}.log"
        else
            rm -f "$tmp"; failed=1
        fi
        tmp="$dest/.worker_${w}_bootstrap.tmp.$$"
        if ssh_do "$w" "cat $TPU_TMP/bootstrap.log 2>/dev/null" > "$tmp"; then
            mv -f "$tmp" "$dest/worker_${w}_bootstrap.log"
        else
            rm -f "$tmp"; failed=1
        fi
    done
    return "$failed"
}

cmd_stop() {
    ensure_workers || exit 1
    # By pid, from the file the launcher wrote. `pkill -f fastwam` would match this very
    # ssh command's own argv and report itself as a mysterious exit 255.
    #
    # TERM then KILL, because TERM alone does not reliably stop this job: measured on the
    # v6e-64 slice, every one of the 16 processes took SIGTERM and kept training for 8+
    # minutes (still emitting step lines, chips still held). A multi-host JAX process
    # sitting in a collective does not get around to running Python signal handlers. The
    # escalation is safe here: checkpoints are versioned and committed atomically, so the
    # worst case is losing the steps since the last save, which is what resume replays.
    local grace=${STOP_GRACE:-45}
    for w in $WORKERS; do
        echo "=== worker $w"
        ssh_do "$w" "PID=\$(cat $TPU_TMP/train.pid 2>/dev/null); [ -n \"\$PID\" ] && kill \$PID 2>/dev/null && echo \"sent TERM to \$PID\" || echo \"no pid on file\""
    done
    echo "-- waiting up to ${grace}s for a clean exit before escalating"
    local waited=0
    while [ "$waited" -lt "$grace" ]; do
        sleep 5; waited=$((waited + 5))
        local alive=0
        for w in $WORKERS; do
            ssh_do "$w" "PID=\$(cat $TPU_TMP/train.pid 2>/dev/null); [ -n \"\$PID\" ] && kill -0 \$PID 2>/dev/null && echo A" | grep -q A && alive=$((alive + 1))
        done
        [ "$alive" -eq 0 ] && { echo "all workers exited on TERM after ${waited}s"; return 0; }
    done
    echo "-- still alive after ${grace}s; escalating to KILL"
    for w in $WORKERS; do
        echo "=== worker $w"
        ssh_do "$w" "PID=\$(cat $TPU_TMP/train.pid 2>/dev/null); if [ -n \"\$PID\" ] && kill -0 \$PID 2>/dev/null; then kill -9 \$PID 2>/dev/null && echo \"KILL9 \$PID\"; else echo \"already gone\"; fi"
    done
    echo "TPU chips take seconds to release; check with: $0 devices"
}

cmd_devices() {
    ensure_workers || exit 1
    # v4 exposes the chips as /dev/accel0..N; v6e exposes them as /dev/vfio/0..N and has no
    # /dev/accel at all. Checking only the v4 spelling on a v6e host reports "(free)" no
    # matter who holds the chips, which is the worst possible answer: it is the check you
    # run right before relaunching, and it would wave you into "TPU is already in use".
    for w in $WORKERS; do
        echo "=== worker $w"
        ssh_do "$w" 'held=$(sudo fuser /dev/accel* /dev/vfio/[0-9]* 2>/dev/null); if [ -n "$held" ]; then echo "HELD BY:$held"; else echo "(free)"; fi'
    done
}

case "${1:-}" in
    sync) shift; cmd_sync "$@" ;;
    workers) shift; cmd_workers "$@" ;;
    syncstatus) shift; cmd_syncstatus "$@" ;;
    launch) shift; cmd_launch "$@" ;;
    status) shift; cmd_status "$@" ;;
    logs) shift; cmd_logs "$@" ;;
    collect-logs) shift; cmd_collect_logs "$@" ;;
    stop) shift; cmd_stop "$@" ;;
    devices) shift; cmd_devices "$@" ;;
    *) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
