#!/usr/bin/env bash
# Keep a training run alive on a SPOT slice, unattended.
#
# Internal supervisor. launch_training.sh validates one self-contained YAML recipe, then
# supplies every value this process needs.
#
#   ./scripts/launch_training.sh configs/training/my_experiment.yaml RUN_NAME
#
# The slice will be preempted — measured spot lifetimes on v6e have been as short as
# 10-40 minutes, against a run that needs many hours. So preemption is the normal case in
# this loop, not the error case. One pass does:
#
#   1. queued resource missing   -> create it (with the env-bootstrap startup-script)
#   2. queued resource FAILED    -> delete and recreate; a QR never retries itself
#   3. not ACTIVE yet            -> wait
#   4. ACTIVE but VM not READY   -> wait
#   5. READY, env not installed  -> sync code to every host
#   6. READY, installed, no run  -> launch with tpu.resume=true
#   7. running                   -> watch the DURABLE step, not the process
#
# Design notes, each one a bug that has already been paid for:
#
#   * IPs change when the slice is re-provisioned, and tpu_ops.sh caches them. The cache is
#     deleted on every re-acquisition; a stale cache means ssh'ing at someone else's VM.
#   * `tpu.resume=true` is passed on EVERY launch including the first. It resolves to
#     allow_fresh (train.py:836 ties allow_fresh to resume_from being None), so a fresh
#     run_dir starts at step 1 and a re-launch picks up the newest finalized checkpoint.
#     Branching on "is this the first launch?" is a branch that would eventually be wrong
#     and would silently restart a finished day of training.
#   * Liveness is judged by the durable step on GCS, not by the process table. A process
#     can be alive and wedged; the step is the only signal that means progress.
#   * The startup-script installs the environment and never launches. See _vm_startup.sh
#     for why exactly one thing may decide to start training.
set -uo pipefail

RUN=${1:?usage: babysit.sh <run_name> [overrides...] (normally use launch_training.sh)}
shift
OVERRIDES="$*"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=_deployment_env.sh
source "${FASTWAM_DEPLOYMENT_ENV_LOADER:-$REPO/scripts/_deployment_env.sh}"
TPU=${TPU:?launch_training.sh must resolve TPU from YAML}
ZONE=${ZONE:?launch_training.sh must resolve ZONE from YAML}
ACC=${ACC:?launch_training.sh must resolve ACC from YAML}
BUCKET=${BUCKET:?launch_training.sh must resolve BUCKET from YAML}
CODE_PREFIX=${CODE_PREFIX:?launch_training.sh must resolve CODE_PREFIX}
TPU_USER=${TPU_USER:?launch_training.sh must resolve TPU_USER}
WORKERS=${WORKERS:-}
OPS=${OPS:?launch_training.sh must select tpu_ops.sh}
RT=${RT:?launch_training.sh must resolve RT from YAML}
RUN_PREFIX=${RUN_PREFIX:?launch_training.sh must resolve RUN_PREFIX}
export CODE_PREFIX TPU_USER RUN_PREFIX
POLL=${POLL:-120}
STALL_LIMIT=${STALL_LIMIT:-3600}   # seconds without a new durable step before relaunching
mkdir -p "$CONTROL_LOG_DIR" "$LAUNCH_LOG_DIR"
LOG=${LOG:-$CONTROL_LOG_DIR/supervisor_$RUN.log}
PROBE_WORKERS=${PROBE_WORKERS:-}
GCS_LOG_SYNC=${GCS_LOG_SYNC:-1}
LOG_SYNC_INTERVAL=${LOG_SYNC_INTERVAL:-300}
last_log_sync=0
TPU_LOG_SESSION=${TPU_LOG_SESSION:-}
RELEASE_ON_COMPLETE=${RELEASE_ON_COMPLETE:-1}

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

sync_logs_if_due() {
    [ "$GCS_LOG_SYNC" = 1 ] || return 0
    local now failed=0
    now=$(date +%s)
    [ $((now - last_log_sync)) -ge "$LOG_SYNC_INTERVAL" ] || return 0
    last_log_sync=$now
    gcloud storage cp "$LOG" "$GCS_LOG_DIR/control/supervisor.log" >/dev/null 2>&1 || failed=1
    gcloud storage cp --recursive "$LAUNCH_LOG_DIR" "$GCS_LOG_DIR/control" >/dev/null 2>&1 || failed=1
    if [ -n "$WORKERS" ]; then
        [ -n "$TPU_LOG_SESSION" ] \
            || TPU_LOG_SESSION="$(date -u +%Y%m%dT%H%M%SZ)-$$-${RANDOM:-0}"
        local snapshot="$CONTROL_LOG_DIR/tpu/$RUN/$TPU_LOG_SESSION"
        mkdir -p "$snapshot"
        "$OPS" collect-logs "$snapshot" >/dev/null 2>&1 || failed=1
        # The trailing slash is intentional: it makes this an existing-folder-style
        # destination on every poll, so the snapshot basename remains the session ID.
        gcloud storage cp --recursive "$snapshot" \
            "$GCS_LOG_DIR/tpu/" >/dev/null 2>&1 || failed=1
    fi
    if [ "$failed" -ne 0 ]; then
        last_log_sync=0
        log "WARNING: durable log snapshot was incomplete; retrying on the next poll"
        return 1
    fi
}

durable_step() {
    gcloud storage ls "gs://$BUCKET/$RUN_PREFIX/$RUN/full/" 2>/dev/null \
        | grep -oE 'step_[0-9]+' | sort -u | tail -1
}

qr_state() {
    # "the query failed" and "the resource is gone" are NOT the same thing, and this
    # function used to return an empty string for both. The caller reads absent as
    # "create one", so a transient describe timeout looked exactly like a preemption --
    # observed 2026-08-04 07:21, which logged a preemption and fired a spurious create
    # while the original VM was healthy and training was still running. Only a NOT_FOUND
    # from the API means absent; anything else unexplained means "ask again later".
    local out rc
    out=$(timeout 120 gcloud compute tpus queued-resources describe "$TPU" \
        --zone="$ZONE" --format='value(state.state)' 2>&1); rc=$?
    if [ "$rc" -eq 0 ] && [ -n "$out" ]; then printf '%s\n' "$out"; return 0; fi
    if printf '%s' "$out" | grep -qiE "not ?found|does not exist|was not found"; then
        echo ABSENT; return 0
    fi
    echo QUERY_FAILED
}

vm_state() {
    timeout 120 gcloud compute tpus tpu-vm describe "$TPU" \
        --zone="$ZONE" --format='value(state)' 2>/dev/null
}

create_qr() {
    local hash
    hash=$(cat "$REPO/.last_sync_hash" 2>/dev/null)
    local meta_args=()
    if [ -n "$hash" ] && [ -f "$REPO/scripts/_vm_startup.sh" ]; then
        meta_args=(--metadata-from-file "startup-script=$REPO/scripts/_vm_startup.sh"
                   --metadata "fastwam-hash=$hash,fastwam-bucket=$BUCKET,fastwam-code-prefix=$CODE_PREFIX,fastwam-user=$TPU_USER")
        log "creating QR with env-bootstrap startup-script (hash=$hash)"
    else
        log "creating QR WITHOUT startup-script (no .last_sync_hash yet); sync will be manual"
    fi
    # Creation can legitimately take longer than the CLI's synchronous wait (the
    # queued-resource operation may spend many minutes assembling a v6e slice).  Submit
    # asynchronously and let the next polling pass observe the resource state; treating
    # the client-side wait timeout as a network failure caused repeated duplicate requests
    # and prevented the supervisor from ever reaching READY.
    timeout 120 gcloud compute tpus queued-resources create "$TPU" \
        --node-id="$TPU" --zone="$ZONE" --accelerator-type="$ACC" \
        --runtime-version="$RT" --spot --async "${meta_args[@]}" 2>&1 | tee -a "$LOG"
}

finish_run() {
    last_log_sync=0
    sync_logs_if_due || log "WARNING: final log snapshot was incomplete"
    if [ "$RELEASE_ON_COMPLETE" != 1 ]; then
        log "training complete; RELEASE_ON_COMPLETE=$RELEASE_ON_COMPLETE, leaving TPU in place"
        return 0
    fi
    log "training complete; releasing queued TPU resource $TPU"
    if timeout 600 gcloud compute tpus queued-resources delete "$TPU" \
        --zone="$ZONE" --quiet --force 2>&1 | tail -2 | tee -a "$LOG"; then
        log "queued TPU resource $TPU released"
    else
        log "WARNING: automatic TPU release failed; release $TPU in $ZONE manually"
    fi
    [ "$GCS_LOG_SYNC" = 1 ] \
        && gcloud storage cp "$LOG" "$GCS_LOG_DIR/control/supervisor.log" >/dev/null 2>&1 \
        || true
}

SSH_READY_MARKER="${TMPDIR:-/tmp}/fastwam-ssh-ready-$TPU"
forget_ips() {
    rm -f "${TMPDIR:-/tmp}/tpu-ops-ips-$TPU" "$SSH_READY_MARKER"
    WORKERS=""
    PROBE_WORKERS=""
    TPU_LOG_SESSION=""
    export WORKERS TPU_LOG_SESSION
}

discover_workers() {
    [ -n "$WORKERS" ] && return 0
    WORKERS=$("$OPS" workers 2>/dev/null) || return 1
    [ -n "$WORKERS" ] || return 1
    export WORKERS
    set -- $WORKERS
    PROBE_WORKERS="$1 ${!#}"
    log "discovered $(wc -w <<<"$WORKERS") TPU workers: $WORKERS"
}

ensure_ssh_access() {
    [ -f "$SSH_READY_MARKER" ] && return 0
    local control_use
    control_user=$(id -un)
    if [ "$TPU_USER" != "$control_user" ]; then
        log "TPU_USER=$TPU_USER differs from control user $control_user; skipping gcloud SSH key priming"
        touch "$SSH_READY_MARKER"
        return 0
    fi
    log "priming gcloud SSH access for $TPU_USER on every worker"
    for w in $WORKERS; do
        timeout 300 gcloud compute tpus tpu-vm ssh "$TPU" --zone="$ZONE" \
            --worker="$w" --command='true' >/dev/null 2>&1 || return 1
    done
    touch "$SSH_READY_MARKER"
    log "gcloud SSH access ready for all workers"
}

# grep -c reads all of its input, so no SIGPIPE and no pipefail false negative here --
# unlike grep -q, which is what broke env_ready.
alive_count() { "$OPS" status 2>/dev/null | grep -c ALIVE; }

# Has the run already finished? The training script is the authority: on resume it computes
# start_step = last_ckpt + 1, finds it past `steps`, runs no iterations and exits with
# "done: 0 steps". A supervisor that only asks "is a process alive" reads that clean exit as
# a crash and relaunches -- forever. The completed 29,365-step run was relaunched eleven
# times that way, and because each launch truncated the log, the record of the launch that
# actually finished it was destroyed.
# Judged on the LAST "done: N steps" line only. Two reasons it must be the last one and
# not any one: the log is now appended across launches, so a previous completion's
# "done: 0 steps" stays in the file forever and would condemn every later run; and the
# window has to be wide enough to contain it (an 8-line tail missed it -- caught by the
# self-test against the real post-completion log, not by reading the code).
#
# WHEN this is asked matters as much as what it asks. It used to be consulted in exactly
# one place: right after a launch, once INIT_CHECK_DELAY had elapsed, and only if the
# process was already dead by then. A finished run does not die that fast -- it still pays
# the ~160 s of model load and pool prefill before discovering it has nothing to do -- so
# at the 150 s check it was ALIVE, the completion test was skipped, and the next poll found
# a dead process and relaunched. Forever, at v6e-64 prices.
#
# Observed on v6e-fixed-30k, 2026-08-06: durable step 29,365 (its configured total) at
# 12:43:41Z, relaunched at 12:46:06Z and again at 12:51:29Z with nothing left to do.
#
# The fix is below in the main loop: ask BEFORE deciding to launch, independently of
# whether anything is alive. That makes the test's timing irrelevant -- however late the
# no-op exit comes, the next pass sees it.
run_finished() {
    local out last
    out=$("$OPS" logs 0 200 2>/dev/null)
    last=$(printf '%s\n' "$out" | grep -oE "done: [0-9]+ steps" | tail -1)
    [ -n "$last" ] || return 1
    [ "$last" = "done: 0 steps" ] || return 1
    return 0
}

# Optional: capture a profiler trace on the next relaunch, with PROFILE_ON_RESTART=1.
#
# `tpu.profile.start_step` is an ABSOLUTE step and profiling is read once at startup, so a
# hardcoded value is wrong the moment the run resumes past it -- it would silently neve
# trigger. The step is therefore derived from wherever this relaunch is actually resuming
# from. Riding the next preemption costs nothing: on a spot slice the restart is going to
# happen anyway, so the trace is free rather than paid for by interrupting a healthy run.
profile_args() {
    [ "${PROFILE_ON_RESTART:-0}" = "1" ] || return 0
    local resume_at=0
    [ -n "${1:-}" ] && resume_at=$((10#${1#step_}))
    echo "tpu.profile.enabled=true tpu.profile.start_step=$((resume_at + 10))" \
         "tpu.profile.num_steps=${PROFILE_STEPS:-5}"
}

env_ready() {  # is the CURRENT package installed on the first and last host?
    local hash
    hash=$(cat "$REPO/.last_sync_hash" 2>/dev/null)
    # An empty hash would turn the check below into `grep "BOOT_OK "`, which matches a host
    # still carrying some OLDER package and reports the slice ready to run code it does not
    # have. No hash means nothing has been synced from here yet: say not-ready and sync.
    [ -n "$hash" ] || return 1
    for w in $PROBE_WORKERS; do
        # Capture, then match. NOT `syncstatus | grep -q`: with `set -o pipefail` (top of
        # this file) that pipeline reports FAILURE on success -- grep -q exits the instant
        # it matches, the still-writing producer takes SIGPIPE, and pipefail surfaces its
        # 141. env_ready would then be permanently false, so the babysitter would sync in
        # a loop and never launch; and since each sync `rm -rf`s the repo on every host,
        # it would eventually do that underneath a running job.
        # Match the WORKER's own line, which starts the line, not syncstatus's summary
        # line "--- 0/1 workers report BOOT_OK <hash>". That summary prints the same
        # "BOOT_OK <hash>" text whether zero or all workers reported, so a substring test
        # is true even when nothing is installed -- which is how a fresh, code-less slice
        # passed this check 53 times in a row and got launched into.
        local out
        out=$(WORKERS="$w" "$OPS" syncstatus "$hash" 2>/dev/null)
        printf '%s\n' "$out" | grep -qE "^[[:space:]]*BOOT_OK $hash([[:space:]]|$)" || return 1
    done
    return 0
}

log "=== babysitter start: run=$RUN overrides='$OVERRIDES' ==="
last_step=""
last_progress=$(date +%s)
LAUNCH_FAILS=0
FLAG_FALLBACK_AFTER=${FLAG_FALLBACK_AFTER:-2}
# How many launches from the SAME durable step are allowed before the slice, not the
# launch, is treated as the problem. See the barren-launch guard in the loop below.
BARREN_LIMIT=${BARREN_LIMIT:-3}
barren=0
barren_step=""
# Has THIS supervisor launched into this slice yet? Gates the completion check; see the
# loop for why a log we did not write cannot be trusted.
launched_once=0

while true; do
    state=$(qr_state)

    case "$state" in
        QUERY_FAILED)
            log "queued-resource query failed (API blip, not a preemption) -> retrying"
            sleep "$POLL"; continue
            ;;
        ABSENT|"")
            log "no queued resource -> creating"
            forget_ips
            create_qr
            sleep "$POLL"; continue
            ;;
        FAILED|SUSPENDED|SUSPENDING)
            log "QR state=$state (preempted or rejected) -> delete + recreate"
            forget_ips
            timeout 600 gcloud compute tpus queued-resources delete "$TPU" \
                --zone="$ZONE" --quiet --force 2>&1 | tail -2 | tee -a "$LOG"
            sleep "$POLL"; continue
            ;;
        ACTIVE) ;;
        *)
            log "QR state=$state -> waiting"
            sleep "$POLL"; continue
            ;;
    esac

    if [ "$(vm_state)" != "READY" ]; then
        log "QR ACTIVE but VM not READY -> waiting"
        forget_ips
        sleep "$POLL"; continue
    fi

    if ! discover_workers; then
        log "VM READY but worker endpoint discovery failed -> retrying"
        sleep "$POLL"; continue
    fi

    if ! ensure_ssh_access; then
        log "gcloud SSH key propagation failed -> waiting before retry"
        sleep "$POLL"; continue
    fi

    # NEVER sync while training is alive. `sync` runs _bootstrap.sh, which does
    # replaces `/tmp/fastwam/repo` on every host -- under a live job that deletes the code
    # out from under it. env_ready is an ssh-based probe, so a transient ssh failure
    # reads as "environment missing"; without this guard that blip destroys a healthy
    # 60-hour run. Structural fix, not a probe fix: syncing under a live job is neve
    # the right thing regardless of why the probe said no.
    if ! env_ready && [ "$(alive_count)" -eq 0 ]; then
        log "environment missing on the slice (and nothing running) -> syncing code to all hosts"
        forget_ips
        "$OPS" sync 2>&1 | tail -3 | tee -a "$LOG"
        sleep 180
        "$OPS" syncstatus 2>&1 | tail -1 | tee -a "$LOG"
        continue
    fi

    step=$(durable_step)
    now=$(date +%s)
    if [ -n "$step" ] && [ "$step" != "$last_step" ]; then
        log "durable progress: $step"
        last_step="$step"; last_progress=$now
        # The slice just proved it can run the job. Whatever launch history got us here
        # stops being evidence against it.
        barren=0; barren_step=""
    fi

    alive=$(alive_count)
    if [ "${alive:-0}" -eq 0 ] && [ "$launched_once" -eq 1 ] && run_finished; then
        # Nothing running, and the last thing WE launched exited having done no work. The
        # job is over. Asked HERE -- before the launch decision, not after a launch -- so
        # that how long a no-op startup takes to exit cannot hide the answer.
        #
        # `launched_once` is not optional. `/tmp/fastwam/train.log` is appended across launches
        # and survives on a slice that is reused rather than re-provisioned, so a brand new
        # run started on a warm machine can read a PREVIOUS run's "done: 0 steps" and would
        # exit without ever launching -- silently, which is the failure mode this whole file
        # is built to avoid. Only a log we have written into this session may be trusted.
        # (This is the property the old post-launch placement had for free, and the reason
        # moving the check needed more than moving the check.)
        log "=================================================================="
        log " RUN COMPLETE: the training script resumed past its last step and"
        log " exited without doing any work. Durable checkpoint: ${step:-unknown}."
        log " The babysitter is stopping -- there is nothing left to supervise."
        log " The babysitter will save final logs and release the TPU."
        log "=================================================================="
        finish_run
        exit 0
    fi

    if [ "${alive:-0}" -eq 0 ]; then
        # Is this slice actually capable of running the job at all?
        #
        # `LAUNCH_FAILS` below cannot answer that: it only counts launches that die WITHIN
        # INIT_CHECK_DELAY. Measured 2026-08-05/06 on this very run -- a slice reported
        # ACTIVE and READY while every launch survived the init check and then died 5-7
        # minutes in, before finishing the 200 steps that write the first checkpoint. This
        # supervisor relaunched into it 25 times across 4h20m and logged nothing but
        # cheerful "launching..." lines. It could not see the loop it was in, because
        # `last_progress` is reset on every launch (below) and the STALL_LIMIT branch is an
        # `elif` on a process being ALIVE -- so a process that keeps dying can never reach
        # the one check that would have caught it.
        #
        # The signal that does see it is the durable step: N launches, same step, nothing
        # moved. That is a property of the SLICE rather than of the code, and the remedy
        # that has always worked in practice is a different slice -- which is exactly what
        # ended the 4h20m episode (a preemption, by luck, not by design).
        #
        # A FINISHED run looks exactly like a barren one to this counter -- same durable
        # step, launch after launch, no progress -- and the remedy would then be applied to
        # a perfectly good machine: three no-op relaunches and the supervisor deletes the
        # queued resource, re-queues, waits an hour, syncs, launches, and finds the run
        # still finished. Two individually correct mechanisms composing into a self-feeding
        # acquire-idle-destroy loop. The completion check above returns before reaching this
        # in every case it can see; this guard covers the one it cannot -- the first pass of
        # a fresh supervisor, where `launched_once` is still 0.
        if run_finished; then
            log "not counting this as a barren launch: the run is already finished"
        elif [ "${step:-step_0}" = "${barren_step:-}" ]; then
            barren=$((barren + 1))
        else
            barren=1; barren_step="${step:-step_0}"
        fi
        if [ "$barren" -gt "$BARREN_LIMIT" ]; then
            # Belt and braces with the exclusion above and the completion check at the top
            # of the loop. Three guards for one condition looks like too many until you
            # notice what the failure costs: this branch DELETES a queued resource, and a
            # finished run is indistinguishable from a bad slice by the only signal this
            # branch reads. If either earlier guard is ever refactored away, the machine
            # that survives is the one this catch stops from being thrown away.
            if run_finished; then
                log "=================================================================="
                log " RUN COMPLETE: $barren_step is the last step and the training script"
                log " exits immediately when relaunched. The slice is FINE -- it is the"
                log " run that is over. Saving final logs and releasing the TPU."
                log "=================================================================="
                finish_run
                exit 0
            fi
            log "!!! $BARREN_LIMIT launches from $barren_step produced NO durable progress."
            log "!!! Treating the SLICE as bad instead of relaunching into it again."
            log "!!! Deleting the queued resource; the next pass will recreate it."
            forget_ips
            timeout 600 gcloud compute tpus queued-resources delete "$TPU" \
                --zone="$ZONE" --quiet --force 2>&1 | tail -2 | tee -a "$LOG"
            barren=0; barren_step=""
            sleep "$POLL"; continue
        fi
        prof=$(profile_args "${step:-}")
        log "no training process on the slice -> launching (resume=true) at ${step:-step_0}${prof:+ [profiling: $prof]}"
        # shellcheck disable=SC2086
        # Full output to its own file, not `| tail -2`. The truncated form cost a whole
        # night: 53 launches failed and every one of their error messages was thrown away,
        # so "why does training never survive" had no evidence to answer it. A superviso
        # that discards the output of the thing it supervises is blind by construction.
        mkdir -p "$LAUNCH_LOG_DIR"
        LAUNCH_OUT="$LAUNCH_LOG_DIR/${RUN}_$(date -u +%Y%m%dT%H%M%SZ).log"
        "$OPS" launch "$RUN" tpu.resume=true $OVERRIDES $prof > "$LAUNCH_OUT" 2>&1
        launched_once=1
        log "launch output -> $LAUNCH_OUT ($(wc -l < "$LAUNCH_OUT") lines)"
        grep -ciE "connection|refused|timed out|denied|FATAL|error" "$LAUNCH_OUT" 2>/dev/null \
            | grep -qv '^0$' && log "  !! launch output contains errors: $(grep -iE "connection|refused|timed out|denied|FATAL|error" "$LAUNCH_OUT" | head -2 | tr '\n' ' ')"
        tail -2 "$LAUNCH_OUT" | tee -a "$LOG"
        last_progress=$(date +%s)

        # Did it survive startup? A bad LIBTPU_INIT_ARGS kills the process during TPU init,
        # within seconds -- and this loop would then relaunch it with the same bad env,
        # forever. That is the same shape as the profiler crash-loop: a config that fails
        # only on restart, in a supervisor whose whole job is to restart. So the env that
        # could cause it is the env this check is allowed to withdraw.
        #
        # This used to be one `sleep 150` and one look. A single point in time cannot
        # answer the question: 150 s was chosen as "less than a healthy startup, so if it is
        # already dead the launch failed" -- but it is ALSO less than the startup of a run
        # that is going to exit immediately (model load + pool prefill is ~160 s; a
        # resume-at-final-step took 3-5 minutes end to end). At the sample it was still
        # alive, so the verdict was "healthy", and the next poll found a corpse and
        # relaunched. Forever.
        #
        # So: poll until the process SETTLES, and bound the wait. Two settled states, and
        # they are not the same event:
        #   * gone     -> it exited. Ask why (finished, or a failed launch).
        #   * running  -> it got past the window in which an init-time rejection would have
        #                 killed it, i.e. the thing this check exists to detect did not
        #                 happen. Stop waiting; the main loop supervises from here.
        settle_t0=$(date +%s)
        settle_alive=1
        while :; do
            sleep "${INIT_POLL:-20}"
            settle_alive=$(alive_count); settle_alive=${settle_alive:-1}
            if [ "$settle_alive" -eq 0 ]; then
                # Confirm before believing it. `alive_count` is an ssh probe, and a blip
                # that reads as death would be counted as a failed launch.
                sleep "${INIT_POLL:-20}"
                settle_alive=$(alive_count); settle_alive=${settle_alive:-1}
                [ "$settle_alive" -eq 0 ] && break            # settled: gone
            fi
            elapsed=$(( $(date +%s) - settle_t0 ))
            # Alive and past the window in which an init-time rejection would have killed
            # it: settled healthy. The main loop supervises from here.
            [ "$elapsed" -ge "${INIT_CHECK_DELAY:-240}" ] && break
            # Bounded regardless, so this can never become the thing that hangs.
            [ "$elapsed" -ge "${INIT_SETTLE_LIMIT:-480}" ] && break
        done
        if [ "$settle_alive" -ne 0 ]; then
            log "launch settled after $(( $(date +%s) - settle_t0 ))s: alive past the init window"
        else
            log "launch exited after $(( $(date +%s) - settle_t0 ))s; asking whether the run is finished"
        fi
        # Secondary completion check. The authoritative one is at the top of the loop, and
        # is the reason this one is now only an optimisation: it ends a finished run one
        # poll sooner. It is reached as soon as the process is GONE, however long that took.
        if [ "${settle_alive:-0}" -eq 0 ] && run_finished; then
            log "=================================================================="
            log " RUN COMPLETE: the training script resumed past its last step and"
            log " exited without doing any work. Durable checkpoint: ${step:-unknown}."
            log " The babysitter will save final logs and release the TPU."
            log "=================================================================="
            finish_run
            exit 0
        fi
        # `settle_alive`, not a fresh probe: the settle loop already established this, and
        # re-asking would reintroduce the single-sample reading the loop exists to replace.
        # A finished run never reaches here -- it exited above -- so this counts only
        # launches that really did fail to start.
        if [ "$settle_alive" -eq 0 ]; then
            LAUNCH_FAILS=$((LAUNCH_FAILS + 1))
            log "launch did NOT survive startup (consecutive failures: $LAUNCH_FAILS)"
            if [ -n "${LAUNCH_ENV:-}" ] && [ "$LAUNCH_FAILS" -ge "$FLAG_FALLBACK_AFTER" ]; then
                log "!!! $LAUNCH_FAILS consecutive init failures WITH a tuning env set."
                log "!!! WITHDRAWING it and continuing flagless. Discarded: $LAUNCH_ENV"
                log "!!! The run continues WITHOUT the tuning flags -- treat this as a"
                log "!!! failed flag experiment, not as a healthy run."
                LAUNCH_ENV=""
                export LAUNCH_ENV
                LAUNCH_FAILS=0
            fi
        else
            [ "$LAUNCH_FAILS" -gt 0 ] && log "launch survived startup; failure counter reset"
            LAUNCH_FAILS=0
        fi
    elif [ $((now - last_progress)) -gt "$STALL_LIMIT" ]; then
        log "processes alive but no new durable step in $((now - last_progress))s -> restarting"
        "$OPS" stop 2>&1 | tail -1 | tee -a "$LOG"
        sleep 120
        "$OPS" devices 2>&1 | grep -c HELD | tee -a "$LOG"
        last_progress=$(date +%s)
    fi

    sync_logs_if_due
    sleep "$POLL"
done
