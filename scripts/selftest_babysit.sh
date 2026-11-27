#!/usr/bin/env bash
# Self-test for babysit.sh. Touches nothing real.
#
#     bash scripts/selftest_babysit.sh                            # non-zero exit on failure
#     TARGET=<other babysit.sh> bash scripts/selftest_babysit.sh  # test another copy
#
# Why this file exists. Every guard in babysit.sh is there because a supervisor could not
# see a loop it was in, and the last three were found by watching a real slice burn rathe
# than by reading the code. The first `run_finished()` was itself broken in exactly that
# way -- an 8-line log tail could not contain the two lines its verdict needed -- and what
# caught it was a REAL post-completion log used as a fixture. So this has two parts:
#
#   Part 1 runs the real `run_finished` against real training logs (tests/fixtures/babysit/).
#   Part 2 drives the whole decision loop from stubs.
#
# `gcloud` and `timeout` are shadowed so a stray call cannot reach the control plane. Not
# doing that the first time this was written meant a test run reached the queued-resource
# DELETE branch and issued a real, malformed API call. It matched nothing, but "the test
# suite talks to the control plane" is not a property to leave to luck.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET=${TARGET:-$HERE/babysit.sh}
FIX="$HERE/../tests/fixtures/babysit"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
FAILED=0

cat > "$WORK/deployment_env.sh" <<'EOS'
export TPU=selftest ZONE=selftest ACC=selftest RT=selftest-runtime
export BUCKET=selftest GCS_BUCKET=selftest
export CODE_PREFIX=selftest/code RUN_PREFIX=selftest/runs
export TPU_USER=selftest
export CONTROL_LOG_DIR=/tmp/selftest-control LAUNCH_LOG_DIR=/tmp/selftest-launch
export GCS_LOG_DIR=gs://selftest/selftest/runs/selftest-run/logs
EOS
export FASTWAM_DEPLOYMENT_ENV_LOADER="$WORK/deployment_env.sh"
export GCS_LOG_SYNC=0

pass() { printf 'ok    %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; FAILED=1; }

# --------------------------------------------------------------------------- #
# Part 0: static call wiring
# --------------------------------------------------------------------------- #
echo "-- part 0: supervisor call wiring"
if grep -qE '^[[:space:]]+create_qr$' "$TARGET" \
   && ! grep -qE '^[[:space:]]+create_q$' "$TARGET"; then
    pass "ABSENT queued-resource state calls the defined create_qr function"
else
    fail "ABSENT queued-resource state is not wired to create_qr"
fi
if grep -q 'snapshot="\$CONTROL_LOG_DIR/tpu/\$RUN/\$TPU_LOG_SESSION"' "$TARGET" \
   && grep -q 'GCS_LOG_DIR/tpu/"' "$TARGET" \
   && grep -q 'finish_run' "$TARGET"; then
    pass "TPU logs are separated by Spot session and completion uses finish_run"
else
    fail "durable Spot-session logs or completion cleanup are not wired"
fi
# --------------------------------------------------------------------------- #
# Part 1: run_finished against REAL logs
# --------------------------------------------------------------------------- #
# The function is used as written: a stub $OPS answers `logs` with a fixture, so the tail
# width, the regex and the last-match rule are all exercised against real text.
cat > "$WORK/ops" <<'EOS'
#!/usr/bin/env bash
[ "$1" = logs ] && exec cat "$FIXTURE"
exit 0
EOS
chmod +x "$WORK/ops"

check_log() {  # check_log <fixture> <finished|not> <why>
    local fixture=$1 expect=$2 why=$3 got
    if FIXTURE="$FIX/$fixture" OPS="$WORK/ops" \
       bash -c ". '$WORK/run_finished.sh'; run_finished"; then got=finished; else got=not; fi
    [ "$got" = "$expect" ] \
        && pass "run_finished($fixture) = $got  -- $why" \
        || fail "run_finished($fixture) = $got, wanted $expect  -- $why"
}

echo "target: $TARGET"
echo "-- part 1: run_finished against real training logs"
awk '/^run_finished\(\) \{/,/^\}/' "$TARGET" > "$WORK/run_finished.sh"
if [ ! -s "$WORK/run_finished.sh" ]; then
    fail "could not extract run_finished() from $TARGET"
elif [ ! -d "$FIX" ]; then
    fail "missing fixtures at $FIX"
else
    # The real tail a supervisor sees after a no-op relaunch. This fixture contains BOTH
    # `done: 26165 steps` (a launch that really trained) and, 44 lines later, the
    # `done: 0 steps` that ends it -- so it pins the "last match wins" rule AND a tail wide
    # enough to reach the verdict. A naive "any done: line" reading calls this unfinished;
    # an 8-line tail sees neither.
    check_log train_log_completed.txt        finished "no-op relaunch: the LAST done: is 0 steps"
    check_log train_log_finished_a_chunk.txt not      "a launch that trained 26,165 steps is not the end of the run"
    check_log train_log_midrun.txt           not      "mid-training: no done: line in the window at all"
fi

# --------------------------------------------------------------------------- #
# Part 2: the decision loop, from stubs
# --------------------------------------------------------------------------- #
python3 - "$TARGET" "$WORK/loop.sh" <<'PY'
import re, sys, pathlib
src = pathlib.Path(sys.argv[1]).read_text()
stubs = '''
gcloud() { echo "GCLOUD $*" >> "$S_GCLOUD"; return 0; }
timeout() { shift; "$@"; }
qr_state() { echo ACTIVE; }
vm_state() { echo READY; }
env_ready() { return 0; }
durable_step() { echo step_000029365; }
alive_count() { cat "$S_ALIVE" 2>/dev/null || echo 0; }
run_finished() { [ -f "$S_FINISHED" ]; }
create_qr() { echo CREATE >> "$S_GCLOUD"; }
forget_ips() { :; }
discover_workers() { WORKERS="0 1"; PROBE_WORKERS="0 1"; export WORKERS; return 0; }
'''
i = src.index('log "=== babysitter start')
src = src[:i] + stubs + "\n" + src[i:]
src = src.replace(
    '"$OPS" launch "$RUN" tpu.resume=true $OVERRIDES $prof > "$LAUNCH_OUT" 2>&1',
    'echo LAUNCHED > "$LAUNCH_OUT"; echo LAUNCH >> "$S_LAUNCHES"; echo 1 > "$S_ALIVE"')
# The scenario that matters: a finished run stays ALIVE through the settle window (it pays
# ~160 s of model load before discovering it has nothing to do) and only then exits. The
# first settle probe therefore sees it alive; later probes see it gone.
src = re.sub(r'sleep "\$\{INIT_POLL:-\d+\}"', 'echo 0 > "$S_ALIVE"; sleep 0', src, count=1)
src = re.sub(r'sleep "\$\{INIT_POLL:-\d+\}"', 'sleep 0', src)
# The OLD single-sample shape, so this file can also be pointed at a pre-fix copy and show
# what it does. There the process must look ALIVE at the one sample -- that is the whole
# bug -- and dead by the next poll.
src = re.sub(r'sleep "\$\{INIT_CHECK_DELAY:-\d+\}"', 'echo 1 > "$S_ALIVE"; sleep 0', src)
src = re.sub(r'sleep "\$POLL"', 'echo 0 > "$S_ALIVE"; sleep 0', src)
pathlib.Path(sys.argv[2]).write_text(src)
PY

case_() {  # case_ <name> <finished?> <want_exit> <lmin> <lmax> <want_complete> <qr: none|any>
    local name=$1 finished=$2 want_exit=$3 lmin=$4 lmax=$5 want_done=$6 want_qr=${7:-any}
    export S_ALIVE=$WORK/alive S_FINISHED=$WORK/fin S_GCLOUD=$WORK/gcloud \
           S_LAUNCHES=$WORK/launches LOG=$WORK/log LAUNCH_LOG_DIR=$WORK
    : > "$S_LAUNCHES"; : > "$S_GCLOUD"; : > "$LOG"; echo 0 > "$S_ALIVE"
    [ "$finished" = yes ] && : > "$S_FINISHED" || rm -f "$S_FINISHED"
    local case_out=$WORK/case.out
    command timeout 10 bash "$WORK/loop.sh" selftest-run >"$case_out" 2>&1
    local rc=$? n d q ok=1
    n=$(grep -c LAUNCH "$S_LAUNCHES"); d=$(grep -c 'RUN COMPLETE' "$LOG")
    # `grep -c` prints its count AND exits 1 when the count is zero, so a `|| echo 0`
    # fallback appends a SECOND zero and the arithmetic below sees "0\n0".
    q=$(grep -c 'queued-resources delete' "$S_GCLOUD" 2>/dev/null || true); q=${q:-0}
    [ "$rc" = "$want_exit" ] || ok=0
    [ "$n" -ge "$lmin" ] && [ "$n" -le "$lmax" ] || ok=0
    [ "$d" = "$want_done" ] || ok=0
    # A completed run deliberately releases exactly one queued resource. More than one is
    # the old acquire-idle-destroy loop; zero leaks an idle TPU after the final log sync.
    [ "$want_qr" = none ] && [ "$q" -ne 0 ] && ok=0
    [ "$want_qr" = one ] && [ "$q" -ne 1 ] && ok=0
    if [ "$ok" = 1 ]; then
        printf 'ok    %-46s exit=%-4s launches=%-3s complete=%s qr-deletes=%s\n' "$name" "$rc" "$n" "$d" "$q"
    else
        printf 'FAIL  %-46s exit=%-4s launches=%-3s complete=%s qr-deletes=%s  (wanted exit=%s launches %s-%s complete=%s qr=%s)\n' \
               "$name" "$rc" "$n" "$d" "$q" "$want_exit" "$lmin" "$lmax" "$want_done" "$want_qr"
        sed 's/^/      /' "$case_out"
        FAILED=1
    fi
}

export TPU=selftest ZONE=selftest ACC=selftest BUCKET=selftest WORKERS="0 1" OPS=/bin/true
export RT=selftest-runtime
export BARREN_LIMIT=3        # the real default: a finished run must not consume it
echo "-- part 2: the decision loop"

# The bug this file was written for. A finished run that outlives the settle window must
# still be recognised, or the supervisor relaunches into it forever at slice prices -- and
# every relaunch also feeds the barren counter, which after 3 deletes a perfectly good
# queued resource and re-queues it. Two individually correct mechanisms composing into an
# acquire-idle-destroy loop.
case_ "finished run, alive through the settle window" yes 0   1 3 1 one

# The regression the fix could have introduced: /tmp/fastwam/train.log is appended and survives
# a slice that is reused rather than re-provisioned, so a NEW run on a warm machine can see
# a PREVIOUS run's "done: 0 steps". It must launch anyway; only a log we wrote counts.
case_ "stale completion in a log we did not write"    yes 0   1 3 1 one

# A healthy run must never trip any of it. (This stub kills every launch instantly, so the
# barren guard SHOULD eventually delete the queued resource -- that is the guard working,
# which is why this case does not forbid it.)
case_ "healthy run, never finished"                   no  124 1 999 0 any

[ "$FAILED" = 0 ] && echo "PASS" || echo "FAILURES"
exit "$FAILED"
