#!/bin/bash
# VM startup-script for the spot slices (v6e-64 B line, v6e-16 small setting). Attached as
# QR metadata by `babysit.sh` when it (re)creates the queued resource. Nothing in it is
# slice-specific: the bucket and the package hash arrive as instance metadata.
#
# It installs the ENVIRONMENT and stops. It deliberately does NOT launch training.
#
# The temptation after a preemption is to have all the hosts self-start the run and take
# the workstation out of the loop entirely. That is how you get two jax.distributed jobs
# on one pod: the babysitter cannot see a launch it did not make, decides the pod is idle,
# and launches its own. Two coordinators on one pod is the "different incarnation" collision
# in tpu_ops_pitfalls §1, and it takes down both. Exactly one thing decides when training
# starts, and that thing is the babysitter.
#
# What this buys: a re-provisioned VM arrives with the code unpacked and `uv sync` already
# done, so the babysitter's recovery is one launch instead of a full sync (~3 min saved
# per preemption, which on a spot slice is the difference that matters).
set -uo pipefail
exec > /var/log/fastwam_startup.log 2>&1
echo "=== fastwam startup $(date -u) ==="

meta() {
    curl -sf -H "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

HASH=$(meta fastwam-hash)
BUCKET=$(meta fastwam-bucket)
USER_NAME=$(meta fastwam-user)
CODE_PREFIX=$(meta fastwam-code-prefix)
[ -n "$HASH" ] || { echo "FATAL: no fastwam-hash in metadata"; exit 1; }
[ -n "$BUCKET" ] || { echo "FATAL: no fastwam-bucket in metadata"; exit 1; }
[ -n "$USER_NAME" ] || { echo "FATAL: no fastwam-user in metadata"; exit 1; }
[ -n "$CODE_PREFIX" ] || { echo "FATAL: no fastwam-code-prefix in metadata"; exit 1; }
echo "hash=$HASH bucket=$BUCKET code_prefix=$CODE_PREFIX user=$USER_NAME"

# The home directory does not exist yet when this runs. A startup-script fires during
# boot, but the user's home on these images is created lazily by pam_mkhomedir on the first
# login -- and nobody has logged in yet. The original one-shot check therefore FATALed on
# every single fresh VM, silently leaving the slice with no code at all. Measured cost:
# 11 hours and 53 launches into an empty machine.
#
# Wait for the account to exist, then create the home ourselves if login has not yet.
HOME_DIR=""
for _ in $(seq 1 60); do
    HOME_DIR=$(getent passwd "$USER_NAME" | cut -d: -f6)
    [ -n "$HOME_DIR" ] && break
    sleep 2
done
[ -n "$HOME_DIR" ] || { echo "FATAL: user $USER_NAME does not exist after 120s"; exit 1; }
if [ ! -d "$HOME_DIR" ]; then
    echo "-- $HOME_DIR does not exist yet; creating it (nobody has logged in)"
    install -d -m 0755 -o "$USER_NAME" -g "$(id -gn "$USER_NAME" 2>/dev/null || echo "$USER_NAME")" "$HOME_DIR"
fi
[ -d "$HOME_DIR" ] || { echo "FATAL: could not create $HOME_DIR"; exit 1; }
echo "home=$HOME_DIR"

TPU_TMP=/tmp/fastwam
install -d -m 0755 -o "$USER_NAME" \
    -g "$(id -gn "$USER_NAME" 2>/dev/null || echo "$USER_NAME")" "$TPU_TMP"

# The bootstrap itself is fetched from the same package it is about to install, so there is
# one copy of that logic. Pull it out of the tarball rather than duplicating it here.
TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])" 2>/dev/null)
case "$TOKEN" in ya29.*) ;; *) echo "FATAL: no usable metadata token"; exit 1 ;; esac

ENCODED_CODE_PREFIX="${CODE_PREFIX//\//%2F}"
OBJ="${ENCODED_CODE_PREFIX}%2Ffastwam_tpu_${HASH}.tgz"
PKG="$TPU_TMP/pkg_${HASH}.tgz"
curl -sf -H "Authorization: Bearer $TOKEN" -o "$PKG" \
    "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${OBJ}?alt=media"
GOT=$(md5sum "$PKG" 2>/dev/null | cut -c1-12)
[ "$GOT" = "$HASH" ] || { echo "FATAL: package md5 $GOT != $HASH"; exit 1; }

BOOT_EXTRACT="$TPU_TMP/startup-extract"
rm -rf "$BOOT_EXTRACT"
mkdir -p "$BOOT_EXTRACT"
# The package is built with `tar czf ... -C "$REPO" .`, so every member is stored as
# `./scripts/_bootstrap.sh`. GNU tar matches extraction arguments against the stored name
# literally, so asking for the bare `scripts/_bootstrap.sh` is "Not found in archive" --
# which is how this startup-script FATALed on every VM it ever ran on, silently, while
# looking healthy from the outside: the babysitter's sync fallback re-installed the code
# over ssh and the run started anyway, ~3 min later than it should have, on every single
# preemption. Same family as the tar-prefix incident in tpu_ops.sh's `pack`.
#
# Try the stored spelling, then a wildcard, so a future change to how the package is
# rolled cannot quietly break this again.
tar xzf "$PKG" -C "$BOOT_EXTRACT" ./scripts/_bootstrap.sh 2>/dev/null \
    || tar xzf "$PKG" -C "$BOOT_EXTRACT" --wildcards '*scripts/_bootstrap.sh' 2>/dev/null
[ -f "$BOOT_EXTRACT/scripts/_bootstrap.sh" ] || { echo "FATAL: no bootstrap in package"; exit 1; }
install -o "$USER_NAME" -m 0755 "$BOOT_EXTRACT/scripts/_bootstrap.sh" "$TPU_TMP/bootstrap.sh"

su - "$USER_NAME" -c "bash $TPU_TMP/bootstrap.sh $HASH $BUCKET $CODE_PREFIX" > "$TPU_TMP/bootstrap.log" 2>&1
chown "$USER_NAME" "$TPU_TMP/bootstrap.log"
if grep -q "BOOT_OK $HASH" "$TPU_TMP/bootstrap.log"; then
    echo "STARTUP_OK $HASH"
else
    echo "STARTUP_FAILED - see $TPU_TMP/bootstrap.log"
    tail -20 "$TPU_TMP/bootstrap.log"
fi
