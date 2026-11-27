#!/bin/bash
# M5 bootstrap: fetch the code package by content hash, install the pinned env.
# Every stage asserts on its PRODUCT, not on the previous command's exit code.
#
# $2 is a bucket in the TPU region.
# $3 is the object prefix that owns the content-addressed code packages.
HASH="${1:?usage: _bootstrap.sh HASH BUCKET CODE_PREFIX}"
BUCKET="${2:?usage: _bootstrap.sh HASH BUCKET CODE_PREFIX}"
CODE_PREFIX="${3:?usage: _bootstrap.sh HASH BUCKET CODE_PREFIX}"
TPU_TMP=/tmp/fastwam
DIR="$TPU_TMP/repo"
PKG="$TPU_TMP/pkg_${HASH}.tgz"
UV_BIN="$TPU_TMP/bin/uv"
mkdir -p "$TPU_TMP/bin"

TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])" 2>/dev/null)
case "$TOKEN" in ya29.*) ;; *) echo "FATAL: no usable metadata token"; exit 1 ;; esac

ENCODED_CODE_PREFIX="${CODE_PREFIX//\//%2F}"
OBJ="${ENCODED_CODE_PREFIX}%2Ffastwam_tpu_${HASH}.tgz"
curl -sf -H "Authorization: Bearer $TOKEN" -o "$PKG" \
  "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${OBJ}?alt=media"
GOT=$(md5sum "$PKG" 2>/dev/null | cut -c1-12)
[ "$GOT" = "$HASH" ] || { echo "FATAL: package md5 $GOT != $HASH"; exit 1; }

rm -rf "$DIR"; mkdir -p "$DIR"
tar xzf "$PKG" -C "$DIR"
[ -f "$DIR/src/fastwam_tpu/training/train.py" ] || { echo "FATAL: train.py missing after extract"; exit 1; }
[ -f "$DIR/uv.lock" ] || { echo "FATAL: uv.lock missing after extract"; exit 1; }

if [ ! -x "$UV_BIN" ]; then
    echo "-- installing uv"
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="$TPU_TMP/bin" sh > "$TPU_TMP/uv-install.log" 2>&1
fi
[ -x "$UV_BIN" ] || { echo "FATAL: uv not installed"; exit 1; }

echo "-- uv sync --extra tpu"
cd "$DIR" || exit 1
"$UV_BIN" sync --extra tpu 2>&1 | tail -4
"$DIR/.venv/bin/python" - <<'PY' || { echo "FATAL: venv unusable"; exit 1; }
import importlib.metadata as m
print("VERSIONS jax", m.version("jax"), "flax", m.version("flax"),
      "orbax", m.version("orbax-checkpoint"), "libtpu", m.version("libtpu"))
PY
md5sum "$DIR/src/fastwam_tpu/training/train.py" \
       "$DIR/src/fastwam_tpu/models/fastwam.py" \
       "$DIR/src/fastwam_tpu/training/sharding.py" \
       "$DIR/src/fastwam_tpu/models/mot.py" \
       "$DIR/src/fastwam_tpu/data/loader.py" \
       "$DIR/src/fastwam_tpu/config.py" \
       "$DIR/src/fastwam_tpu/specs/fastwam.yaml" \
       "$DIR/src/fastwam_tpu/specs/robotwin.yaml" \
       "$DIR/configs/training/small.example.yaml" \
       "$DIR/configs/training/full.example.yaml"
echo "BOOT_OK $HASH"
