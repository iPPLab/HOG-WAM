#!/usr/bin/env bash
# Internal loader. Public callers select one self-contained training recipe.

DEPLOYMENT_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOYMENT_REPO="$(cd "$DEPLOYMENT_SCRIPTS_DIR/.." && pwd)"

: "${FASTWAM_TRAINING_CONFIG:?missing training config path}"
: "${FASTWAM_RUN_NAME:?missing run name}"

DEPLOYMENT_ENV=$(PYTHONPATH="$DEPLOYMENT_REPO/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m fastwam_tpu.deployment \
      "$FASTWAM_TRAINING_CONFIG" "$FASTWAM_RUN_NAME") \
    || { return 1 2>/dev/null || exit 1; }
eval "$DEPLOYMENT_ENV"

# Control-host paths are repository-local and never belong in YAML.
export CONTROL_LOG_DIR="$DEPLOYMENT_REPO/logs"
export LAUNCH_LOG_DIR="$DEPLOYMENT_REPO/logs/launch"
# gcloud creates the TPU-side login for the current control-host identity.
export TPU_USER="${TPU_USER:-$(id -un)}"

unset DEPLOYMENT_ENV DEPLOYMENT_SCRIPTS_DIR DEPLOYMENT_REPO
