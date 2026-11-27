#!/bin/bash
# Install the pinned RoboTwin simulation dependencies and assets.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
EVAL=$REPO/eval
TP=$EVAL/third_party
RT=${TW_ROBOTWIN:-$TP/RoboTwin}
VENV=${FW_SIM_VENV:-$EVAL/.venv-robotwin}
PY=$VENV/bin/python

# Pinned, not floating. The eval numbers in this repo were produced against these two
# trees; a later upstream commit is a different simulator and a different score.
ROBOTWIN_URL=${ROBOTWIN_URL:-https://github.com/RoboTwin-Platform/RoboTwin.git}
ROBOTWIN_COMMIT=${ROBOTWIN_COMMIT:-c3ddfa8b97d5519efa828b075999bd0006778e5e}
CUROBO_URL=${CUROBO_URL:-https://github.com/NVlabs/curobo.git}
CUROBO_TAG=${CUROBO_TAG:-v0.7.8}
PYVER=${FW_SIM_PYVER:-3.11}

# Full install by default, INCLUDING the 11 GB background-texture pack. `demo_randomized`
# is a real evaluation target, not an extra, and a setup that quietly cannot run it is a
# setup that lies. `--clean-only` skips the pack for someone who just wants demo_clean
# working quickly; a randomized campaign then refuses to start (run.sh checks) rather
# than failing per-episode deep inside the simulator.
WITH_TEXTURES=1
ARGS=()
for a in "$@"; do
    case "$a" in
        --clean-only)    WITH_TEXTURES=0 ;;
        --with-textures) WITH_TEXTURES=1 ;;   # now the default; kept so old notes still work
        *) ARGS+=("$a") ;;
    esac
done
STEP=${ARGS[0]:-all}

T0=$(date +%s)
head_() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()    { printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad()   { printf '  \033[31mFAIL\033[0m %s\n' "$*"; }
note()  { printf '  --   %s\n' "$*"; }
die()   { bad "$*"; exit 1; }
took()  { printf '  --   %s took %ss\n' "$1" "$(( $(date +%s) - $2 ))"; }
# Say how long a silent step will take BEFORE it goes silent. curobo compiles for ~8 min
# without printing, and the encoder download is 22 GB; both look like a hang to someone
# running this for the first time, and a killed setup is the expensive outcome.
eta()   { printf '  \033[1m..\033[0m   %s\n       (no output until it finishes -- this is normal)\n' "$*"; }

# --------------------------------------------------------------------------------- #
# 1. RoboTwin at the pinned commit, plus our delta
# --------------------------------------------------------------------------------- #
step_robotwin() {
    head_ "RoboTwin @ ${ROBOTWIN_COMMIT:0:8}"
    local t=$(date +%s)
    mkdir -p "$TP"
    if [ ! -d "$RT/.git" ]; then
        git clone "$ROBOTWIN_URL" "$RT" || die "clone failed"
    else
        ok "clone already present"
    fi
    # Detached at the pin. Re-running after someone checked out something else puts it
    # back, which is the point of pinning it at all.
    if [ "$(git -C "$RT" rev-parse HEAD)" != "$ROBOTWIN_COMMIT" ]; then
        git -C "$RT" fetch --all --quiet
        git -C "$RT" checkout --quiet --detach "$ROBOTWIN_COMMIT" || die "no commit $ROBOTWIN_COMMIT"
    fi
    ok "HEAD = $(git -C "$RT" rev-parse --short HEAD)"

    # Our delta. Applied with `git apply`, and a patch that is already applied is
    # detected by reverse-checking it rather than by a marker file, so a half-finished
    # setup and a hand-edited tree both resolve correctly.
    for p in "$EVAL"/patches/0*.patch; do
        local name; name=$(basename "$p")
        if git -C "$RT" apply --check "$p" 2>/dev/null; then
            git -C "$RT" apply "$p" || die "$name failed to apply"
            ok "applied $name"
        elif git -C "$RT" apply --reverse --check "$p" 2>/dev/null; then
            ok "$name already applied"
        else
            die "$name neither applies nor is applied -- the checkout has been edited by hand"
        fi
    done

    # The client adapter, reached the way RoboTwin reaches every policy: policy/<name>/.
    ln -sfnT "$EVAL/policies/fastwam_policy" "$RT/policy/fastwam_policy"
    ok "policy/fastwam_policy -> $(readlink "$RT/policy/fastwam_policy")"

    # Our patched client lives outside the checkout (eval/rt_client/), and
    # get_camera_config resolves task_config relative to the SCRIPT, not to cwd. This is
    # the one path that cannot be fixed by choosing a cwd.
    ln -sfnT "$RT/task_config" "$EVAL/rt_client/task_config"
    ok "rt_client/task_config -> $(readlink "$EVAL/rt_client/task_config")"
    took robotwin "$t"
}

# --------------------------------------------------------------------------------- #
# 2. the simulator venv (which is also the text-encoder venv)
# --------------------------------------------------------------------------------- #
step_venv() {
    head_ "simulator venv ($VENV, python $PYVER)"
    local t=$(date +%s)
    if [ ! -x "$PY" ]; then
        if command -v uv >/dev/null 2>&1; then
            # uv fetches a managed CPython if the host has no 3.11, which is what makes
            # this step work on a box whose system python is something else.
            uv venv --python "$PYVER" "$VENV" || die "uv venv failed"
        else
            command -v "python$PYVER" >/dev/null 2>&1 \
                || die "need uv or python$PYVER on PATH (sapien 3.0.0b1 ships no wheel for other versions)"
            "python$PYVER" -m venv "$VENV" || die "venv failed"
        fi
    fi
    ok "python $("$PY" -V 2>&1 | cut -d' ' -f2)"

    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PY" -r "$EVAL/requirements-sim.txt" || die "pip install failed"
    else
        "$PY" -m pip install --upgrade pip >/dev/null 2>&1
        "$PY" -m pip install -r "$EVAL/requirements-sim.txt" || die "pip install failed"
    fi
    ok "requirements-sim.txt installed"

    # --- the two source edits RoboTwin's own _install.sh makes to its dependencies ---
    # Both are `sed` on installed site-packages, both are idempotent (after the edit the
    # pattern no longer matches), and both are load-bearing.
    local site; site=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')

    # sapien reads URDF/SRDF with the locale default encoding and RoboTwin's assets are
    # UTF-8; the `.srdf` typo in the same block means the SRDF is silently never found.
    local urdf="$site/sapien/wrapper/urdf_loader.py"
    if [ -f "$urdf" ]; then
        sed -i -E 's/("r")(\))( as)/\1, encoding="utf-8") as/g' "$urdf"
        ok "patched sapien urdf_loader.py"
    else
        bad "no $urdf -- sapien layout changed"
    fi

    # mplib's screw planner treats "collides somewhere along the path" as a hard failure,
    # which rejects most legitimate RoboTwin approaches. Upstream RoboTwin drops the term.
    local planner="$site/mplib/planner.py"
    if [ -f "$planner" ]; then
        sed -i -E 's/(if np\.linalg\.norm\(delta_twist\) < 1e-4 )(or collide )(or not within_joint_limit:)/\1\3/g' "$planner"
        ok "patched mplib planner.py"
    else
        bad "no $planner -- mplib layout changed"
    fi
    took venv "$t"
}

# --------------------------------------------------------------------------------- #
# 3. curobo
# --------------------------------------------------------------------------------- #
# Pick the CUDA toolkit to build against, deliberately.
#
# torch's own resolution is `which nvcc`, and on any box with conda earlier in PATH that
# finds conda's toolkit rather than the system one. Here that was CUDA 11.7 against a
# cu121 torch, which is a MAJOR mismatch and a hard `RuntimeError` from
# `_check_cuda_version` -- a 20-minute build that dies at the first compile. A minor
# mismatch (12.4 toolkit, 12.1 torch) is only a warning, so matching the major is the
# whole requirement.
#
# There is no pip escape: `nvidia-cuda-nvcc-cu12` ships `ptxas` only, at every version
# from 12.1 to 12.9. The toolkit has to come from the host.
resolve_cuda_home() {
    local want; want=$("$PY" -c 'import torch; print(torch.version.cuda.split(".")[0])' 2>/dev/null)
    [ -n "$want" ] || return 1
    local cands=() c nvcc maj
    [ -n "${CUDA_HOME:-}" ] && cands+=("$CUDA_HOME")
    # /usr first: the Debian/Ubuntu `nvidia-cuda-toolkit` package puts nvcc in
    # /usr/bin and the headers in /usr/include, so /usr is the valid CUDA_HOME.
    # /usr/lib/nvidia-cuda-toolkit has the binaries but NO include/, so torch fails
    # there on the first header -- it must not win.
    cands+=(/usr)
    for c in /usr/local/cuda /usr/local/cuda-*; do [ -d "$c" ] && cands+=("$c"); done
    command -v nvcc >/dev/null 2>&1 && cands+=("$(dirname "$(dirname "$(command -v nvcc)")")")
    for c in "${cands[@]}"; do
        nvcc=$c/bin/nvcc
        [ -x "$nvcc" ] && [ -f "$c/include/cuda_runtime.h" ] || continue
        maj=$("$nvcc" --version 2>/dev/null | sed -nE 's/.*release ([0-9]+)\..*/\1/p')
        [ "$maj" = "$want" ] && { echo "$c"; return 0; }
    done
    echo "$want" >&2
    return 1
}

step_curobo() {
    head_ "curobo $CUROBO_TAG"
    local t=$(date +%s)
    local dst=$RT/envs/curobo
    [ -x "$PY" ] || die "run the venv step first"

    local cuda_home
    if cuda_home=$(resolve_cuda_home); then
        export CUDA_HOME=$cuda_home
        export PATH=$CUDA_HOME/bin:$PATH
        ok "CUDA_HOME=$CUDA_HOME ($("$CUDA_HOME/bin/nvcc" --version | sed -nE 's/.*release ([0-9.]+).*/\1/p'))"
    elif ! "$PY" -c 'import curobo' 2>/dev/null; then
        bad "no CUDA toolkit whose major version matches torch ($("$PY" -c 'import torch;print(torch.version.cuda)'))"
        echo "     curobo compiles against torch, so the majors must agree. Options:"
        echo "       * install one:  sudo apt install nvidia-cuda-toolkit   (Debian/Ubuntu)"
        echo "       * point at one: CUDA_HOME=/usr/local/cuda-12.x bash eval/setup.sh curobo"
        echo "     pip cannot supply it -- nvidia-cuda-nvcc-cu12 ships ptxas, not nvcc."
        return 1
    fi
    # It goes INSIDE the checkout because robot.py resolves the planner config from
    # CONFIGS.ROOT_PATH; moving it elsewhere means patching RoboTwin.
    if [ ! -d "$dst/.git" ]; then
        git clone --branch "$CUROBO_TAG" --depth 1 "$CUROBO_URL" "$dst" || die "curobo clone failed"
    else
        ok "curobo source already present"
    fi
    if "$PY" -c 'import curobo' 2>/dev/null; then
        ok "curobo already importable"
    else
        eta "building curobo CUDA kernels: ~8 min on a 12-core box, single-threaded, sm_86 only"
        ( cd "$dst" && \
          if command -v uv >/dev/null 2>&1; then
              uv pip install --python "$PY" -e . --no-build-isolation
          else
              "$PY" -m pip install -e . --no-build-isolation
          fi ) || die "curobo build failed -- check nvcc and the torch CUDA version"
        # Pinned by RoboTwin's _install.sh, after curobo, in this order.
        if command -v uv >/dev/null 2>&1; then
            uv pip install --python "$PY" "warp-lang==1.12.0" "setuptools==69.5.1" || die
        else
            "$PY" -m pip install "warp-lang==1.12.0" "setuptools==69.5.1" || die
        fi
        ok "curobo built"
    fi
    took curobo "$t"
}

# --------------------------------------------------------------------------------- #
# 4. assets
# --------------------------------------------------------------------------------- #
step_assets() {
    head_ "RoboTwin assets"
    local t=$(date +%s)
    [ -x "$PY" ] || die "run the venv step first"
    command -v unzip >/dev/null 2>&1 || die "need unzip on PATH"

    local want=("objects" "embodiments")
    if [ "$WITH_TEXTURES" = 1 ]; then
        want+=("background_texture")
    else
        note "--clean-only: skipping background_texture (11 GB). demo_clean will work;"
        note "  demo_randomized will not, and run.sh will refuse it rather than fail per-episode."
    fi

    local todo=()
    for a in "${want[@]}"; do
        [ -d "$RT/assets/$a" ] && ok "$a already extracted" || todo+=("$a")
    done
    if [ ${#todo[@]} -gt 0 ]; then
        # Budget the PEAK of this step only. objects+embodiments are ~5.3 G of zip that
        # is ~5.3 G extracted, and the zip is deleted as soon as it unpacks, so the high
        # water mark is ~10.6 G plus room to work. The venv is a prerequisite of this
        # step, so charging for it again refuses installs that would have fit -- the same
        # double-count that made run.sh reject a healthy GPU for its own resident server.
        local need=13
        [ "$WITH_TEXTURES" = 1 ] && need=30   # texture pack peaks at zip + extracted = 22 G
        local free; free=$(df -BG --output=avail "$RT" | tail -1 | tr -dc '0-9')
        [ "$free" -ge "$need" ] || die "only ${free}G free; this step peaks at ~$((need - 2))G (zip + extracted, zips deleted as they unpack)"
        local mins=4
        [ "$WITH_TEXTURES" = 1 ] && mins=15
        eta "downloading ${todo[*]} -- ~${mins} min. Resumable: re-run this step and \
huggingface_hub picks up where it stopped."
        note "peak disk during extraction is ~$((need - 3))G; it drops back afterwards"

        FW_PATTERNS="${todo[*]}" "$PY" - "$RT/assets" <<'PY' || die "asset download failed"
import os, sys
from huggingface_hub import snapshot_download

target = sys.argv[1]
patterns = [f"{name}.zip" for name in os.environ["FW_PATTERNS"].split()]
snapshot_download(
    repo_id="TianxingChen/RoboTwin2.0",
    allow_patterns=patterns,
    local_dir=target,
    repo_type="dataset",
)
PY
        for a in "${todo[@]}"; do
            [ -f "$RT/assets/$a.zip" ] || die "no $a.zip after download"
            ( cd "$RT/assets" && unzip -q -o "$a.zip" && rm -f "$a.zip" ) || die "unzip $a failed"
            ok "extracted $a"
        done
    fi
    # Rewrites the absolute paths baked into the embodiment configs. Cheap and it must
    # run after ANY re-extraction, so it is outside the guard above.
    ( cd "$RT" && "$PY" ./script/update_embodiment_config_path.py ) >/dev/null \
        || die "update_embodiment_config_path.py failed"
    ok "embodiment config paths rewritten"
    du -sh "$RT/assets" | sed 's/^/  --   /'
    took assets "$t"
}

# --------------------------------------------------------------------------------- #
# 5. the text encoder weights
# --------------------------------------------------------------------------------- #
step_textmodel() {
    head_ "UMT5-XXL text encoder"
    local t=$(date +%s)
    [ -x "$PY" ] || die "run the venv step first"
    eta "22 GB into the shared HF cache -- the single biggest download in this setup."
    note "Safe to interrupt: huggingface_hub writes .incomplete blobs and resumes from"
    note "  them, and a re-run against a complete cache returns in ~0.5s (measured), so"
    note "  running this step again is always cheap and never re-downloads."
    note "Skipping it is also fine -- the text sidecar fetches the same files on its"
    note "  first start. Doing it here just moves the wait out of the first episode."
    # Only the encoder and its tokenizer: the repo also carries a transformer and a VAE
    # that this path never loads, and they would roughly double the download.
    "$PY" - <<'PY' || die "text encoder download failed"
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    allow_patterns=["text_encoder/*", "tokenizer/*", "model_index.json"],
)
print(f"  --   cached at {path}")
PY
    took textmodel "$t"
}

# --------------------------------------------------------------------------------- #
# 6. check
# --------------------------------------------------------------------------------- #
step_check() {
    head_ "check"
    local failed=0
    [ -x "$PY" ] && ok "sim venv $PY" || { bad "no sim venv"; failed=1; }
    [ -d "$RT/.git" ] && ok "RoboTwin $(git -C "$RT" rev-parse --short HEAD)" || { bad "no RoboTwin"; failed=1; }
    [ -d "$RT/assets/objects" ] && ok "assets/objects" || { bad "no assets/objects"; failed=1; }
    [ -d "$RT/assets/embodiments" ] && ok "assets/embodiments" || { bad "no assets/embodiments"; failed=1; }
    [ -e "$RT/policy/fastwam_policy/deploy_policy.yml" ] && ok "policy/fastwam_policy" || { bad "policy symlink broken"; failed=1; }
    [ -e "$EVAL/rt_client/task_config/demo_clean.yml" ] && ok "rt_client/task_config" || { bad "task_config symlink broken"; failed=1; }

    if [ -x "$PY" ]; then
        # cwd matters: RoboTwin's modules resolve assets and configs relative to it.
        ( cd "$RT" && "$PY" - <<'PY'
import sys
bad = []
for mod in ("sapien", "mplib", "toppra", "curobo", "torch", "transformers",
            "trimesh", "open3d", "gymnasium", "cv2", "transforms3d"):
    try:
        __import__(mod)
    except Exception as exc:
        bad.append(f"{mod}: {exc}")
import numpy
if not numpy.__version__.startswith("1."):
    bad.append(f"numpy is {numpy.__version__}; sapien/mplib need the 1.x C ABI")
# `import curobo` is pure python and passes even when no CUDA kernel was built.
# aloha-agilex sets `planner: "curobo"`, so what has to work is the compiled half
# and the planner it feeds. torch first: the extensions link libc10.
try:
    import torch  # noqa: F811
    from curobo.curobolib import geom_cu, kinematics_fused_cu  # noqa: F401
    from curobo.wrap.reacher.motion_gen import MotionGen  # noqa: F401
except Exception as exc:
    bad.append(f"curobo CUDA kernels: {exc}")
if bad:
    print("  IMPORT FAILURES:"); [print("   ", b) for b in bad]; sys.exit(1)
print("  --   sim imports ok (numpy %s, torch %s)" % (
    numpy.__version__, __import__("torch").__version__))
PY
        ) || failed=1

        "$PY" "$EVAL/scripts/check_protocol.py" >/dev/null 2>&1 \
            && ok "wire protocol" || { bad "check_protocol.py"; failed=1; }
        "$PY" "$EVAL/scripts/check_adapter.py" >/dev/null 2>&1 \
            && ok "adapter replan loop" || { bad "check_adapter.py"; failed=1; }
    fi

    local model_python=${FW_PYTHON:-$REPO/.venv/bin/python}
    if [ -x "$model_python" ]; then
        "$model_python" -c "import jax; assert any(d.platform=='gpu' for d in jax.devices())" 2>/dev/null \
            && ok "model-server venv sees a GPU" \
            || bad "model-server environment needs CUDA JAX"
    else
        bad "no $REPO/.venv -- run: uv sync --extra cuda"
    fi

    echo
    if [ "$failed" -eq 0 ]; then
        echo "See $REPO/README.md for model assets and evaluation commands."
    else
        echo "setup INCOMPLETE -- fix the items above"; return 1
    fi
}

case "$STEP" in
    all)       step_robotwin && step_venv && step_curobo && step_assets && step_check ;;
    robotwin)  step_robotwin ;;
    venv)      step_venv ;;
    curobo)    step_curobo ;;
    assets)    step_assets ;;
    textmodel) step_textmodel ;;
    check)     step_check ;;
    *) echo "usage: $0 [--clean-only] {all|robotwin|venv|curobo|assets|textmodel|check}"
       echo "  --clean-only  skip the 11 GB background_texture pack (demo_randomized then unavailable)"
       exit 2 ;;
esac
rc=$?
printf '\ntotal %ss\n' "$(( $(date +%s) - T0 ))"
exit $rc
