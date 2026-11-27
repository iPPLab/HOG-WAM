# Repository-relative defaults for the RoboTwin evaluation scripts.
FW_REPO=${FW_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}

# The Python servers import the checkout's package from src/.
export FASTWAM_JAX_PATH=${FASTWAM_JAX_PATH:-$FW_REPO}

# Simulator installation root.
EV=${TW_EVAL_REPO:-$FW_REPO/eval}

# The text encoder uses the simulator environment's PyTorch installation.
PY=${FW_PYTHON:-$FW_REPO/.venv/bin/python}
SIM_PY=${FW_SIM_PYTHON:-$EV/.venv-robotwin/bin/python}
TEXT_PY=${FW_TEXT_PYTHON:-$SIM_PY}

# Patched client supports explicit scene seeds.
TW_CLIENT_PY=${TW_CLIENT_PY:-$FW_REPO/eval/rt_client/script/eval_policy_client.py}

FW_ROOT=${FW_ROOT:-$FW_REPO/outputs/eval}
RT_ROOT=${TW_ROBOTWIN:-$EV/third_party/RoboTwin}

# The model needs the tensor contract; the local recipe is enough for inference.
FW_TRAINING_CONFIG=${FW_TRAINING_CONFIG:-$FW_REPO/configs/training/robotwin-hog.example.yaml}
export FW_TRAINING_CONFIG

TW_TASK_CONFIG=${TW_TASK_CONFIG:-demo_clean}
