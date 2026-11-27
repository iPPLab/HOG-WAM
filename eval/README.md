# Simulation evaluation

This directory contains the RoboTwin 2.0 adapter, closed-loop launcher and
simulator setup. It does not contain model weights, episode data, SAPIEN
installations or evaluation results. For LIBERO Plus, see the
[main reproduction guide](../README.md#libero-plus).

## RoboTwin observation and action contract

The policy receives a uint8 RGB canvas of shape [384, 320, 3]. The 320 x 256
head-camera image fills the top. The 160 x 128 left and right wrist images fill
the lower half. Input channels are RGB and are not flipped. State is a 14-D
current joint-position vector. Inference returns 32 absolute 14-D joint targets;
the benchmark executes the task configuration's standard 24-step replanning
horizon.

The exported HOG-WAM policy does not require HOG at inference. It loads RGB,
Action and proprioception parameters from the BF16 export; the removable HOG
parameters are omitted.

## Install the simulator

Run the pinned simulator setup in the repository root:

~~~bash
bash eval/setup.sh all
~~~

This prepares the simulator environment, renderer dependencies, cuRobo build
and background textures needed by randomized scenes. The simulator environment
is separate from the JAX environment. The default simulator requirements
use PyTorch 2.4.1 and target GPUs supported by that build, such as Ampere.
On RTX 5090/Blackwell, use a Blackwell-compatible PyTorch/CUDA build and rebuild
cuRobo for that GPU. Use OIDN 2.3.3 isolated from SAPIEN's bundled OIDN 2.0.1;
the runtime is available from
[Intel's OIDN v2.3.3 release](https://github.com/RenderKit/oidn/releases/tag/v2.3.3)
and should be installed only in the simulator environment. Set
`FW_SIM_PYTHON` to an alternative simulator interpreter or a wrapper that sets
its library paths if needed. Inspect the loaded library path and run a render
smoke test before a formal campaign.

## Run a condition

Set the matching training recipe and a complete parameter-only export. Clean
and randomized use the native simulator scene configs:

~~~bash
export FW_TRAINING_CONFIG=$PWD/configs/training/robotwin-hog.example.yaml
export FW_CKPT=$PWD/outputs/runs/robotwin-hog/params_bf16_step0029365
export FW_VAE=$PWD/checkpoints/wan/vae
export FW_STATS=$PWD/data/robotwin/dataset_stats.json
export FW_TEXT_CACHE=$PWD/data/robotwin/eval_text_cache

bash eval/run_robotwin.sh clean hog-clean
~~~

For RoboTwin-Plus, set the image corruption mode before the encoder:

~~~bash
bash eval/run_robotwin.sh mixed hog-plus-mixed
bash eval/run_robotwin.sh warm hog-plus-warm
bash eval/run_robotwin.sh underexposure hog-plus-underexposure
bash eval/run_robotwin.sh contrast hog-plus-contrast
~~~

Each campaign uses 50 tasks and 100 valid episodes per task. By default the
wrapper starts one inference server and one simulator worker on GPU 0. Additional
workers and device ids can be supplied after the campaign tag:

~~~bash
bash eval/run_robotwin.sh randomized hog-random 4 0,1,2,3
~~~

The text service, inference servers, and simulator workers share host RAM.
Each listed GPU gets one inference server; simulator workers are distributed
across the list. The defaults are TCP port 6135 for the shared CPU text service
and ports starting at 6033 for model servers.

Each worker writes atomic per-episode JSON under
outputs/eval/<campaign>/results/<campaign>/<task>/<seed>.json. A renderer error
or exhausted infrastructure retry leaves the trial absent or marked giveup; it
is never counted as a model failure. Use a unique tag for each model and
condition. A campaign manifest locks the tag to its condition and checkpoint.

## Monitor and stop

The wrapper launches background processes and returns once they are ready. Use
the campaign root and tag to inspect progress:

~~~bash
tag=hog-clean
root="$PWD/outputs/eval/$tag"
FW_ROOT="$root" bash eval/scripts/fastwam_eval/status.sh "$tag"
~~~

For the default one-worker, one-GPU launch, create the two stop files below
after the rollout is complete. To interrupt a running campaign, request the
worker stop first, wait for it to exit, then request the server stop. For a
multi-worker campaign, create one worker marker per worker ID (starting at 1)
and one server marker per model-server port. The CPU text service may be kept
running for the next campaign.

~~~bash
touch "$root/stop-1"
# Once the worker has exited:
touch "$root/stop-server-6033"
~~~

To resume a campaign, use its original tag and inputs. After resolving an
infrastructure error, remove only that trial's `.giveup` marker so it can retry.

## Report

Aggregate complete RoboTwin-Plus conditions:

~~~bash
uv run python tools/report_simulation.py robotwin \
  outputs/eval/hog-plus-mixed/results/hog-plus-mixed \
  outputs/eval/hog-plus-warm/results/hog-plus-warm \
  outputs/eval/hog-plus-underexposure/results/hog-plus-underexposure \
  outputs/eval/hog-plus-contrast/results/hog-plus-contrast
~~~

The report validates all 50 task denominators, 100 episodes per task, unique
task/seed pairs and recorded scene seeds before computing the four-condition
mean.

## Seeds and result manifests

Without an explicit seed list, the evaluator follows RoboTwin's native scene
seed progression and expert-solvability check. The paper compares the same
task populations and episode budgets without requiring paired scene seeds.
Keep the task config, seed
policy, checkpoint step, renderer version, GPU assignment and per-episode
results together. Do not combine data from a smoke run with formal results.
