# Training

## Compute and storage

Local GPU training uses one host. In tpu.zone: local-gpu, tpu.accelerator must specify the GPU count, for example gpu-4. On Cloud TPU, select a zone and a GCS bucket in that region for RoboTwin. The recipe derives the bucket guard from its run directory and rejects inputs from a different bucket. The bucket location must match the TPU region. Use local GPU recipes for LIBERO HDF5 training.

Place weights and datasets outside Git. Set explicit absolute paths in a copy of an example recipe; local recipes resolve relative paths from the repository. Keep the training configuration under configs/training/ so the loader validates and records it.

## Wan2.2 assets

The base model is Wan2.2-TI2V-5B. FastWAM-JAX distributes an Orbax conversion of its video DiT and VAE. The video DiT initializes the trainable RGB DiT and, in HOG-WAM, the trainable HOG DiT. The frozen VAE maps images to 48-channel latents. The text encoder is used offline and is not optimized.

Prepare the ActionDiT initialization from the video DiT:

~~~bash
uv run --extra convert python tools/convert_weights.py build-action-init \
  /path/to/video_dit /path/to/action_init \
  --config configs/training/robotwin-baseline.example.yaml
~~~

Every model pair starts from identical base weights and has separate output paths.

## Data contracts

RoboTwin uses the official RoboTwin 2.0 demonstrations. Convert the RGB, state, action and instruction trajectories into TWP3 and prepare a UMT5 text side table:

~~~bash
uv run --extra dataset python tools/prepare_robotwin.py \
  --source /data/robotwin2.0 --output /data/robotwin/twp
uv run --extra convert python tools/prepare_robotwin_text.py collect \
  --l3 /data/robotwin/twp --out /data/robotwin/text
uv run --extra convert python tools/prepare_robotwin_text.py encode \
  --dir /data/robotwin/text --device cuda --dtype float32
~~~

The release recipes use the aligned TWP3 profile: RGB channel order, state values from the packed state stream, and zero action shift. The legacy profile exists for older converted artifacts; do not mix its color/shift conventions with aligned TWP3.

LIBERO uses 2,000 demonstrations across Spatial, Object, Goal and LIBERO-10. The preparation code retains gripper transitions, computes min/max statistics from training trajectories only, and encodes the official instructions with the UMT5 prompt template. Its actions are seven-dimensional OSC-POSE targets and states are eight-dimensional end-effector/gripper vectors.

## Matched models

The baseline selects training_towers: rgb_action and lambda_hog: 0. HOG-WAM selects rgb_hog_action, predicts future RGB and HOG, and uses lambda_hog: 1. Its HOG expert starts from the same video DiT. Parameter-only BF16 exports omit the HOG expert while retaining the RGB/action policy.

The replacement ablation selects visual_input_mode: rgb_to_hog and rgb_action. It converts current RGB to HOG before the VAE and uses the ordinary visual loss to predict future HOG. This is distinct from HOG-WAM.

| Benchmark | Tau copies | Batch | Steps | Full checkpoint interval |
|---|---:|---:|---:|---:|
| RoboTwin baseline / HOG-WAM | 8 | 1024 | 29,365 | 250 optimizer steps |
| LIBERO baseline / HOG-WAM | 8 | 256 | 13,200 | 250 optimizer steps |
| LIBERO RGB-to-HOG | 8 | 256 | 13,200 | 250 optimizer steps |

The RGB-to-HOG run can use eight GPUs with microbatch 256. The baseline and
HOG-WAM runs use microbatch 32 plus gradient accumulation for the same effective
batch. Do not change the effective batch, dataset or step count for one member
of a pair and treat the comparison as matched.

## Launch

Copy a template, set absolute local paths or complete gs:// URIs, and choose an output directory:

~~~bash
cp configs/training/robotwin-hog.example.yaml configs/training/my_robotwin.yaml
# Edit paths and GPU count if they differ from the default layout.
~~~

Local GPU:

~~~bash
JAX_PLATFORMS=cuda,cpu PYTHONPATH=src \
  uv run --extra cuda python -m fastwam_tpu.training.train \
  configs/training/my_robotwin.yaml tpu.run_name=robotwin-hog
~~~

Cloud TPU: select a region-matched TPU and bucket, inspect the launch plan, and
launch from a persistent Linux controller:

~~~bash
bash scripts/launch_training.sh configs/training/my_robotwin.yaml robotwin-hog --print-plan
tmux new -s hogwam
bash scripts/launch_training.sh configs/training/my_robotwin.yaml robotwin-hog
~~~

Every run records source/configuration metadata and keeps resumable optimizer
checkpoints. A deployment export is complete only when its commit marker records
the expected final step. Resume only from the same run's full checkpoint.
