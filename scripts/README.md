# Training entry points

The user-facing Cloud TPU command is:

~~~bash
bash scripts/launch_training.sh configs/training/my_experiment.yaml my-run --print-plan
bash scripts/launch_training.sh configs/training/my_experiment.yaml my-run
~~~

The plan command validates the recipe before allocating a TPU. Use a persistent
Linux control host and run the actual launch in tmux or screen. The launcher
stages the reviewed source package, records the run configuration, starts the
babysitter and resumes only that run's committed checkpoint after preemption.

For local GPU training, use the command in README.md; this bypasses the
Cloud-specific allocation and staging scripts:

~~~bash
JAX_PLATFORMS=cuda,cpu PYTHONPATH=src \
  uv run --extra cuda python -m fastwam_tpu.training.train \
  configs/training/my_experiment.yaml
~~~

Helper scripts beginning with an underscore, stage_code.sh, babysit.sh and
tpu_ops.sh are implementation details of the Cloud TPU launcher, not alternate
training interfaces. Store personal recipes, run logs and checkpoints outside
version control.
