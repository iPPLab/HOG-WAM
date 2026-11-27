# RoboTwin 2.0 and RoboTwin-Plus

## Train

The paper training profile uses three RGB cameras in the head/left-wrist/right-
wrist layout, a 384 x 320 RGB canvas, 14-D state and action, 33 source rows per
window, and a 32-action prediction horizon. The source collection has 27,500
demonstrations over 50 tasks: 50 clean and 500 randomized demonstrations per task.
The trajectory-level split is 27,225/275 for training/validation. The reported
checkpoint is step 29,365.

Use the official FastWAM RoboTwin normalization statistics. Copy the official
dataset_stats.json into operator-controlled storage and set the recipe path.
Do not recompute these constants from repackaged episodes; the published
statistics use a specific aggregation rule.

## Clean and randomized

Run the native demo_clean and demo_randomized configurations with 100 valid
episodes per each of 50 tasks and policy. Preserve the benchmark's solvability
checks and native task-success criterion.

## RoboTwin-Plus image conditions

Each corruption acts on the composed uint8 RGB observation immediately before
visual encoding. The task, policy, instruction and proprioception are unchanged.

| Condition | Image operation |
|---|---|
| Mixed noise | Add Gaussian noise with standard deviation 10 intensity levels, then multiplicative speckle with standard deviation 0.05; round and clip to [0, 255]. |
| Warm filter | Multiply RGB channels by (1.05, 0.955, 0.92). |
| Underexposure | Multiply RGB intensity by 2^-0.5. |
| Reduced contrast | Transform intensity I to (I - 127.5) x 0.80 + 127.5. |

RoboTwin-Plus is the unweighted mean of the four corruption-condition success
rates. Report clean and native randomized results separately.

## Evaluation checks

Use the checkpoint associated with the selected recipe, with latent compatibility
disabled. The server prints its asset paths and inferred model contract at
startup. Confirm the full 14-D qpos input and 32 x 14 action chunk. RTX 5090 and
other Blackwell evaluation requires isolated OIDN 2.3.3. Renderer and
infrastructure failures are not policy episodes.
