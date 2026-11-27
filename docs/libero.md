# LIBERO and LIBERO Plus

## Training data

Download the pinned yifengzhu-hf/LIBERO-datasets revision with
tools/download_libero.py. The four suites contain 40 tasks and 2,000
demonstrations. tools/prepare_libero.py writes training-only min/max statistics,
a retained-trajectory index, and the UMT5 instruction cache.

Each policy uses raw OpenGL RGB from agentview and eye-in-hand cameras, resized
by the shared training transform and concatenated horizontally to a 224 x 448
canvas. No flip, channel swap or orientation correction is applied. Actions are
seven-dimensional OSC-POSE commands; proprioception is eight dimensions. Use
the same statistics at training and evaluation.

## Official Plus benchmark

tools/setup_libero_plus.py checks out the benchmark at revision
4976dc30028e805ff8094b55501d532c48fec182 and downloads the official assets from
the pinned Sylvest/LIBERO-plus dataset revision. Assets and simulator
configuration are isolated under the specified root.

The evaluator constructs a manifest for all 10,030 variants and runs one trial
per variant using the first official initial state. It matches initial states,
instructions and variant seeds between policies. Perturbation categories retain
the training instruction except for language variants. The policy predicts 32
commands, executes eight before replanning, and uses ten flow-matching steps.

## Run and report

Use an inference server with the model's own step-13,200 BF16 export, matching
training configuration, LIBERO statistics and prepared text cache. The server
and simulator communicate over a local Unix-domain socket. Smoke runs validate
shapes and connectivity only and remain outside formal results.

Each completed trial records a Boolean success, variant identity, suite,
category, seed, model and simulator version. Errors are stored separately and
never counted as failures. The report tool requires all 10,030 unique variants.

## RGB-to-HOG ablation

The ablation uses the same LIBERO data, action expert and policy horizon but
replaces RGB with the deterministic HOG image before the VAE. The visual DiT
predicts future HOG with the ordinary visual objective. At inference, the
rgb_to_hog input mode must remain enabled.
