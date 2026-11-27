"""Deployed inference: KV-cache prefill + 10-step action denoising.

Three modules, split along the lines the deployment actually needs:

    policy.py  the replan itself — canvas conventions, normalization, noise, and the
               call into `FastWAM.infer_action`. Pure jax + numpy; no torch, no I/O.
    text.py    UMT5-XXL and the `sha256(prompt)` disk cache. The cache half is numpy
               only, so a jax-CUDA model server can read it without torch in its venv;
               the encoder half imports torch lazily and runs in its own process.
    load.py    where the bytes live — the three Orbax subtrees, the VAE, the official
               `dataset_stats.json`, and the restore order that holds one copy of a
               12 GB tree instead of two.

Deliberately nothing is imported at package-import time: `load` pulls in orbax and
`text` can pull in torch, and neither belongs in a training process that only wanted
`fastwam_tpu.inference` on its path.
"""

__all__ = ["FastWAMPolicy", "PolicyError"]


def __getattr__(name):
    if name in __all__:
        from . import policy

        return getattr(policy, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
