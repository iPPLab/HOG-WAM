"""Data: RoboTwin L3 `.twp` pixels on GCS, streamed into official-shaped samples.

    twp       the one-episode-one-file format (vendored from wam-data, byte for byte)
    pool      which episodes are resident, and why that is a pure function of (seed, step)
    windows   which raw indices a window start implies, including the padding flags
    canvas    JPEG -> the official 384x320 three-camera board, in true RGB
    normalize the official z-score constants, copied and never recomputed
    text      where a sample's [128, 4096] conditioning comes from
    loader    assembly, the train/val split, and the host-local -> global jax.Array step

No latents are materialised: the VAE runs inside the training step, which is what the
official pipeline does (see `loader.py`'s module docstring for why the alternative is
both 782 GiB and a known way to score zero).
"""
