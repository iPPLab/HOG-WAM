"""Pytest configuration.

GPU PINNING (2026-08-04): once `jax[cuda12]` is installed in this venv, an unpinned test
run grabs the default CUDA device and tries to preallocate a large arena on it. On this
box GPU0 belongs to a live eval campaign, and a test suite is never a good enough reason
to take VRAM from one — the allocation merely failed and fell back that time, but the
next one might not. Tests are CPU-only unless a test explicitly asks otherwise; nothing
here needs an accelerator to be correct.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

"""Put the repo root on sys.path so tests can import `tools.*` alongside the package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
