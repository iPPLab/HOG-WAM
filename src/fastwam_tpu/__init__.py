"""FastWAM_TPU — a TPU/JAX reproduction of FastWAM (uncond), RoboTwin first.

Reference implementation: FastWAM (PyTorch, audited at HEAD 45d8e14).

Only `config` is implemented at M0; every other module is a documented stub.
"""

from fastwam_tpu.config import (
    Config,
    ConfigError,
    DerivedShapes,
    build_config,
    load_config,
    load_raw,
)

__all__ = [
    "Config", "ConfigError", "DerivedShapes", "build_config",
    "load_config", "load_raw",
]
