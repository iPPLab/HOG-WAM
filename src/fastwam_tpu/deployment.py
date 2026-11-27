"""Validate one training recipe and emit shell-safe launch variables."""

from __future__ import annotations

import argparse
import re
import shlex
import sys
from pathlib import Path

import yaml

from .config import ConfigError, default_configs_dir, load_config, load_training_spec


class DeploymentConfigError(ValueError):
    pass


def _resource_name(run_name: str) -> str:
    if not run_name.strip():
        raise DeploymentConfigError("run name must not be empty")
    slug = re.sub(r"[^a-z0-9-]+", "-", run_name.lower()).strip("-")
    slug = re.sub(r"-+", "-", slug)
    name = f"fastwam-{slug}"
    if not slug or len(name) > 63:
        raise DeploymentConfigError(
            "run name cannot be converted to a TPU resource name of at most 63 characters"
        )
    return name


def _repo_relative(path: Path, configs_dir: Path) -> str:
    repo = configs_dir.parent.resolve()
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError as exc:
        raise DeploymentConfigError(f"config path is outside the repository: {path}") from exc


def _bucket(uri: str) -> str:
    return uri.removeprefix("gs://").split("/", 1)[0]


def _object_prefix(uri: str, expected_bucket: str) -> str:
    prefix = f"gs://{expected_bucket}/"
    if not uri.startswith(prefix):
        raise DeploymentConfigError(f"{uri} is outside gs://{expected_bucket}")
    return uri.removeprefix(prefix).rstrip("/")


def shell_environment(training_config: str | Path, run_name: str) -> dict[str, str]:
    # Building the dataclass runs type, tensor-geometry and zone/bucket checks before a
    # shell script is allowed to create or touch cloud resources.
    load_config(training_config)
    configs_dir = default_configs_dir()
    selected = load_training_spec(training_config, configs_dir)
    write = selected["paths"]["write"]
    tpu = selected["tpu"]
    bucket = _bucket(write["runs"])

    return {
        "TRAINING_CONFIG": _repo_relative(selected["config_path"], configs_dir),
        "GCS_BUCKET": bucket,
        "CODE_PREFIX": _object_prefix(write["code_packages"], bucket),
        "RUN_PREFIX": _object_prefix(write["runs"], bucket),
        "GCS_LOG_DIR": f"{write['runs'].rstrip('/')}/{run_name}/logs",
        "TPU": _resource_name(run_name),
        "ZONE": tpu["zone"],
        "ACC": tpu["accelerator"],
        "RT": tpu["runtime_version"],
    }


def emit_shell(env: dict[str, str]) -> str:
    return "\n".join(f"export {name}={shlex.quote(value)}" for name, value in env.items())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_config")
    parser.add_argument("run_name")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        print(emit_shell(shell_environment(args.training_config, args.run_name)))
    except (ConfigError, DeploymentConfigError, yaml.YAMLError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
