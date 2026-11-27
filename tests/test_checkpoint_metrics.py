"""Loss/checkpoint commit tests that do not require a local TPU/JAX installation."""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def checkpoint_module(monkeypatch):
    class FakeStandardCheckpointer:
        def save(self, directory, _item):
            path = Path(directory)
            path.mkdir(parents=True)
            (path / "orbax-data").write_text("saved")

        def wait_until_finished(self):
            return None

    fake_jax = types.SimpleNamespace(process_count=lambda: 1, process_index=lambda: 0)
    fake_ocp = types.ModuleType("orbax.checkpoint")
    fake_ocp.StandardCheckpointer = FakeStandardCheckpointer
    fake_orbax = types.ModuleType("orbax")
    fake_orbax.checkpoint = fake_ocp
    monkeypatch.setitem(sys.modules, "jax", fake_jax)
    monkeypatch.setitem(sys.modules, "orbax", fake_orbax)
    monkeypatch.setitem(sys.modules, "orbax.checkpoint", fake_ocp)

    source = Path(__file__).parents[1] / "src/fastwam_tpu/training/checkpoint.py"
    spec = importlib.util.spec_from_file_location("checkpoint_metrics_under_test", source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_full_checkpoint_commits_loss_metrics_and_keeps_them_after_gc(
    tmp_path, checkpoint_module
):
    checkpointer = checkpoint_module.Checkpointer(str(tmp_path / "run"), keep_last=1)
    records = [
        {"step": 10, "loss": 1.25, "loss_video": 0.75, "loss_action": 0.5},
        {"step": 20, "loss": 1.0, "loss_video": 0.6, "loss_action": 0.4},
    ]

    assert checkpointer.save(20, {}, {}, metrics=records)
    marker = json.loads((tmp_path / "run/full/step_000000020/commit_success.txt").read_text())
    metrics_path = tmp_path / "run/logs/metrics/checkpoint_step_000000020.jsonl"
    assert marker["metrics"]["path"] == str(metrics_path)
    assert marker["metrics"]["records"] == 2
    assert [json.loads(line) for line in metrics_path.read_text().splitlines()] == records

    assert checkpointer.save(40, {}, {}, metrics=[])
    assert not (tmp_path / "run/full/step_000000020").exists()
    assert metrics_path.exists(), "checkpoint GC must not delete the durable loss curve"


def test_metrics_failure_leaves_full_checkpoint_uncommitted(
    tmp_path, checkpoint_module, monkeypatch
):
    checkpointer = checkpoint_module.Checkpointer(str(tmp_path / "run"))

    def fail_metrics(*_args, **_kwargs):
        raise OSError("metrics upload failed")

    monkeypatch.setattr(checkpointer, "_write_metrics", fail_metrics)
    assert not checkpointer.save(20, {}, {}, metrics=[{"step": 20, "loss": 1.0}])
    assert not (tmp_path / "run/full/step_000000020/commit_success.txt").exists()
    assert checkpointer.save_fail_count == 1
