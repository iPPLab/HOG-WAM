import pytest

from fastwam_tpu.deployment import DeploymentConfigError, shell_environment


SMALL = "training/small.example.yaml"
FULL = "training/full.example.yaml"


@pytest.mark.parametrize(("recipe", "accelerator"), [(SMALL, "v6e-16"), (FULL, "v6e-64")])
def test_one_recipe_resolves_one_deployment(recipe, accelerator):
    env = shell_environment(recipe, "trial-a")
    assert env["ACC"] == accelerator
    assert env["TPU"] == "fastwam-trial-a"
    assert "WORKERS" not in env
    assert env["TRAINING_CONFIG"].startswith("configs/training/")


def test_explicit_write_paths_drive_deployment():
    env = shell_environment(SMALL, "trial-a")
    assert env["CODE_PREFIX"].endswith("fastwam_tpu/code")
    assert env["RUN_PREFIX"].endswith("fastwam_tpu/runs")
    assert env["GCS_LOG_DIR"] == f"gs://your-bucket/{env['RUN_PREFIX']}/trial-a/logs"


def test_run_name_is_sanitized_for_the_tpu_resource():
    env = shell_environment(SMALL, "My_Trial.01")
    assert env["TPU"] == "fastwam-my-trial-01"


def test_too_long_run_name_is_rejected():
    with pytest.raises(DeploymentConfigError, match="63"):
        shell_environment(SMALL, "x" * 60)
