"""CPU-only launch and result checks for the published experiment entry points."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.report_simulation import ROBOTWIN_PLUS, robotwin_campaigns

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "linux" or not shutil.which("bash"), reason="Linux Bash launcher")
def test_robotwin_wrapper_pins_campaign_inputs(tmp_path):
    eval_dir = tmp_path / "eval"
    engine = eval_dir / "scripts/fastwam_eval/run.sh"
    engine.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "eval/run_robotwin.sh", eval_dir / "run_robotwin.sh")
    engine.write_text('printf "%s %s %s %s\n" "$TW_TASK_CONFIG" "$TW_SEEDS" "$2" "$3"\n')
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "commit_success.txt").write_text('{"step": 29365}')
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("training_towers: rgb_hog_action\n")
    stats = tmp_path / "stats.json"
    stats.write_text("{}")
    vae = tmp_path / "vae"
    vae.mkdir()
    output = tmp_path / "campaign"
    env = dict(
        os.environ, FW_CKPT=str(checkpoint), FW_ROOT=str(output), FW_VAE=str(vae),
        FW_STATS=str(stats), FW_TRAINING_CONFIG=str(recipe), FW_TEXT_CACHE=str(tmp_path / "text"),
        TW_SEEDS="2",
    )
    argv = ["bash", str(eval_dir / "run_robotwin.sh"), "mixed", "test", "4", "0,1"]
    launch = subprocess.run(argv, env=env, text=True, capture_output=True, check=True)
    assert launch.stdout.strip() == "demo_clean 100 4 0,1"
    manifest = json.loads((output / "campaign.json").read_text())
    assert manifest["checkpoint"] == str(checkpoint)
    assert manifest["condition"] == "mixed"
    subprocess.run(argv, env=env, check=True, capture_output=True)
    recipe.write_text("training_towers: rgb_action\n")
    retry = subprocess.run(argv, env=env, text=True, capture_output=True)
    assert retry.returncode != 0 and "campaign manifest mismatch" in retry.stderr
    (checkpoint / "commit_success.txt").write_text('{"step": 250}')
    bad = subprocess.run(argv, env=env, text=True, capture_output=True)
    assert bad.returncode != 0 and "step-29365" in bad.stderr


def test_robotwin_plus_aggregation_requires_all_corruptions_and_one_policy(tmp_path):
    manifests, directories = {}, []
    for condition in sorted(ROBOTWIN_PLUS):
        campaign = tmp_path / condition
        results = campaign / "results" / condition
        results.mkdir(parents=True)
        manifest = campaign / "campaign.json"
        contents = {
            "tag": condition, "condition": condition, "episodes_per_task": 100,
            "checkpoint": "/weights/hog", "recipe_sha256": "abc", "vae": "/weights/vae",
            "stats_sha256": "def", "noise_seed_base": 20260927,
        }
        manifest.write_text(json.dumps(contents))
        manifests[condition] = (manifest, contents)
        directories.append(results)
    robotwin_campaigns(directories)
    path, contents = manifests["mixed"]
    path.write_text(json.dumps({**contents, "condition": "clean"}))
    with pytest.raises(ValueError, match="requires mixed"):
        robotwin_campaigns(directories)
    path.write_text(json.dumps({**contents, "checkpoint": "/weights/baseline"}))
    with pytest.raises(ValueError, match="mismatch: checkpoint"):
        robotwin_campaigns(directories)
