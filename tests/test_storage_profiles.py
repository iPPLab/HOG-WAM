import yaml

from fastwam_tpu.config import default_configs_dir, load_raw, load_training_spec


SMALL = "training/small.example.yaml"
FULL = "training/full.example.yaml"


def test_training_recipes_cover_the_paper_models():
    configs = default_configs_dir()
    files = {
        path.relative_to(configs).as_posix()
        for path in configs.rglob("*.example.yaml")
    }
    assert {"training/small.example.yaml", "training/full.example.yaml",
            "training/robotwin-baseline.example.yaml", "training/robotwin-hog.example.yaml",
            "training/libero-baseline.example.yaml", "training/libero-hog.example.yaml",
            "training/libero-rgb-to-hog.example.yaml"} <= files


def test_every_storage_location_is_explicit():
    for recipe in (SMALL, FULL):
        spec = load_training_spec(recipe)
        for group in spec["paths"].values():
            assert group
            assert all(uri.startswith("gs://") or uri.startswith("/") for uri in group.values())
        assert set(spec["paths"]["read"]) == {
            "video_dit_orbax",
            "vae_orbax",
            "action_dit_init",
            "training_data",
            "dataset_stats",
            "text_embeddings",
        }
        assert set(spec["paths"]["write"]) == {"runs", "jax_cache", "code_packages"}


def test_recipes_are_self_contained_and_select_the_expected_scale():
    small = load_raw(SMALL)
    full = load_raw(FULL)
    for recipe in (SMALL, FULL):
        raw = yaml.safe_load((default_configs_dir() / recipe).read_text())
        assert set(raw) == {"paths", "tpu", "data", "flow", "training"}

    assert len(small["data"]["l3"]["task_filter"]) == 10
    assert small["tpu"]["accelerator"] == "v6e-16"
    assert small["tpu"]["global_batch_size"] == 256
    assert full["data"]["l3"]["task_filter"] is None
    assert full["tpu"]["accelerator"] == "v6e-64"
    assert full["tpu"]["global_batch_size"] == 1024


def test_fixed_specs_are_internal_not_user_configs():
    package = default_configs_dir().parent / "src" / "fastwam_tpu" / "specs"
    assert {path.name for path in package.glob("*.yaml")} == {"fastwam.yaml", "robotwin.yaml", "libero.yaml"}
