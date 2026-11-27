"""Config tests: the ported values match the official yamls, and the loader is strict.

The value assertions below are the actual audit trail for "did we port the recipe
correctly" — each one is quoted from reference/FastWAM/configs/.
"""

import copy

import pytest
import yaml

from fastwam_tpu import config as cfgmod
from fastwam_tpu.config import ConfigError, build_config, load_config, load_raw

TASK = "training/full.example.yaml"


@pytest.fixture(scope="module")
def raw() -> dict:
    return load_raw(TASK)


@pytest.fixture(scope="module")
def cfg():
    return load_config(TASK)


# --------------------------------------------------------------------------- #
# shipped yaml files
# --------------------------------------------------------------------------- #
def test_all_composed_config_files_parse():
    configs_dir = cfgmod.default_configs_dir()
    paths = sorted(configs_dir.rglob("*.example.yaml"))
    for path in paths:
        assert path.is_file(), path
        with path.open() as fh:
            loaded = yaml.safe_load(fh)
        assert isinstance(loaded, dict) and loaded, path


def test_load_config_succeeds(cfg):
    assert cfg.model.model_id == "Wan-AI/Wan2.2-TI2V-5B"


# --------------------------------------------------------------------------- #
# ported values: fixed files plus the selected full config files
# --------------------------------------------------------------------------- #
def test_training_recipe(cfg):
    assert cfg.lr_scheduler_type == "cosine"
    assert cfg.learning_rate == pytest.approx(1e-4)
    assert cfg.num_epochs == 5
    assert cfg.max_steps is None
    assert cfg.log_every == 10
    assert cfg.save_every == 2500
    assert cfg.eval_every == 500
    assert cfg.eval_num_inference_steps == 10
    assert cfg.gradient_accumulation_steps == 1
    assert cfg.mixed_precision == "bf16"
    assert cfg.seed == 42
    assert cfg.max_grad_norm == pytest.approx(1.0)
    assert cfg.weight_decay == pytest.approx(1e-2)
    assert cfg.resume is None


def test_floats_are_floats(cfg):
    """PyYAML parses `1e-4` as a string; every float below must really be a float."""
    for value in (cfg.learning_rate, cfg.weight_decay, cfg.max_grad_norm,
                  cfg.model.video_dit_config.eps, cfg.model.action_dit_config.eps,
                  cfg.model.video_scheduler.train_shift):
        assert isinstance(value, float)


def test_optimizer_recipe(cfg):
    # trainer.py:89-94 and :204-253 — hardcoded upstream, explicit here.
    assert cfg.optimizer.name == "adamw"
    assert (cfg.optimizer.beta1, cfg.optimizer.beta2) == (0.9, 0.95)
    assert cfg.optimizer.eps == pytest.approx(1e-8)
    assert cfg.optimizer.warmup_ratio == pytest.approx(0.05)
    assert cfg.optimizer.min_lr_ratio == pytest.approx(0.01)
    assert cfg.optimizer.use_ema is False


# --------------------------------------------------------------------------- #
# ported values: internal specs/fastwam.yaml
# --------------------------------------------------------------------------- #
def test_video_expert(cfg):
    v = cfg.model.video_dit_config
    assert v.patch_size == (1, 2, 2)
    assert (v.in_dim, v.out_dim) == (48, 48)
    assert (v.hidden_dim, v.ffn_dim) == (3072, 14336)
    assert (v.num_heads, v.attn_head_dim, v.num_layers) == (24, 128, 30)
    assert v.attn_inner_dim == 3072
    assert (v.freq_dim, v.text_dim) == (256, 4096)
    assert v.eps == pytest.approx(1e-6)
    assert v.seperated_timestep is True
    assert v.fuse_vae_embedding_in_latents is True
    assert v.video_attention_mask_mode == "first_frame_causal"
    assert v.action_conditioned is False


def test_action_expert(cfg):
    a = cfg.model.action_dit_config
    assert (a.hidden_dim, a.ffn_dim) == (1024, 4096)
    assert (a.num_heads, a.attn_head_dim, a.num_layers) == (24, 128, 30)
    # The whole point of the action expert: narrow residual stream, full attention width.
    assert a.attn_inner_dim == 3072 != a.hidden_dim
    assert a.action_dim == 14


def test_schedulers_and_loss(cfg):
    for sched in (cfg.model.video_scheduler, cfg.model.action_scheduler):
        assert sched.train_shift == pytest.approx(5.0)
        assert sched.infer_shift == pytest.approx(5.0)
        assert sched.num_train_timesteps == 1000
    assert cfg.model.loss.lambda_video == pytest.approx(1.0)
    assert cfg.model.loss.lambda_action == pytest.approx(1.0)


def test_gradient_checkpointing_is_derived(cfg, raw):
    # task overrides mot_checkpoint_mixed_attn to false -> both experts follow.
    assert cfg.model.mot_checkpoint_mixed_attn is False
    assert cfg.model.video_dit_config.use_gradient_checkpointing is False
    assert cfg.model.action_dit_config.use_gradient_checkpointing is False

    flipped = copy.deepcopy(raw)
    flipped["model"]["mot_checkpoint_mixed_attn"] = True
    other = build_config(flipped)
    assert other.model.video_dit_config.use_gradient_checkpointing is True
    assert other.model.action_dit_config.use_gradient_checkpointing is True


# --------------------------------------------------------------------------- #
# ported values: internal specs/robotwin.yaml
# --------------------------------------------------------------------------- #
def test_robotwin_data(cfg):
    d = cfg.data.train
    assert [img.key for img in d.shape_meta.images] == ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    assert all(img.raw_shape == (3, 480, 640) and img.shape == (3, 240, 320) for img in d.shape_meta.images)
    assert d.num_frames == 33
    assert d.action_video_freq_ratio == 4
    assert d.video_size == (384, 320)
    assert d.global_sample_stride == 1
    assert d.val_set_proportion == pytest.approx(0.01)
    assert d.concat_multi_camera == "robotwin"
    assert d.context_len == 128
    assert d.skip_padding_as_possible is False

    p = d.processor
    assert p.num_output_cameras == 3
    assert p.action_output_dim == 14 and p.proprio_output_dim == 14
    assert p.norm_default_mode == "z-score"
    assert p.use_stepwise_action_norm is False
    assert p.action_state_merger == "ConcatLeftAlign"
    assert p.train_transforms.resize == (240, 320)
    assert p.val_transforms.resize == (240, 320)


def test_proprio_dim_matches_data(cfg):
    assert cfg.model.proprio_dim == cfg.data.train.processor.proprio_output_dim == 14


# --------------------------------------------------------------------------- #
# derived geometry (spec §4.5)
# --------------------------------------------------------------------------- #
def test_derived_shapes(cfg):
    s = cfg.shapes
    assert s.video_frames == 9
    assert s.action_horizon == 32
    assert s.latent_shape == (48, 3, 24, 20)
    assert s.tokens_per_frame == 120
    assert s.video_tokens == 360
    assert s.action_tokens == 32
    assert s.joint_seq == 392
    assert s.prefill_video_tokens == 120
    assert s.prefill_joint_seq == 152
    assert s.context_tokens == 129


# --------------------------------------------------------------------------- #
# the loader is strict: every one of these must raise, never warn
# --------------------------------------------------------------------------- #
def test_unknown_key_raises(raw):
    bad = copy.deepcopy(raw)
    bad["model"]["video_dit_config"]["hiden_dim"] = 3072
    with pytest.raises(ConfigError, match="unknown key"):
        build_config(bad)


def test_missing_key_raises(raw):
    bad = copy.deepcopy(raw)
    del bad["model"]["video_dit_config"]["num_heads"]
    with pytest.raises(ConfigError, match="missing key"):
        build_config(bad)


def test_derived_key_in_yaml_raises(raw):
    bad = copy.deepcopy(raw)
    bad["model"]["video_dit_config"]["use_gradient_checkpointing"] = True
    with pytest.raises(ConfigError, match="derived"):
        build_config(bad)


def test_string_float_raises(raw):
    """`learning_rate: 1e-4` in yaml is the string '1e-4' — catch it, don't coerce."""
    bad = copy.deepcopy(raw)
    bad["learning_rate"] = "1e-4"
    with pytest.raises(ConfigError, match="expected float"):
        build_config(bad)


def test_wrong_type_raises(raw):
    bad = copy.deepcopy(raw)
    bad["num_epochs"] = "5"
    with pytest.raises(ConfigError, match="expected int"):
        build_config(bad)


def test_training_bucket_is_selected_by_the_operator(raw):
    bad = copy.deepcopy(raw)
    bad["tpu"]["gcs"]["jax_cache_dir"] = "gs://another-bucket/jax-cache"
    with pytest.raises(ConfigError, match="outside gs://"):
        build_config(bad)


def test_region_bucket_is_derived_from_the_output_path(raw):
    """All configured GCS reads and writes must use the explicitly selected run bucket."""
    bad = copy.deepcopy(raw)
    bad["tpu"]["gcs"]["region_bucket"] = "wrong-bucket"
    for key, path in bad["tpu"]["gcs"].items():
        if key != "region_bucket":
            bad["tpu"]["gcs"][key] = path.replace("gs://your-bucket/", "gs://another-bucket/")
    with pytest.raises(ConfigError, match="outside gs://"):
        build_config(bad)


def test_operator_selected_bucket_is_preserved_for_any_zone(raw):
    bad = copy.deepcopy(raw)
    bad["tpu"]["zone"] = "us-west1-c"
    assert build_config(bad).tpu.zone == "us-west1-c"


def test_region_is_not_bound_to_one_bucket_name(raw):
    """The output prefix selects the bucket; every input must resolve to that bucket."""
    good = copy.deepcopy(raw)
    good["tpu"]["zone"] = "us-east1-d"
    good["tpu"]["accelerator"] = "v6e-64"
    old_bucket = good["tpu"]["gcs"]["region_bucket"]
    good["tpu"]["gcs"]["region_bucket"] = "selected-bucket"
    for key, path in good["tpu"]["gcs"].items():
        if key != "region_bucket":
            good["tpu"]["gcs"][key] = path.replace(f"gs://{old_bucket}/", "gs://selected-bucket/")
    for key in ("video_dit_orbax", "vae_orbax", "action_dit_init"):
        path = good["model"]["weights"][key]
        good["model"]["weights"][key] = path.replace(f"gs://{old_bucket}/", "gs://selected-bucket/")
    for key in ("prefix", "stats_path", "text_dir"):
        path = good["data"]["l3"][key]
        if path is not None and path.startswith("gs://"):
            good["data"]["l3"][key] = path.replace(f"gs://{old_bucket}/", "gs://selected-bucket/")
    cfg = build_config(good)
    assert cfg.tpu.gcs.region_bucket == "selected-bucket"
    assert cfg.tpu.gcs.run_dir.startswith("gs://selected-bucket/")


def test_train_val_drift_raises(raw):
    bad = copy.deepcopy(raw)
    bad["data"]["val"]["val_set_proportion"] = 0.05
    with pytest.raises(ConfigError, match="differ on"):
        build_config(bad)


def test_expert_mismatch_raises(raw):
    bad = copy.deepcopy(raw)
    bad["model"]["action_dit_config"]["num_layers"] = 28
    with pytest.raises(ConfigError, match="must agree on num_layers"):
        build_config(bad)


def test_action_dim_mismatch_raises(raw):
    bad = copy.deepcopy(raw)
    bad["model"]["action_dit_config"]["action_dim"] = 7
    with pytest.raises(ConfigError, match="action_dim"):
        build_config(bad)


def test_unsupported_flag_raises(raw):
    bad = copy.deepcopy(raw)
    bad["model"]["video_dit_config"]["action_conditioned"] = True
    with pytest.raises(ConfigError, match="action_conditioned"):
        build_config(bad)


def test_bad_geometry_raises(raw):
    bad = copy.deepcopy(raw)
    bad["data"]["train"]["video_size"] = [385, 320]
    bad["data"]["val"]["video_size"] = [385, 320]
    with pytest.raises(ConfigError, match="must divide by 16"):
        build_config(bad)


def test_canvas_must_match_the_declared_video_size(raw):
    """M4 replaced the `materialized` latent block with `l3`, so the geometry
    cross-check moved: the config's video_size now has to be the canvas the loader
    actually composes, not a latent shape written out by hand."""
    bad = copy.deepcopy(raw)
    for node in ("train", "val"):
        bad["data"][node]["video_size"] = [320, 512]
    with pytest.raises(ConfigError, match="does not match the ported RoboTwin canvas"):
        build_config(bad)


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("source_color_order", "rgb_baked", "source_color_order"),
        ("text_mode", "in_loop", "text_mode"),
        ("episode_weighting", "frames", "episode_weighting"),
        ("action_shift", 2, "action_shift"),
        ("pool_rate", 8192.0, "pool_rate"),
    ],
)
def test_l3_enums_are_checked(raw, key, value, message):
    bad = copy.deepcopy(raw)
    bad["data"]["l3"][key] = value
    with pytest.raises(ConfigError, match=message):
        build_config(bad)


def test_l3_paths_must_stay_in_region(raw):
    bad = copy.deepcopy(raw)
    bad["data"]["l3"]["prefix"] = "gs://another-bucket/data/train/pixel_v3"
    with pytest.raises(ConfigError, match="outside gs://"):
        build_config(bad)


def test_override_key_must_exist_in_base():
    with pytest.raises(ConfigError, match="not present in the base config"):
        cfgmod._deep_override({"a": 1}, {"b": 2})


def test_missing_task_file_raises():
    with pytest.raises(ConfigError, match="not found"):
        load_config("training/no-such.yaml")


# --------------------------------------------------------------------------- #
# the small setting: one self-contained recipe
# --------------------------------------------------------------------------- #
SMALL = "training/small.example.yaml"


@pytest.fixture(scope="module")
def small():
    return load_config(SMALL)


def test_task_filter_defaults_to_the_whole_set(cfg):
    """Every existing checkpoint and every published number was produced with no filter,
    so the default has to stay `null` in the base data config."""
    assert cfg.data.l3.task_filter is None


def test_task_filter_rejects_a_list_that_would_lie_about_its_size(raw):
    empty = copy.deepcopy(raw)
    empty["data"]["l3"]["task_filter"] = []
    with pytest.raises(ConfigError, match="empty list"):
        build_config(empty)

    dupe = copy.deepcopy(raw)
    dupe["data"]["l3"]["task_filter"] = ["adjust_bottle", "click_bell", "adjust_bottle"]
    with pytest.raises(ConfigError, match="repeats"):
        build_config(dupe)

    padded = copy.deepcopy(raw)
    padded["data"]["l3"]["task_filter"] = ["adjust_bottle "]
    with pytest.raises(ConfigError, match="space-padded"):
        build_config(padded)


def test_the_small_setting_geometry(small):
    """The run geometry the small setting exists to pin. Each of these is a line in the
    task file's DEVIATION list; a change to one without the other is the bug."""
    assert small.tpu.global_batch_size == 256
    assert small.tpu.micro_batch_size == 256
    assert small.tpu.grad_accum == 1  # micro == global, no accumulation
    assert small.tpu.accelerator == "v6e-16"
    # 16 chips, 4 hosts: the batch has to divide over both or train.py refuses at startup.
    assert small.tpu.global_batch_size % 16 == 0 and small.tpu.global_batch_size % 4 == 0
    assert small.tpu.global_batch_size // 16 == 16  # samples/chip, the measured geometry
    assert small.learning_rate == pytest.approx(1e-4)
    assert small.weight_decay == pytest.approx(1e-2)
    assert small.lr_scheduler_type == "cosine"


def test_the_small_setting_anneals_the_whole_schedule(small):
    """steps == decay_steps, deliberately against train.yaml's rule for smokes: 30,000 is
    the whole run, so the cosine must reach the floor rather than stop partway."""
    assert small.tpu.steps == 30000
    assert small.tpu.decay_steps == 30000
    assert small.tpu.steps == small.tpu.decay_steps


def test_the_small_setting_checkpoint_cadence(small):
    assert small.tpu.ckpt.full_every == 200      # ~5 min of replay on a spot slice
    assert small.tpu.ckpt.bf16_every == 2500     # the official artifact cadence, unchanged
    assert small.save_every == 2500


def test_the_small_setting_stays_in_us_east1(small):
    """Every GCS path in the example uses the selected output bucket."""
    assert small.tpu.zone == "us-east1-d"
    assert small.tpu.gcs.region_bucket == "your-bucket"
    for path in small.tpu.gcs.paths().values():
        assert path.startswith("gs://your-bucket/"), path
    for name in ("prefix", "stats_path", "text_dir"):
        assert getattr(small.data.l3, name).startswith("gs://your-bucket/")


def test_the_small_setting_trains_the_tasks_the_fast_eval_scores(small):
    """One source of truth for "which ten tasks". The training filter and the eval task
    list are two files that must agree; if they drift, a small run is scored on tasks it
    never trained on and the gap looks like a model problem."""
    t10 = cfgmod.default_configs_dir().parent / "eval/scripts/fastwam_eval/tasks/t10.txt"
    names = [line.strip() for line in t10.read_text().splitlines() if line.strip()]
    assert len(names) == 10
    assert list(small.data.l3.task_filter) == names


def test_the_small_setting_deviates_from_the_big_recipe_only_where_it_says(small):
    """Everything not on the DEVIATION list must be the v6e-64 B-line recipe verbatim.
    The task loader cannot inherit one task file from another, so these values are copies
    — and a copy is exactly the thing that drifts."""
    big = load_config(TASK)
    declared = {
        "tpu.accelerator", "tpu.global_batch_size", "tpu.micro_batch_size",
        "tpu.steps", "tpu.decay_steps", "tpu.ckpt.full_every", "tpu.ckpt.bf16_every",
        "data.l3.task_filter",
    }
    same = [
        "learning_rate", "weight_decay", "num_epochs",
        "seed", "max_grad_norm", "mixed_precision", "save_every", "eval_every",
        "log_every", "gradient_accumulation_steps", "lr_scheduler_type",
        "optimizer.warmup_ratio", "optimizer.min_lr_ratio", "optimizer.beta1",
        "optimizer.beta2", "tpu.zone", "tpu.remat_policy", "tpu.scan_layers",
        "tpu.param_dtype", "tpu.compute_dtype", "tpu.param_axis_policy",
        "tpu.shard_params", "tpu.shard_activations", "tpu.sharding_constraints",
        "tpu.fused_accum", "tpu.vae_encode_splits", "tpu.ckpt.keep_last",
        "tpu.gcs.region_bucket", "tpu.gcs.run_dir",
        "data.l3.prefix", "data.l3.text_dir", "data.l3.text_mode",
        "data.l3.stats_path", "data.l3.action_shift", "data.l3.source_color_order",
        "data.l3.episode_weighting", "data.l3.pool_size", "data.l3.pool_rate",
        "model.weights.video_dit_orbax", "model.weights.vae_orbax",
        "model.mot_checkpoint_mixed_attn",
    ]
    assert not declared & set(same)
    drifted = {
        path: (cfgmod._getattr_path(small, path), cfgmod._getattr_path(big, path))
        for path in same
        if cfgmod._getattr_path(small, path) != cfgmod._getattr_path(big, path)
    }
    assert not drifted, f"undeclared divergence from the B-line recipe: {drifted}"
