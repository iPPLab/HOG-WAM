"""Config loading: yaml + frozen dataclasses, with everything checked at load time.

Design rules (project constitution, see README "Silent failure is a bug"):

* No hydra. A launch selects one self-contained ``configs/training/*.yaml`` recipe.
* Every GCS object is an explicit ``gs://`` URI. Nothing derives storage paths from a
  hidden layout or from a bucket/username convention.
* Unknown key -> raise. Missing key -> raise. Wrong type -> raise. A CLI override key that
  does not already exist in the composed config -> raise.
* Every value that hydra used to `${...}`-interpolate is written out explicitly in
  the yaml, and the coupling it expressed is re-asserted here as a cross-check.
* Values derived from other values (currently only `use_gradient_checkpointing`)
  are computed here and are an error to write in yaml — one source of truth.

Everything a downstream module needs about tensor geometry comes from
:class:`DerivedShapes` (``Config.shapes``), never from a literal in that module.

Status: implemented (M0). Cross-checks are added as milestones land.
"""

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .copies import CopyLengthError, resolve_copy_lengths

__all__ = [
    "AXIS_POLICIES",
    "Config",
    "ConfigError",
    "DerivedShapes",
    "build_config",
    "load_config",
    "load_raw",
    "load_training_spec",
]

AXIS_POLICIES = ("largest", "last")


class ConfigError(ValueError):
    """Any config problem. Always raised, never logged-and-continued."""


# Wan2.2 VAE 3.8 geometry (spec §2.7): 16x spatial (2 from patchify + 8 from strided
# convs), 4x temporal with a causal first frame.
VAE_SPATIAL_FACTOR = 16
VAE_TEMPORAL_FACTOR = 4

# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImageSpec:
    key: str
    raw_shape: tuple[int, int, int]
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class FieldSpec:
    key: str
    raw_shape: int
    shape: int


@dataclass(frozen=True)
class ShapeMeta:
    images: tuple[ImageSpec, ...]
    action: tuple[FieldSpec, ...]
    state: tuple[FieldSpec, ...]


@dataclass(frozen=True)
class TransformSpec:
    """One official transform list, flattened. Official is a `_target_` list:
    [image.ToTensor, torchvision.transforms.Resize(size=resize)]."""

    to_tensor: bool
    resize: tuple[int, int]


@dataclass(frozen=True)
class ProcessorConfig:
    num_obs_steps: int
    num_output_cameras: int
    action_output_dim: int
    proprio_output_dim: int
    action_state_transforms: str | None
    use_stepwise_action_norm: bool
    norm_default_mode: str
    norm_exception_mode: str | None
    action_state_merger: str
    train_transforms: TransformSpec
    val_transforms: TransformSpec


@dataclass(frozen=True)
class DatasetNodeConfig:
    dataset_dirs: tuple[str, ...]
    shape_meta: ShapeMeta
    num_frames: int
    global_sample_stride: int
    action_video_freq_ratio: int
    video_size: tuple[int, int]
    camera_key: str | None
    val_set_proportion: float
    is_training_set: bool
    pretrained_norm_stats: str | None
    skip_padding_as_possible: bool
    concat_multi_camera: str
    processor: ProcessorConfig
    text_embedding_cache_dir: str
    context_len: int


@dataclass(frozen=True)
class L3DataConfig:
    """[FastWAM_TPU] The L3 `.twp` pixel layer the TPU reads (M4).

    Read-only: this repo writes nothing under any of these paths. See
    `specs/robotwin.yaml` for what each convention costs when it is wrong.
    """

    prefix: str
    index_fingerprint: str | None
    stats_path: str
    stats_sha256: str | None
    source_color_order: str
    action_shift: int
    text_mode: str
    text_dir: str | None
    # null = every task. A list of `task_key`s (the upstream task directory names, the one
    # natural key of the RoboTwin set) restricts the run to those tasks: this is the data
    # half of a small-scale setting. A name that matches nothing raises in the loader.
    task_filter: tuple[str, ...] | None
    episode_weighting: str
    pool_size: int
    pool_rate: float
    pool_prefetch_threads: int
    decode_threads: int


@dataclass(frozen=True)
class DataConfig:
    format: str
    train: DatasetNodeConfig
    val: DatasetNodeConfig
    l3: L3DataConfig


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class VideoDiTConfig:
    has_image_input: bool
    patch_size: tuple[int, int, int]
    in_dim: int
    hidden_dim: int
    ffn_dim: int
    freq_dim: int
    text_dim: int
    out_dim: int
    num_heads: int
    attn_head_dim: int
    num_layers: int
    eps: float
    seperated_timestep: bool
    require_clip_embedding: bool
    require_vae_embedding: bool
    fuse_vae_embedding_in_latents: bool
    video_attention_mask_mode: str
    action_conditioned: bool
    action_dim: int
    action_group_causal_mask_mode: str
    # derived from ModelConfig.mot_checkpoint_mixed_attn
    use_gradient_checkpointing: bool = field(init=False, default=False)

    _derived_fields = frozenset({"use_gradient_checkpointing"})

    @property
    def attn_inner_dim(self) -> int:
        return self.num_heads * self.attn_head_dim


@dataclass(frozen=True)
class ActionDiTConfig:
    action_dim: int
    hidden_dim: int
    ffn_dim: int
    num_heads: int
    attn_head_dim: int
    num_layers: int
    text_dim: int
    freq_dim: int
    eps: float
    use_gradient_checkpointing: bool = field(init=False, default=False)

    _derived_fields = frozenset({"use_gradient_checkpointing"})

    @property
    def attn_inner_dim(self) -> int:
        """3072 for both experts, even though hidden_dim is 1024 here (spec §2.3)."""
        return self.num_heads * self.attn_head_dim


@dataclass(frozen=True)
class SchedulerConfig:
    train_shift: float
    infer_shift: float
    num_train_timesteps: int


@dataclass(frozen=True)
class LossConfig:
    lambda_video: float
    lambda_hog: float
    lambda_action: float


@dataclass(frozen=True)
class WeightsConfig:
    """[FastWAM_TPU] Orbax trees on in-region GCS, replacing the official HF download."""

    video_dit_orbax: str
    vae_orbax: str
    action_dit_init: str


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    tokenizer_model_id: str
    tokenizer_max_len: int
    load_text_encoder: bool
    proprio_dim: int
    # Input representation for the two-tower deployment path.  ``rgb`` is the
    # released baseline; ``rgb_to_hog`` converts every RGB canvas to the TPU-native
    # HOG representation before the VAE and uses the ordinary video objective to
    # predict future HOG latents.  The auxiliary three-tower HOG recipes keep the
    # default ``rgb`` mode and are selected by their model construction path.
    visual_input_mode: str
    # Training-only HOG observation policy.  ``current`` is the historical
    # auxiliary-tower recipe; ``none`` keeps the HOG tower in the three-tower
    # forward pass but replaces its observed frame with a constant zero frame.
    # ``noise`` replaces only that frame with pure standard Gaussian noise,
    # independent of the observed HOG and of its sampled flow time.
    # Future HOG latents remain noisy inputs and their clean values remain the
    # flow-matching targets.
    hog_condition_mode: str
    training_towers: str
    # Initialization only: no change to the auxiliary HOG architecture or mask.
    hog_dit_init: str
    redirect_common_files: bool
    mot_checkpoint_mixed_attn: bool
    action_dit_pretrained_path: str | None
    skip_dit_load_from_pretrain: bool
    video_dit_config: VideoDiTConfig
    action_dit_config: ActionDiTConfig
    video_scheduler: SchedulerConfig
    action_scheduler: SchedulerConfig
    loss: LossConfig
    weights: WeightsConfig

    def __post_init__(self) -> None:
        for expert in (self.video_dit_config, self.action_dit_config):
            object.__setattr__(expert, "use_gradient_checkpointing", self.mot_checkpoint_mixed_attn)


# --------------------------------------------------------------------------- #
# training / runtime
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FlowConfig:
    """[FastWAM_TPU] How many tau samples the action objective gets per step.

    No official counterpart: the reference draws one flow time per sample per stream.
    `tau_copies > 1` packs that many independent copies of the predicted chunk into the
    same forward, and `copy_lengths` makes them different horizons. Both default to the
    single-copy behaviour, which is bit-identical to having neither. See `copies.py` for
    the layout and `DerivedShapes.copy_lengths` for the resolved tuple.
    """

    tau_copies: int
    copy_lengths: tuple[int, ...] | None


@dataclass(frozen=True)
class OptimizerConfig:
    name: str
    beta1: float
    beta2: float
    eps: float
    warmup_ratio: float
    min_lr_ratio: float
    use_ema: bool


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool
    workspace: str | None
    project: str
    name: str
    group: str | None
    mode: str


@dataclass(frozen=True)
class GcsConfig:
    region_bucket: str
    run_dir: str
    jax_cache_dir: str

    def paths(self) -> dict[str, str]:
        return {
            "run_dir": self.run_dir,
            "jax_cache_dir": self.jax_cache_dir,
        }


@dataclass(frozen=True)
class CkptRuntimeConfig:
    """[FastWAM_TPU] Cadence and retention for `training/checkpoint.py`.

    `full_every` / `bf16_every` are None by default and fall back to the official
    `save_every`, so the recipe keeps one source of truth for the cadence.
    """

    full_every: int | None
    bf16_every: int | None
    keep_last: int
    save_fail_limit: int
    early_verify_step: int


@dataclass(frozen=True)
class ProfileConfig:
    """[FastWAM_TPU] A jax.profiler trace window; chief-only, off by default."""

    enabled: bool
    dir: str
    start_step: int
    num_steps: int


@dataclass(frozen=True)
class TpuConfig:
    accelerator: str
    zone: str
    mesh_axes: tuple[str, ...]
    ici_data_parallelism: int
    ici_fsdp_parallelism: int
    dcn_data_parallelism: int
    dcn_fsdp_parallelism: int
    remat_policy: str
    scan_layers: bool
    param_dtype: str
    compute_dtype: str
    multihost: bool
    sharding_constraints: bool
    shard_params: bool
    shard_activations: bool
    param_axis_policy: str
    global_batch_size: int
    micro_batch_size: int
    steps: int | None
    decay_steps: int | None
    run_name: str | None
    resume: bool
    resume_from: str | None
    fused_accum: bool
    vae_encode_splits: int
    mfu_peak_tflops_per_chip: float | None
    ckpt: CkptRuntimeConfig
    profile: ProfileConfig
    gcs: GcsConfig

    @property
    def grad_accum(self) -> int:
        """Microsteps per optimizer step. Exact division is checked in `_cross_check`."""
        return self.global_batch_size // self.micro_batch_size


@dataclass(frozen=True)
class DerivedShapes:
    """Tensor geometry implied by the config. Numbers in the docstring are the
    RoboTwin values from spec §4.5 and exist so a mistake is visible in review.

    RoboTwin: video [3,9,384,320] -> latents [B,48,3,24,20] -> 120 tokens/frame
    -> Sv 360, Sa 32, joint 392; fast inference prefills 1 latent frame -> Sv 120.

    `action_tokens` is what the DATA carries and what inference denoises; with
    `flow.tau_copies > 1` training packs `train_action_tokens` rows instead (the copies,
    see `copies.py`). The two are deliberately separate names: reading the training
    geometry off `action_tokens` would understate the MFU denominator, and reading the
    inference geometry off the training one would size the KV-cache path for copies that
    only ever existed during training.
    """

    video_frames: int  # 9
    action_horizon: int  # 32
    latent_channels: int  # 48
    latent_frames: int  # 3
    latent_height: int  # 24
    latent_width: int  # 20
    tokens_per_frame: int  # 120
    video_tokens: int  # 360
    action_tokens: int  # 32
    joint_seq: int  # 392
    prefill_video_tokens: int  # 120
    prefill_joint_seq: int  # 152
    context_tokens: int  # 129 = 128 text + 1 proprio
    copy_lengths: tuple[int, ...]  # (32,) single copy; (32,)*4 or (32,16,8,4) with copies
    train_action_tokens: int  # 32 == sum(copy_lengths)
    train_joint_seq: int  # 392 == video_tokens + train_action_tokens

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        """(C, T_l, H_l, W_l), batch excluded."""
        return (self.latent_channels, self.latent_frames, self.latent_height, self.latent_width)


@dataclass(frozen=True)
class Config:
    lr_scheduler_type: str
    learning_rate: float
    num_epochs: int
    max_steps: int | None
    log_every: int
    save_every: int
    eval_every: int
    eval_num_inference_steps: int
    gradient_accumulation_steps: int
    mixed_precision: str
    seed: int
    max_grad_norm: float
    weight_decay: float
    resume: str | None
    output_dir: str
    optimizer: OptimizerConfig
    flow: FlowConfig
    wandb: WandbConfig
    tpu: TpuConfig
    model: ModelConfig
    data: DataConfig

    @property
    def shapes(self) -> DerivedShapes:
        return _derive_shapes(self)


# --------------------------------------------------------------------------- #
# strict builder
# --------------------------------------------------------------------------- #
def _type_name(tp: object) -> str:
    return getattr(tp, "__name__", str(tp))


def _coerce(tp: object, value: object, path: str) -> object:
    if dataclasses.is_dataclass(tp):
        return _from_mapping(tp, value, path)  # type: ignore[arg-type]

    origin = typing.get_origin(tp)
    if origin in (types.UnionType, typing.Union):
        args = typing.get_args(tp)
        if value is None:
            if type(None) in args:
                return None
            raise ConfigError(f"{path}: null is not allowed (expected {_type_name(tp)})")
        concrete = [a for a in args if a is not type(None)]
        if len(concrete) != 1:
            raise ConfigError(f"{path}: unsupported union annotation {tp!r} (internal error)")
        return _coerce(concrete[0], value, path)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {type(value).__name__}")
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(a, v, f"{path}[{i}]") for i, (a, v) in enumerate(zip(args, value)))

    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected bool, got {type(value).__name__} ({value!r})")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected int, got {type(value).__name__} ({value!r})")
        return value
    if tp is float:
        # yaml `1e-4` parses as a *string* under PyYAML's 1.1 resolver; catching that
        # here is exactly the kind of silent drift this loader exists to prevent.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"{path}: expected float, got {type(value).__name__} ({value!r}); "
                "write exponents as 1.0e-4, not 1e-4"
            )
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected str, got {type(value).__name__} ({value!r})")
        return value

    raise ConfigError(f"{path}: unsupported annotation {tp!r} (internal error)")


def _annotations(cls: type) -> dict[str, object]:
    """Resolved annotations. `dataclasses.Field.type` is a string under PEP 649
    (3.14+) and under `from __future__ import annotations`, so never read it."""
    if cls not in _ANNOTATION_CACHE:
        _ANNOTATION_CACHE[cls] = typing.get_type_hints(cls)
    return _ANNOTATION_CACHE[cls]


_ANNOTATION_CACHE: dict[type, dict[str, object]] = {}


def _from_mapping(cls: type, raw: object, path: str):
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(raw).__name__}")

    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = _annotations(cls)
    derived = frozenset(getattr(cls, "_derived_fields", frozenset()))

    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ConfigError(f"{path or '<root>'}: unknown key(s) {unknown}; known: {sorted(fields)}")
    written_derived = sorted(set(raw) & derived)
    if written_derived:
        raise ConfigError(
            f"{path or '<root>'}: {written_derived} are derived by config.py and must not "
            "appear in yaml"
        )
    missing = sorted(set(fields) - set(raw) - derived)
    if missing:
        raise ConfigError(f"{path or '<root>'}: missing key(s) {missing}")

    kwargs = {
        name: _coerce(hints[name], raw[name], f"{path}.{name}" if path else name)
        for name in fields
        if name in raw
    }
    return cls(**kwargs)


# --------------------------------------------------------------------------- #
# composition
# --------------------------------------------------------------------------- #
def default_configs_dir(start: Path | None = None) -> Path:
    """Walk up from this file until the two public training recipes turn up."""
    here = (start or Path(__file__)).resolve()
    for parent in (here, *here.parents):
        candidate = parent / "configs"
        if all(
            (candidate / "training" / name).is_file()
            for name in ("small.example.yaml", "full.example.yaml")
        ):
            return candidate
    raise ConfigError(f"no configs/training/{{small,full}}.yaml found above {here}")


def _read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a top-level mapping, got {type(raw).__name__}")
    return raw


def _deep_override(base: dict, override: dict, path: str = "") -> dict:
    """Merge `override` onto `base`. Every override key must already exist in base."""
    out = dict(base)
    for key, value in override.items():
        where = f"{path}.{key}" if path else key
        if key not in out:
            raise ConfigError(f"override key '{where}' is not present in the base config")
        if isinstance(value, dict):
            if not isinstance(out[key], dict):
                raise ConfigError(f"override '{where}': cannot merge a mapping onto a scalar")
            out[key] = _deep_override(out[key], value, where)
        else:
            out[key] = value
    return out


def _exact_keys(raw: dict, expected: set[str], where: str) -> None:
    actual = set(raw)
    if actual != expected:
        raise ConfigError(f"{where}: keys must be {sorted(expected)}, got {sorted(actual)}")


def _mapping(value: object, where: str) -> dict:
    if not isinstance(value, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(value).__name__}")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where}: expected a non-empty string")
    return value.strip()


def _gcs_uri(value: object, where: str) -> str:
    text = _text(value, where).rstrip("/")
    body = text.removeprefix("gs://")
    if not text.startswith("gs://") or "/" not in body or not all(body.split("/", 1)):
        raise ConfigError(f"{where}: expected a full gs://bucket/object URI, got {text!r}")
    if "YOUR_" in text or "SET_BY" in text or "{" in text or "}" in text:
        raise ConfigError(f"{where}: unresolved placeholder {text!r}")
    if ".." in text.removeprefix("gs://").split("/"):
        raise ConfigError(f"{where}: '..' is not allowed in a GCS URI")
    return text


def _gcs_bucket(uri: str) -> str:
    return uri.removeprefix("gs://").split("/", 1)[0]


def _source_path(ref: str | Path, configs_dir: Path, where: str) -> Path:
    path = Path(ref)
    if not path.is_absolute() and not path.is_file():
        path = configs_dir / path
    path = path.resolve()
    try:
        path.relative_to(configs_dir.resolve())
    except ValueError as exc:
        raise ConfigError(f"{where}: selected path leaves configs/: {ref}") from exc
    if path.suffix not in {".yaml", ".yml"}:
        raise ConfigError(f"{where}: selected file is not yaml: {ref}")
    if not path.is_file():
        raise ConfigError(f"{where}: config file not found: {path}")
    return path


def _local_training_path(value: object, where: str) -> str:
    text = _text(value, where)
    if not text.startswith("/") or ".." in text.split("/"):
        raise ConfigError(f"{where}: local-gpu requires an absolute local path without '..'")
    return text.rstrip("/")


def load_training_spec(training_config: str | Path, configs_dir: Path | None = None) -> dict:
    """Read one public training recipe and reject implicit or unknown configuration."""
    configs_dir = Path(configs_dir) if configs_dir is not None else default_configs_dir()
    path = _source_path(training_config, configs_dir, "training_config")
    selected = _read_yaml(path)
    recipe_keys = {"paths", "tpu", "data", "flow", "training"}
    if "visual_input_mode" in selected:
        recipe_keys.add("visual_input_mode")
    if "hog_condition_mode" in selected:
        recipe_keys.add("hog_condition_mode")
    if "training_towers" in selected:
        recipe_keys.add("training_towers")
    if "hog_dit_init" in selected:
        recipe_keys.add("hog_dit_init")
    if "loss" in selected:
        recipe_keys.add("loss")
    _exact_keys(selected, recipe_keys, "training_config")
    loss = selected.get("loss")
    if loss is not None:
        loss = _mapping(loss, "loss")
        _exact_keys(loss, {"lambda_video", "lambda_hog", "lambda_action"}, "loss")
        for name, weight in loss.items():
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0 <= weight < float("inf"):
                raise ConfigError(f"loss.{name} must be a finite nonnegative number")

    hog_condition_mode = selected.get("hog_condition_mode", "current")
    if hog_condition_mode not in {"current", "none", "noise"}:
        raise ConfigError(
            "hog_condition_mode must be 'current', 'none', or 'noise', "
            f"got {hog_condition_mode!r}"
        )

    hog_dit_init = selected.get("hog_dit_init", "copy_rgb")
    if hog_dit_init not in {"copy_rgb", "random"}:
        raise ConfigError("hog_dit_init must be 'copy_rgb' or 'random'")

    paths = _mapping(selected["paths"], "paths")
    _exact_keys(paths, {"read", "write"}, "paths")
    read_paths = _mapping(paths["read"], "paths.read")
    write_paths = _mapping(paths["write"], "paths.write")
    _exact_keys(
        read_paths,
        {
            "video_dit_orbax", "vae_orbax", "action_dit_init",
            "training_data", "dataset_stats", "text_embeddings",
        },
        "paths.read",
    )
    _exact_keys(
        write_paths,
        {"runs", "jax_cache", "code_packages"},
        "paths.write",
    )
    tpu = _mapping(selected["tpu"], "tpu")
    _exact_keys(tpu, {"zone", "accelerator", "runtime_version"}, "tpu")
    tpu = {key: _text(value, f"tpu.{key}") for key, value in tpu.items()}
    # An explicitly selected single-host GPU run uses a complete local mirror.
    # TPU recipes continue to require GCS paths and their original region checks.
    path_parser = _local_training_path if tpu["zone"] == "local-gpu" else _gcs_uri
    if tpu["zone"] == "local-gpu":
        def path_parser(value, where):
            path = Path(_text(value, where)).expanduser()
            if ".." in path.parts:
                raise ConfigError(f"{where}: parent directory segments are not allowed")
            return _local_training_path(str(path.resolve()), where)
    read_paths = {key: path_parser(value, f"paths.read.{key}") for key, value in read_paths.items()}
    write_paths = {key: path_parser(value, f"paths.write.{key}") for key, value in write_paths.items()}
    expected_bucket = None if tpu["zone"] == "local-gpu" else _gcs_bucket(write_paths["runs"])
    if expected_bucket is not None:
        for mode, group in (("read", read_paths), ("write", write_paths)):
            for name, uri in group.items():
                if _gcs_bucket(uri) != expected_bucket:
                    raise ConfigError(
                        f"paths.{mode}.{name} = {uri!r} is outside gs://{expected_bucket}; "
                        f"TPU zone {tpu['zone']} must not access GCS across regions"
                    )

    data = _mapping(selected["data"], "data")
    optional_data = {"format", "robotwin_packing"}
    _exact_keys(data, {"task_filter"} | (optional_data & data.keys()), "data")
    data = {"format": "robotwin_twp", **data}
    if data["format"] not in {"robotwin_twp", "libero_hdf5"}:
        raise ConfigError("data.format must be robotwin_twp or libero_hdf5")
    training_towers = selected.get("training_towers", "rgb_hog_action")
    if training_towers not in {"rgb_action", "rgb_hog_action"}:
        raise ConfigError("training_towers must be rgb_action or rgb_hog_action")

    flow = _mapping(selected["flow"], "flow")
    _exact_keys(flow, {"tau_copies", "copy_lengths"}, "flow")

    training = _mapping(selected["training"], "training")
    _exact_keys(
        training,
        {
            "learning_rate", "weight_decay", "num_epochs", "steps", "decay_steps",
            "global_batch_size", "micro_batch_size", "checkpoint",
        },
        "training",
    )
    checkpoint = _mapping(training["checkpoint"], "training.checkpoint")
    _exact_keys(checkpoint, {"full_every", "bf16_every"}, "training.checkpoint")

    return {
        "config_path": path,
        "paths": {"read": read_paths, "write": write_paths},
        "tpu": tpu,
        "data": data,
        "flow": flow,
        "training": training,
        "loss": loss,
        "visual_input_mode": selected.get("visual_input_mode", "rgb"),
        "hog_condition_mode": hog_condition_mode,
        "training_towers": training_towers,
        "hog_dit_init": hog_dit_init,
    }


def _fixed_training_raw() -> dict:
    """Implementation constants, deliberately not another selectable config file."""
    return {
        "lr_scheduler_type": "cosine",
        "max_steps": None,
        "log_every": 10,
        "save_every": 2500,
        "eval_every": 500,
        "eval_num_inference_steps": 10,
        "gradient_accumulation_steps": 1,
        "mixed_precision": "bf16",
        "seed": 42,
        "max_grad_norm": 1.0,
        "resume": None,
        "output_dir": "/tmp/fastwam/output",
        "optimizer": {
            "name": "adamw", "beta1": 0.9, "beta2": 0.95, "eps": 1.0e-8,
            "warmup_ratio": 0.05, "min_lr_ratio": 0.01, "use_ema": False,
        },
        "wandb": {
            "enabled": False, "workspace": None, "project": "fast-wam",
            "name": "train", "group": None, "mode": "online",
        },
        "tpu": {
            "mesh_axes": ["data", "fsdp"],
            "ici_data_parallelism": 1,
            "ici_fsdp_parallelism": -1,
            "dcn_data_parallelism": 1,
            "dcn_fsdp_parallelism": 1,
            "remat_policy": "full",
            "scan_layers": True,
            "param_dtype": "float32",
            "compute_dtype": "bfloat16",
            "multihost": True,
            "sharding_constraints": True,
            "shard_params": True,
            "shard_activations": True,
            "param_axis_policy": "largest",
            "run_name": None,
            "resume": False,
            "resume_from": None,
            "fused_accum": False,
            "vae_encode_splits": 1,
            "mfu_peak_tflops_per_chip": None,
            "ckpt": {"keep_last": 2, "save_fail_limit": 3, "early_verify_step": 0},
            "profile": {
                "enabled": False, "dir": "/tmp/fastwam/profiles", "start_step": 10,
                "num_steps": 5,
            },
        },
    }


def load_raw(training_config: str | Path, configs_dir: Path | None = None) -> dict:
    """Compose one public recipe with immutable model/data/runtime specifications."""
    configs_dir = Path(configs_dir) if configs_dir is not None else default_configs_dir()
    selected = load_training_spec(training_config, configs_dir)
    specs_dir = Path(__file__).resolve().with_name("specs")
    raw = _fixed_training_raw()
    model = _read_yaml(specs_dir / "fastwam.yaml")
    # Keep the immutable architecture spec as the source of the default while
    # allowing a recipe to opt into an explicitly different *input* representation.
    # This does not add a tower or change any DiT dimensions.
    model["visual_input_mode"] = selected["visual_input_mode"]
    model["hog_condition_mode"] = selected["hog_condition_mode"]
    model["training_towers"] = selected["training_towers"]
    model["hog_dit_init"] = selected["hog_dit_init"]
    if selected["loss"] is not None:
        model["loss"] = selected["loss"]
    data = _read_yaml(specs_dir / (
        "libero.yaml" if selected["data"]["format"] == "libero_hdf5" else "robotwin.yaml"
    ))
    data["format"] = selected["data"]["format"]
    if data["format"] == "libero_hdf5":
        model["proprio_dim"] = 8
        model["video_dit_config"]["action_dim"] = 7
        model["action_dit_config"]["action_dim"] = 7

    read = selected["paths"]["read"]
    write = selected["paths"]["write"]
    model["weights"] = {
        key: read[key]
        for key in ("video_dit_orbax", "vae_orbax", "action_dit_init")
    }
    data_l3 = _mapping(data.get("l3"), "fixed.data.l3")
    if data["format"] == "robotwin_twp":
        packing = selected["data"].get("robotwin_packing", "legacy")
        if packing == "aligned":
            data_l3.update(source_color_order="rgb", action_shift=0, index_fingerprint=None)
        elif packing != "legacy":
            raise ConfigError("data.robotwin_packing must be 'aligned' or 'legacy'")
    data_l3.update(
        {
            "prefix": read["training_data"],
            "stats_path": read["dataset_stats"],
            "text_dir": read["text_embeddings"],
            "task_filter": selected["data"]["task_filter"],
        }
    )

    training = selected["training"]
    raw.update(
        {
            "learning_rate": training["learning_rate"],
            "weight_decay": training["weight_decay"],
            "num_epochs": training["num_epochs"],
            "flow": selected["flow"],
            "model": model,
            "data": data,
        }
    )
    tpu_raw = _mapping(raw.get("tpu"), "fixed runtime tpu")
    tpu_raw.update(
        {
            "accelerator": selected["tpu"]["accelerator"],
            "zone": selected["tpu"]["zone"],
            "multihost": selected["tpu"]["zone"] != "local-gpu",
            "global_batch_size": training["global_batch_size"],
            "micro_batch_size": training["micro_batch_size"],
            "steps": training["steps"],
            "decay_steps": training["decay_steps"],
            "gcs": {
                "region_bucket": "local" if selected["tpu"]["zone"] == "local-gpu" else _gcs_bucket(write["runs"]),
                "run_dir": write["runs"],
                "jax_cache_dir": write["jax_cache"],
            },
        }
    )
    tpu_raw["ckpt"].update(training["checkpoint"])
    return raw


def build_config(raw: dict) -> Config:
    """Raw mapping -> validated Config. Separated from load_raw so tests can poke."""
    cfg: Config = _from_mapping(Config, raw, "")
    _cross_check(cfg)
    return cfg


def parse_overrides(assignments) -> dict:
    """`["tpu.steps=50", "tpu.resume=true"]` -> a nested mapping for `_deep_override`.

    Values go through the yaml scalar parser, so `true`, `null`, `[1,2]` and `1.0e-4` all
    mean what they look like. PyYAML's 1.1 resolver does NOT read `1e-4` as a float (it
    yields the string), which would then fail the type check with a confusing message, so
    that one spelling is retried as a float here — the same trap the training recipes document.

    A dotted key that does not already exist raises in `_deep_override`, so a typo on the
    command line cannot quietly create a key that nothing reads.
    """
    out: dict = {}
    for item in assignments:
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not of the form key.path=value")
        dotted, raw = item.split("=", 1)
        dotted = dotted.strip()
        if not dotted:
            raise ConfigError(f"override {item!r} has an empty key")
        value = yaml.safe_load(raw)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"override {dotted!r} descends into a scalar set by an earlier one")
        node[parts[-1]] = value
    return out


def load_config(training_config: str | Path, configs_dir: Path | None = None, overrides=None) -> Config:
    """Compose one selected recipe, apply CLI overrides, and validate."""
    raw = load_raw(training_config, configs_dir)
    if overrides:
        raw = _deep_override(raw, parse_overrides(overrides))
    return build_config(raw)


# --------------------------------------------------------------------------- #
# cross-checks: every ${...} the official configs used, plus what we don't support
# --------------------------------------------------------------------------- #
def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigError(message)


_UNSUPPORTED = (
    # (attr path, required value, why)
    ("model.video_dit_config.action_conditioned", False,
     "the action-conditioned branch is dead in every shipped config and is not ported"),
    ("model.video_dit_config.has_image_input", False, "I2V is not part of FastWAM"),
    ("model.video_dit_config.require_clip_embedding", False, "no CLIP branch"),
    ("model.video_dit_config.require_vae_embedding", False, "no separate VAE embedding input"),
    ("model.video_dit_config.seperated_timestep", True,
     "only the per-token timestep path is implemented (spec §2.2)"),
    ("model.video_dit_config.fuse_vae_embedding_in_latents", True,
     "latent frame 0 must stay clean; the KV-cache equivalence depends on it"),
    ("model.video_dit_config.video_attention_mask_mode", "first_frame_causal",
     "the other mask modes are unused upstream"),
    ("data.train.processor.use_stepwise_action_norm", False,
     "always False upstream; stepwise stats are not materialized"),
    ("model.load_text_encoder", False, "text embeddings are precomputed offline; no UMT5 on the TPU"),
)


def _getattr_path(obj: object, dotted: str) -> object:
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _cross_check(cfg: Config) -> None:
    m, d = cfg.model, cfg.data
    video, action = m.video_dit_config, m.action_dit_config

    _require(
        m.visual_input_mode in ("rgb", "rgb_to_hog"),
        "model.visual_input_mode must be 'rgb' or 'rgb_to_hog'",
    )
    _require(m.hog_condition_mode in ("current", "none", "noise"), "invalid hog_condition_mode")
    _require(m.training_towers in ("rgb_action", "rgb_hog_action"), "invalid training_towers")
    _require(m.hog_dit_init in ("copy_rgb", "random"), "invalid hog_dit_init")
    _require(
        m.hog_dit_init != "random" or (
            m.visual_input_mode == "rgb" and m.training_towers == "rgb_hog_action"
        ),
        "random HOG initialization requires the auxiliary RGB/HOG/Action architecture",
    )
    _require(d.format in ("robotwin_twp", "libero_hdf5"), "invalid data.format")
    _require(
        d.train.concat_multi_camera == ("horizontal" if d.format == "libero_hdf5" else "robotwin"),
        "concat_multi_camera must match the selected dataset profile",
    )

    # --- interpolations the official configs expressed with ${...} ---
    _require(
        m.proprio_dim == d.train.processor.proprio_output_dim,
        f"model.proprio_dim ({m.proprio_dim}) must equal "
        f"data.train.processor.proprio_output_dim ({d.train.processor.proprio_output_dim})",
    )
    for name, dit in (("video_dit_config", video), ("action_dit_config", action)):
        _require(
            dit.action_dim == d.train.processor.action_output_dim,
            f"model.{name}.action_dim ({dit.action_dim}) must equal "
            f"data.train.processor.action_output_dim ({d.train.processor.action_output_dim})",
        )
    for node_name, node in (("train", d.train), ("val", d.val)):
        _require(
            node.processor.num_obs_steps == node.num_frames,
            f"data.{node_name}.processor.num_obs_steps must equal num_frames",
        )

    # --- build-time asserts from fastwam.py:140-146 ---
    for attr in ("num_heads", "attn_head_dim", "num_layers"):
        _require(
            getattr(video, attr) == getattr(action, attr),
            f"the two experts must agree on {attr}: "
            f"{getattr(video, attr)} vs {getattr(action, attr)}",
        )
    _require(
        video.hidden_dim == video.attn_inner_dim,
        "the video expert's hidden_dim must equal num_heads*attn_head_dim",
    )

    # --- train/val drift ---
    train_fields = dataclasses.asdict(d.train)
    val_fields = dataclasses.asdict(d.val)
    drifted = sorted(
        k for k in train_fields if k != "is_training_set" and train_fields[k] != val_fields[k]
    )
    if drifted:
        raise ConfigError(
            f"data.train and data.val differ on {drifted}; upstream they are identical except "
            "is_training_set, so this is almost certainly a half-applied override"
        )
    _require(
        d.train.is_training_set and not d.val.is_training_set,
        "is_training_set must be true for train and false for val",
    )

    # --- things this repo deliberately does not implement ---
    for dotted, expected, why in _UNSUPPORTED:
        got = _getattr_path(cfg, dotted)
        _require(got == expected, f"{dotted} must be {expected!r} ({why}); got {got!r}")
    _require(
        d.train.processor.action_state_transforms is None,
        "data.train.processor.action_state_transforms must be null (RoboTwin uses none)",
    )

    # --- runtime enums ---
    _require(
        cfg.mixed_precision in ("no", "fp16", "bf16"),
        f"mixed_precision must be no|fp16|bf16, got {cfg.mixed_precision!r}",
    )
    _require(
        cfg.lr_scheduler_type == "cosine",
        f"only the cosine schedule is ported, got {cfg.lr_scheduler_type!r}",
    )
    _require(cfg.optimizer.name == "adamw", f"only adamw is ported, got {cfg.optimizer.name!r}")
    _require(cfg.gradient_accumulation_steps >= 1, "gradient_accumulation_steps must be >= 1")

    # --- [FastWAM_TPU] run geometry (M5) ---
    t = cfg.tpu
    _require(t.global_batch_size >= 1, "tpu.global_batch_size must be >= 1")
    _require(t.micro_batch_size >= 1, "tpu.micro_batch_size must be >= 1")
    _require(
        t.global_batch_size % t.micro_batch_size == 0,
        f"tpu.global_batch_size {t.global_batch_size} must divide by tpu.micro_batch_size "
        f"{t.micro_batch_size}; the quotient is the gradient-accumulation count",
    )
    # The official effective batch is 16 x 64 processes = 1024. Deviating from it is
    # a real change to the recipe, but it is sometimes the right call — a run that must fit
    # in a fixed wall clock trades batch for steps. So this is not a hard stop; instead
    # train.py prints the comparison at startup, loudly, every time. What IS a hard stop is
    # an effective batch that nothing chose: global must be an exact multiple of micro
    # (checked above), so the number in the config is always the number being trained at.
    for name in ("steps", "decay_steps"):
        value = getattr(t, name)
        _require(value is None or value >= 1, f"tpu.{name} must be >= 1 or null, got {value}")
    _require(t.vae_encode_splits >= 1, "tpu.vae_encode_splits must be >= 1")
    _require(
        t.micro_batch_size % t.vae_encode_splits == 0,
        f"tpu.micro_batch_size {t.micro_batch_size} must divide by tpu.vae_encode_splits "
        f"{t.vae_encode_splits}",
    )
    _require(
        t.param_axis_policy in AXIS_POLICIES,
        f"tpu.param_axis_policy must be one of {AXIS_POLICIES}, got {t.param_axis_policy!r}",
    )
    # `shard_activations=true` with `sharding_constraints=false` is deliberately ALLOWED,
    # and it is a no-op: the master switch turns every constraint into the identity, pins
    # included. This used to raise, which was right while shard_activations defaulted to
    # false and asking for both was a contradiction. Now that it defaults to true, raising
    # would mean the documented `sharding_constraints=false` debug bypass could not be used
    # without also remembering to turn off a second, unrelated-looking switch. The runtime
    # reports which one actually applied — `describe_mesh` prints the EFFECTIVE pin state,
    # not the requested one — so the combination cannot be silently misread.
    _require(t.ckpt.keep_last >= 1, "tpu.ckpt.keep_last must be >= 1")
    _require(t.ckpt.save_fail_limit >= 1, "tpu.ckpt.save_fail_limit must be >= 1")
    for name in ("full_every", "bf16_every"):
        value = getattr(t.ckpt, name)
        _require(value is None or value >= 1, f"tpu.ckpt.{name} must be >= 1 or null, got {value}")
    _require(
        t.scan_layers,
        "tpu.scan_layers=false is not a supported configuration: M2 measured 30 unrolled MoT "
        "layers still tracing after 15 minutes, and full-slice FSDP relies on the scan boundary "
        "to release each layer's gathered weights (unrolled, XLA holds all 30 at once).",
    )
    _require(
        cfg.resume is None,
        f"the official `resume` key ({cfg.resume!r}) points at a torch/accelerate state "
        "directory, which this repo cannot read. Use tpu.resume / tpu.resume_from.",
    )

    _check_storage(cfg)

    # --- data conventions the loader parameterises (M4) ---
    _check_data_conventions(cfg)


def _check_storage(cfg: Config) -> None:
    m, d = cfg.model, cfg.data
    if cfg.tpu.zone == "local-gpu":
        _require(cfg.tpu.accelerator.startswith("gpu-"), "local-gpu requires a gpu-* accelerator")
        _require(not cfg.tpu.multihost, "local-gpu supports one host; set tpu.multihost=false")
        for name, path in cfg.tpu.gcs.paths().items():
            _local_training_path(path, f"tpu.gcs.{name}")
        for name in ("video_dit_orbax", "vae_orbax", "action_dit_init"):
            _local_training_path(getattr(m.weights, name), f"model.weights.{name}")
        for name in ("prefix", "stats_path", "text_dir"):
            path = getattr(d.l3, name)
            if path is not None:
                _local_training_path(path, f"data.l3.{name}")
        return

    # --- region guardrail: a cross-region bucket is a silent cost/latency bug ---
    # The operator selects a bucket in the location of the TPU; require all paths to use it.
    _require(bool(cfg.tpu.gcs.region_bucket), "TPU training requires a GCS bucket")
    prefix = f"gs://{cfg.tpu.gcs.region_bucket}/"
    for name, path in cfg.tpu.gcs.paths().items():
        _require(
            path.startswith(prefix),
            f"tpu.gcs.{name} = {path!r} is outside gs://{cfg.tpu.gcs.region_bucket}; "
            f"{cfg.tpu.accelerator} in {cfg.tpu.zone} must not read across regions",
        )
    for name in ("video_dit_orbax", "vae_orbax", "action_dit_init"):
        path = getattr(m.weights, name)
        _require(
            path.startswith(prefix),
            f"model.weights.{name} = {path!r} is outside gs://{cfg.tpu.gcs.region_bucket}",
        )
    for name in ("prefix", "stats_path", "text_dir"):
        path = getattr(d.l3, name)
        if path is not None and path.startswith("gs://"):
            _require(
                path.startswith(prefix),
                f"data.l3.{name} = {path!r} is outside gs://{cfg.tpu.gcs.region_bucket}; streaming "
                f"pixels across regions on every step is the expensive kind of silent bug",
            )

def _check_data_conventions(cfg: Config) -> None:
    d = cfg.data
    _require(
        d.l3.source_color_order in ("bgr", "rgb"),
        f"data.l3.source_color_order must be bgr|rgb, got {d.l3.source_color_order!r}",
    )
    _require(
        d.l3.text_mode in ("sidetable", "baked"),
        f"data.l3.text_mode must be sidetable|baked, got {d.l3.text_mode!r}",
    )
    _require(
        d.l3.text_mode != "sidetable" or d.l3.text_dir is not None,
        "data.l3.text_mode='sidetable' needs data.l3.text_dir",
    )
    if d.l3.task_filter is not None:
        _require(
            len(d.l3.task_filter) >= 1,
            "data.l3.task_filter is an empty list; write null to mean 'every task'",
        )
        _require(
            all(name and name == name.strip() for name in d.l3.task_filter),
            f"data.l3.task_filter has an empty or space-padded entry: {list(d.l3.task_filter)}",
        )
        # A duplicate is harmless to the loader (it de-duplicates) but it makes
        # `len(task_filter)` lie, and that number is what the run banner reports as the
        # size of the subset.
        seen = sorted({n for n in d.l3.task_filter if d.l3.task_filter.count(n) > 1})
        _require(not seen, f"data.l3.task_filter repeats {seen}")
    _require(
        d.l3.episode_weighting in ("transitions", "uniform"),
        f"data.l3.episode_weighting must be transitions|uniform, got {d.l3.episode_weighting!r}",
    )
    _require(
        d.l3.action_shift in (0, 1),
        f"data.l3.action_shift must be 0 or 1 (D5), got {d.l3.action_shift}",
    )
    _require(d.l3.pool_size >= 1, "data.l3.pool_size must be >= 1")
    _require(
        0.0 < d.l3.pool_rate <= d.l3.pool_size,
        f"data.l3.pool_rate must lie in (0, pool_size={d.l3.pool_size}], got {d.l3.pool_rate}",
    )

    # --- geometry (raises inside if the frame counts do not divide) ---
    _derive_shapes(cfg)

    # The canvas the loader builds has to be the canvas the config says it trains on.
    # Local import: config.py stays importable without the data package's dependencies.
    from fastwam_tpu.data.canvas import assert_canvas_matches

    try:
        if d.format == "libero_hdf5":
            _require(d.train.video_size == (224, 448), "LIBERO requires the 224x448 two-camera canvas")
            _require(d.train.processor.num_output_cameras == 2, "LIBERO requires two RGB cameras")
            _require(cfg.model.action_dit_config.action_dim == 7 and cfg.model.proprio_dim == 8,
                     "LIBERO requires action_dim=7 and proprio_dim=8")
        else:
            assert_canvas_matches(d.train.video_size, d.train.processor.num_output_cameras)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    _require(
        len(d.train.shape_meta.images) == d.train.processor.num_output_cameras,
        f"shape_meta lists {len(d.train.shape_meta.images)} cameras but num_output_cameras is "
        f"{d.train.processor.num_output_cameras}",
    )


def _derive_shapes(cfg: Config) -> DerivedShapes:
    node = cfg.data.train
    ratio = node.action_video_freq_ratio
    num_frames = node.num_frames

    # robot_video_dataset.py:_get asserts both of these.
    _require(
        (num_frames - 1) % ratio == 0,
        f"(num_frames-1) must divide by action_video_freq_ratio: {num_frames}, {ratio}",
    )
    video_frames = (num_frames - 1) // ratio + 1
    _require(
        (video_frames - 1) % VAE_TEMPORAL_FACTOR == 0,
        f"video frames after subsampling ({video_frames}) must satisfy (T-1) % {VAE_TEMPORAL_FACTOR} == 0",
    )

    height, width = node.video_size
    for name, size in (("height", height), ("width", width)):
        _require(
            size % VAE_SPATIAL_FACTOR == 0,
            f"video_size {name} ({size}) must divide by {VAE_SPATIAL_FACTOR}",
        )
    latent_h = height // VAE_SPATIAL_FACTOR
    latent_w = width // VAE_SPATIAL_FACTOR

    _, patch_h, patch_w = cfg.model.video_dit_config.patch_size
    for name, size, patch in (("height", latent_h, patch_h), ("width", latent_w, patch_w)):
        _require(size % patch == 0, f"latent {name} ({size}) must divide by patch {patch}")
    tokens_per_frame = (latent_h // patch_h) * (latent_w // patch_w)

    latent_frames = (video_frames - 1) // VAE_TEMPORAL_FACTOR + 1
    action_horizon = num_frames - 1

    # The yaml knobs -> the one tuple every consumer reads. Raises here, at config load,
    # rather than inside a jitted step on a queued slice. Re-raised as ConfigError so a
    # bad `flow:` block fails like every other bad key.
    try:
        copy_lengths = resolve_copy_lengths(
            cfg.flow.tau_copies, cfg.flow.copy_lengths, action_horizon
        )
    except CopyLengthError as exc:
        raise ConfigError(f"flow: {exc}") from exc
    train_action_tokens = sum(copy_lengths)

    return DerivedShapes(
        video_frames=video_frames,
        action_horizon=action_horizon,
        latent_channels=cfg.model.video_dit_config.in_dim,
        latent_frames=latent_frames,
        latent_height=latent_h,
        latent_width=latent_w,
        tokens_per_frame=tokens_per_frame,
        video_tokens=latent_frames * tokens_per_frame,
        action_tokens=action_horizon,
        joint_seq=latent_frames * tokens_per_frame + action_horizon,
        prefill_video_tokens=tokens_per_frame,
        prefill_joint_seq=tokens_per_frame + action_horizon,
        context_tokens=node.context_len + 1,  # +1 proprio token (fastwam.py:219)
        copy_lengths=copy_lengths,
        train_action_tokens=train_action_tokens,
        train_joint_seq=latent_frames * tokens_per_frame + train_action_tokens,
    )
