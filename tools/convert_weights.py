"""torch/safetensors -> JAX Orbax weight conversion. Run once, on a host, on CPU.

M1 implements the video DiT; M2 adds the ActionDiT initialization and the
official-checkpoint path. The VAE tree belongs to models/wan/vae.py's owner.

Ported from
    codebases/TinyWAM-v4/tools/convert_wan.py — the PROCESS is the deliverable here:
        download -> weight_map against the NNX abstract state with hard key/shape
        asserts -> numeric verify against a torch reference on a fixed tiny input ->
        Orbax save (local or gs://) -> read back and re-verify -> write a sidecar.
        Measured 2026-07: transformer 3.3e-6, VAE decode 9.9e-6.
    reference/maxdiffusion/src/maxdiffusion/models/wan/wan_utils.py
        rename_for_custom_trasformer (L65) — the rename table for NATIVE Wan naming
        (blocks.N.self_attn.q.weight, head.head.weight, ...), which is exactly what
        the Wan2.2-TI2V-5B safetensors use. get_key_and_value (L91) packs
        blocks.7.xxx into slice 7 of the scanned stack.
    reference/maxdiffusion/src/maxdiffusion/models/modeling_flax_pytorch_utils.py
        torch2jax, rename_key_and_reshape_tensor (Linear -> transpose; Conv3d ->
        transpose(2,3,4,1,0)).

M1 RESULT — there is no rename table for the video DiT.
    The public `ipplab/Wan2.2-TI2V-5B-Orbax` release holds the
    converted Wan2.2-TI2V-5B DiT in an Orbax tree compatible with
    maxdiffusion's NNX tree: 825 keys = 30 blocks x 27 + 15 top-level. This port
    deliberately uses the same module attribute names (see models/wan/transformer.py),
    so checkpoint key -> module path is the IDENTITY and the conversion collapses to
    a validation: every key present, every shape equal, nothing missing, nothing
    extra. `check_video_dit` raises on any of the three.
    The safetensors -> Orbax step is therefore already done and is not re-run; if it
    ever has to be, the maxdiffusion rename table above is the source.

Non-negotiable, and the reason this script exists at all: the reference loads
everything with strict=False and maxdiffusion's validate_flax_state_dict only logs.
Here a missing key, an extra key or a shape mismatch RAISES.

CLI
    # metadata-only, no arrays touched — safe on a laptop
    uv run python tools/convert_weights.py check-transformer \\
        checkpoints/wan/video_dit --config configs/training/robotwin-baseline.example.yaml

    # when the host has gsutil credentials but no application-default credentials
    # (gcsfs/orbax need ADC), copy the tree metadata down first:
    #   gsutil cp gs://.../transformer/_METADATA /tmp/_METADATA
    uv run python tools/convert_weights.py check-transformer \\
        --config configs/training/robotwin-baseline.example.yaml --metadata-file /tmp/_METADATA

    # Full restore and shape check (requires sufficient host RAM):
    uv run python tools/convert_weights.py load-transformer checkpoints/wan/video_dit \\
        --config configs/training/robotwin-baseline.example.yaml

    # --- M2 ---
    # what the released .pt must contain, derived from our model (metadata only):
    uv run python tools/convert_weights.py dump-official-inventory /tmp/expected.json \\
        --config configs/training/robotwin-baseline.example.yaml
    uv run python tools/convert_weights.py check-official key_inventory.json \\
        --config configs/training/robotwin-baseline.example.yaml

    # ActionDiT init: interpolate the video DiT 3072 -> 1024 (spec 5.1). ~20 GB in,
    # ~4 GB out.
    uv run python tools/convert_weights.py build-action-init \\
        checkpoints/wan/video_dit checkpoints/wan/action_init \\
        --config configs/training/robotwin-baseline.example.yaml

    # Official checkpoint -> Orbax (requires PyTorch).
    uv run --extra convert python tools/convert_weights.py convert-official \\
        robotwin_uncond_3cam_384.pt \\
        checkpoints/fastwam/official_uncond --config configs/training/robotwin-baseline.example.yaml
"""

import argparse
import json
import sys
from pathlib import Path

__all__ = [
    "ConversionError",
    "video_dit_param_shapes",
    "action_dit_param_shapes",
    "read_orbax_shapes",
    "read_orbax_shapes_from_metadata_file",
    "compare_param_shapes",
    "stack_blocks_for_scan",
    "unstack_scanned_blocks",
    "load_video_dit_params",
    "to_torch_layout",
    "from_torch_layout",
    "jax_to_torch_key",
    "official_torch_inventory",
    "check_official_inventory",
    "verify_official_fingerprints",
    "build_action_dit_init",
    "convert_official_checkpoint",
    "save_params",
    "main",
]


class ConversionError(ValueError):
    """Any key/shape disagreement between a checkpoint and the model. Never logged."""


def _dotted(path: tuple) -> str:
    return ".".join(str(part) for part in path)


# --------------------------------------------------------------------------- #
# what the model expects
# --------------------------------------------------------------------------- #
def video_dit_param_shapes(config, *, scan_layers: bool = False) -> dict[str, tuple[int, ...]]:
    """{dotted param path: shape} for a WanVideoDiT built from `config`.

    Uses nnx.eval_shape, so nothing is allocated — this is safe to call for the 5 B
    model on a laptop.
    """
    from flax import nnx

    from fastwam_tpu.models.wan.transformer import WanVideoDiT

    def build() -> WanVideoDiT:
        return WanVideoDiT.from_config(config, rngs=nnx.Rngs(0), scan_layers=scan_layers)

    abstract = nnx.eval_shape(build)
    state = nnx.state(abstract, nnx.Param)
    return {_dotted(path): tuple(value.shape) for path, value in nnx.to_flat_state(state)}


# --------------------------------------------------------------------------- #
# what the checkpoint holds
# --------------------------------------------------------------------------- #
def read_orbax_shapes(path: str) -> dict[str, tuple[int, ...]]:
    """{dotted param path: shape} from an Orbax tree, reading metadata only.

    Works for local paths and gs://. The gs:// case needs APPLICATION-DEFAULT
    credentials, which are not the same thing as a working `gsutil`: gsutil uses the
    gcloud user credentials, gcsfs (and therefore orbax) uses ADC, and a host with the
    former but not the latter gets a bare 401. On a TPU VM the metadata server supplies
    ADC, so this path is the normal one there; on a dev box without ADC, use
    `read_orbax_shapes_from_metadata_file` instead — the error below says how.
    """
    import orbax.checkpoint as ocp

    try:
        metadata = ocp.PyTreeCheckpointer().metadata(path)
    except Exception as exc:  # noqa: BLE001 - re-raised with the fix attached
        if path.startswith("gs://") and "401" in str(exc):
            raise ConversionError(
                f"no application-default credentials for {path} (gsutil working is not enough).\n"
                "Either run this on the TPU VM, or copy the tree metadata down and pass it:\n"
                f"  gsutil cp {path.rstrip('/')}/_METADATA /tmp/_METADATA\n"
                "  ... check-transformer --metadata-file /tmp/_METADATA"
            ) from exc
        raise

    # orbax 0.12 wraps the tree: `metadata()` now returns a StepMetadata whose
    # `item_metadata` holds what used to be returned directly. Flattening the wrapper
    # yields exactly one entry keyed by the empty path, so the old code produced a
    # single bogus "" key instead of raising -- silent, which is the failure mode this
    # module exists to prevent. Verified against the live 825-key wan22_5b tree.
    tree = getattr(metadata, "item_metadata", metadata)
    flat = ocp.tree.to_flat_dict(tree)
    if len(flat) <= 1:
        raise ConversionError(
            f"{path}: metadata flattened to {len(flat)} entries, which cannot be a real "
            "parameter tree -- the orbax metadata API has moved again"
        )
    # Entries are ArrayMetadata under some orbax versions and bare shape tuples under
    # others; both are accepted, anything else is a hard error rather than a guess.
    shapes: dict[str, tuple[int, ...]] = {}
    for key, value in flat.items():
        shape = getattr(value, "shape", value)
        if not isinstance(shape, (tuple, list)):
            raise ConversionError(f"{path}: {_dotted(key)} has no readable shape ({value!r})")
        shapes[_dotted(key)] = tuple(shape)
    return shapes


def read_orbax_shapes_from_metadata_file(metadata_file: str | Path) -> dict[str, tuple[int, ...]]:
    """Same, from a copy of the tree's `_METADATA` json (`gsutil cp` it down).

    `_METADATA` records `key_metadata` (the pytree path) and `write_shape` for every
    array, which is all a key/shape audit needs — no array bytes are involved.
    """
    raw = json.loads(Path(metadata_file).read_text())
    tree = raw.get("tree_metadata")
    if not isinstance(tree, dict):
        raise ConversionError(f"{metadata_file}: not an orbax _METADATA file (no tree_metadata)")
    shapes: dict[str, tuple[int, ...]] = {}
    for entry in tree.values():
        path = tuple(part["key"] for part in entry["key_metadata"])
        value = entry["value_metadata"]
        if "write_shape" not in value:
            raise ConversionError(f"{metadata_file}: {_dotted(path)} has no write_shape")
        shapes[_dotted(path)] = tuple(value["write_shape"])
    return shapes


# --------------------------------------------------------------------------- #
# the audit
# --------------------------------------------------------------------------- #
def compare_param_shapes(
    expected: dict[str, tuple[int, ...]],
    actual: dict[str, tuple[int, ...]],
    *,
    what: str = "checkpoint",
) -> int:
    """Raise unless `actual` has exactly `expected`'s keys with exactly its shapes."""
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(k for k in set(expected) & set(actual) if expected[k] != actual[k])
    if missing or extra or mismatched:
        lines = [f"{what} does not match the model:"]
        if missing:
            lines.append(f"  missing {len(missing)}: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        if extra:
            lines.append(f"  extra {len(extra)}: {extra[:10]}{' ...' if len(extra) > 10 else ''}")
        for key in mismatched[:10]:
            lines.append(f"  shape {key}: model {expected[key]} vs {what} {actual[key]}")
        if len(mismatched) > 10:
            lines.append(f"  ... and {len(mismatched) - 10} more shape mismatches")
        raise ConversionError("\n".join(lines))
    return len(expected)


def check_video_dit(config, source: dict[str, tuple[int, ...]], *, what: str = "checkpoint") -> int:
    """Validate a checkpoint's key/shape inventory against the video tower."""
    return compare_param_shapes(video_dit_param_shapes(config, scan_layers=False), source, what=what)


# --------------------------------------------------------------------------- #
# scanned <-> unscanned block layout
# --------------------------------------------------------------------------- #
def stack_blocks_for_scan(params: dict[str, "object"], num_layers: int) -> dict[str, "object"]:
    """`blocks.0.x ... blocks.29.x` -> one `blocks.x` with a leading [num_layers] axis.

    The checkpoint is written unscanned (a plain list of blocks); `scan_layers=True`
    needs every block parameter stacked. maxdiffusion's `get_key_and_value` does the
    same packing on the safetensors side.
    """
    import numpy as np

    stacked: dict[str, object] = {}
    per_layer: dict[str, list] = {}
    for key, value in params.items():
        if not key.startswith("blocks."):
            stacked[key] = value
            continue
        _, index, rest = key.split(".", 2)
        per_layer.setdefault(rest, [None] * num_layers)[int(index)] = value
    for rest, values in per_layer.items():
        if any(v is None for v in values):
            missing = [i for i, v in enumerate(values) if v is None]
            raise ConversionError(f"blocks.*.{rest} is missing layers {missing}")
        stacked[f"blocks.{rest}"] = np.stack([np.asarray(v) for v in values], axis=0)
    return stacked


def unstack_scanned_blocks(params: dict[str, "object"], num_layers: int) -> dict[str, "object"]:
    """Inverse of :func:`stack_blocks_for_scan`, for writing a checkpoint back out."""
    import numpy as np

    flat: dict[str, object] = {}
    for key, value in params.items():
        if not key.startswith("blocks."):
            flat[key] = value
            continue
        array = np.asarray(value)
        if array.shape[0] != num_layers:
            raise ConversionError(f"{key}: leading axis {array.shape[0]} != num_layers {num_layers}")
        rest = key.split(".", 1)[1]
        for index in range(num_layers):
            flat[f"blocks.{index}.{rest}"] = array[index]
    return flat


# --------------------------------------------------------------------------- #
# the real load — TPU side
# --------------------------------------------------------------------------- #
def load_video_dit_params(path: str, config, *, scan_layers: bool = False):
    """Restore the video tower and return its parameters as a flat {dotted: array} dict.

    Thin wrapper over `models/wan/transformer.py::load_video_dit` — one restore
    implementation, including the flax-0.10-vs-0.12 list-index key adaptation, so the
    CLI and the runtime cannot drift apart. This adds the post-restore shape audit
    against the model, which the local metadata check cannot prove for real arrays.

    RUN ON THE TPU VM. The 5 B tree is 17.25 GiB on disk and ~20 GB materialised; the
    dev host does not have room for it, which is why the local gate is
    `check_video_dit` (metadata only).
    """
    from flax import nnx

    from fastwam_tpu.models.wan.transformer import load_video_dit

    model = load_video_dit(path, config, scan_layers=scan_layers)
    flat = {_dotted(key): value[...] for key, value in nnx.to_flat_state(nnx.state(model, nnx.Param))}
    compare_param_shapes(
        video_dit_param_shapes(config, scan_layers=scan_layers),
        {key: tuple(value.shape) for key, value in flat.items()},
        what="restored checkpoint",
    )
    return flat


def load_vae_params(*args, **kwargs):
    raise NotImplementedError(
        "the Wan2.2 (z=48) VAE tree belongs to models/wan/vae.py; its rename map has to be "
        "written from scratch because maxdiffusion only ships a 2.1-diffusers map"
    )


# --------------------------------------------------------------------------- #
# M2: torch <-> jax layout and key names
# --------------------------------------------------------------------------- #
# Every rename is one of three kinds:
#   linear   torch Linear weight [out, in]        -> flax kernel [in, out]      (transpose)
#   conv3d   torch Conv3d weight [out,in,t,h,w]   -> flax kernel [t,h,w,in,out] (2,3,4,1,0)
#   identity biases, LayerNorm/RMSNorm scales, the two modulation tables
# `.bias`, `.scale` and the tables are all identity; only `.kernel` ever moves.
_LINEAR_RENAMES = {
    "condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.time_proj": "time_projection.1",
    "condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.2",
}
_ATTENTION_RENAMES = {"attn1": "self_attn", "attn2": "cross_attn"}
_PROJECTION_RENAMES = {"query": "q", "key": "k", "value": "v", "proj_attn": "o"}
# The leaf-name translation. flax `.kernel` is torch `.weight`; flax norm `.scale` is
# torch `.weight` too, which is why the leaf map cannot be derived from the suffix alone.
_LEAF_RENAMES = {"kernel": "weight", "scale": "weight", "bias": "bias"}

# Which of OUR keys have no counterpart in the video tower, per tower. The video tower's
# `proj_out` (the patchified head) and the action tower's `proj_out` (a plain action
# head) share a name and are different tensors — see models/action_dit.py.
_VIDEO_ONLY_TOP_LEVEL = ("patch_embedding.", "scale_shift_table")


def to_torch_layout(key: str, array):
    """flax parameter -> the torch tensor it came from (transpose / permute / as-is)."""
    import numpy as np

    array = np.asarray(array)
    if not key.endswith(".kernel"):
        return array
    if array.ndim == 5:  # Conv3d: [p_t, p_h, p_w, in, out] -> [out, in, p_t, p_h, p_w]
        return np.transpose(array, (4, 3, 0, 1, 2))
    if array.ndim == 2:  # Linear: [in, out] -> [out, in]
        return array.T
    raise ConversionError(f"{key}: a .kernel must be 2-D (Linear) or 5-D (Conv3d), got {array.shape}")


def from_torch_layout(key: str, array):
    """Inverse of :func:`to_torch_layout`."""
    import numpy as np

    array = np.asarray(array)
    if not key.endswith(".kernel"):
        return array
    if array.ndim == 5:
        return np.transpose(array, (2, 3, 4, 1, 0))
    if array.ndim == 2:
        return array.T
    raise ConversionError(f"{key}: a .kernel must be 2-D (Linear) or 5-D (Conv3d), got {array.shape}")


def jax_to_torch_key(key: str, tower: str) -> str:
    """Our dotted parameter path -> FastWAM's torch state-dict key, for one tower.

    Raises on anything it does not recognise: an unmapped key is exactly the silent
    partial load this whole module exists to prevent.
    """
    if tower not in ("video", "action"):
        raise ConversionError(f"tower must be 'video' or 'action', got {tower!r}")
    head, _, leaf = key.rpartition(".")
    if leaf not in _LEAF_RENAMES and not key.endswith("_table"):
        raise ConversionError(f"unrecognised parameter leaf in {key!r}")
    torch_leaf = _LEAF_RENAMES.get(leaf)

    if key in ("scale_shift_table", "blocks.adaln_scale_shift_table"):
        return "head.modulation"
    if head in _LINEAR_RENAMES:
        return f"{_LINEAR_RENAMES[head]}.{torch_leaf}"
    if head == "patch_embedding":
        if tower == "action":
            raise ConversionError("the action expert has no patch_embedding")
        return f"patch_embedding.{torch_leaf}"
    if head == "action_encoder":
        if tower == "video":
            raise ConversionError("the video expert has no action_encoder")
        return f"action_encoder.{torch_leaf}"
    if head == "proj_out":
        # The one genuinely tower-dependent name: `Head.head` vs a bare `head` Linear.
        return f"head.head.{torch_leaf}" if tower == "video" else f"head.{torch_leaf}"

    if key.startswith("blocks."):
        _, index, rest = key.split(".", 2)
        if rest == "adaln_scale_shift_table":
            return f"blocks.{index}.modulation"
        parts = rest.split(".")
        if parts[0] in _ATTENTION_RENAMES:
            attn = _ATTENTION_RENAMES[parts[0]]
            if parts[1] in _PROJECTION_RENAMES:
                return f"blocks.{index}.{attn}.{_PROJECTION_RENAMES[parts[1]]}.{torch_leaf}"
            if parts[1] in ("norm_q", "norm_k"):
                return f"blocks.{index}.{attn}.{parts[1]}.weight"
        if parts[0] == "norm2":  # ours-norm2 == FastWAM-norm3 (affine, pre-cross-attn)
            return f"blocks.{index}.norm3.{torch_leaf}"
        if parts[:3] == ["ffn", "act_fn", "proj"]:
            return f"blocks.{index}.ffn.0.{torch_leaf}"
        if parts[:2] == ["ffn", "proj_out"]:
            return f"blocks.{index}.ffn.2.{torch_leaf}"

    raise ConversionError(f"no torch name known for {key!r} (tower {tower})")


def action_dit_param_shapes(config, *, scan_layers: bool = False) -> dict[str, tuple[int, ...]]:
    """{dotted param path: shape} for an ActionDiT built from `config.action_dit_config`."""
    from flax import nnx

    from fastwam_tpu.models.action_dit import ActionDiT

    abstract = nnx.eval_shape(
        lambda: ActionDiT.from_config(config, rngs=nnx.Rngs(0), scan_layers=scan_layers)
    )
    state = nnx.state(abstract, nnx.Param)
    return {_dotted(path): tuple(value.shape) for path, value in nnx.to_flat_state(state)}


# --------------------------------------------------------------------------- #
# M2: the ActionDiT interpolation initialization (spec §5.1)
# --------------------------------------------------------------------------- #
def _interpolate_axis(tensor, axis: int, new_size: int):
    """F.interpolate(mode="linear", align_corners=True) along one axis, in fp32.

    align_corners=True: output i reads source position i*(L-1)/(N-1), so both endpoints
    survive exactly. N == 1 is torch's degenerate case (scale 0, index 0).
    """
    import numpy as np

    length = tensor.shape[axis]
    if length == new_size:
        return tensor
    moved = np.moveaxis(tensor, axis, -1)
    if new_size == 1:
        out = moved[..., :1]
    else:
        position = np.arange(new_size, dtype=np.float64) * ((length - 1) / (new_size - 1))
        low = np.floor(position).astype(np.int64)
        high = np.minimum(low + 1, length - 1)
        weight = (position - low).astype(np.float32)
        out = moved[..., low] * (np.float32(1.0) - weight) + moved[..., high] * weight
    return np.moveaxis(out, -1, axis)


def _resize_to_shape(src, target_shape):
    """`_resize_tensor_to_shape` (preprocess_action_dit_backbone.py:61-96), fp32.

    The reference's rank-fixing unsqueeze/squeeze loop is not reproduced: it never fires
    for any ActionDiT key (every backbone tensor has the same rank in both towers), so
    porting it would be untested code. A rank mismatch raises instead.
    """
    import numpy as np

    src = np.asarray(src, dtype=np.float32)
    if src.ndim != len(target_shape):
        raise ConversionError(f"rank mismatch resizing {src.shape} -> {target_shape}")
    out = src
    for axis, new_size in enumerate(target_shape):
        out = _interpolate_axis(out, axis, new_size)
    if tuple(out.shape) != tuple(target_shape):
        raise ConversionError(f"resize produced {out.shape}, expected {target_shape}")
    return out


def build_action_dit_init(
    video_params: dict, action_shapes: dict[str, tuple[int, ...]], *, apply_alpha_scaling: bool = True
) -> dict:
    """The ActionDiT backbone init: the video expert, linearly interpolated to 1024.

    `scripts/preprocess_action_dit_backbone.py`, spec §5.1. For every ActionDiT
    parameter except `action_encoder.*` / `proj_out.*`, take the same-named video tensor
    and resize it by SEQUENTIAL per-dimension 1-D linear interpolation
    (align_corners=True) in float32; then, if the tensor is at least 2-D and its
    **torch-layout last dimension** changed, scale by

        alpha = sqrt(video_last_dim / action_last_dim)

    which is sqrt(3) for the 3072->1024 pairs but sqrt(3.5) for `ffn.2` (14336->4096) —
    the constant is per-tensor, not global. Note "torch-layout last dimension": for a
    Linear that is the INPUT width, i.e. our kernel's axis 0, which is why every tensor
    is round-tripped through :func:`to_torch_layout` rather than interpolated in place.
    Interpolating in flax layout without that flip silently alpha-scales `attn.o` and
    silently does not scale `attn.q` — the exact opposite of the reference.

    This is NOT random initialization. `action_encoder` / `proj_out` stay random and are
    the caller's to supply (they are absent from the returned dict).

    Args:
        video_params: {dotted path: array} for the video expert, flax layout.
        action_shapes: {dotted path: shape} for the action expert, flax layout,
            INCLUDING the two skipped modules (they are filtered here, and their
            presence is checked, so a rename cannot widen the skip silently).

    Returns:
        {dotted path: fp32 array} covering exactly the backbone keys.
    """
    import numpy as np

    from fastwam_tpu.models.action_dit import ACTION_ONLY_KEYS

    present = ACTION_ONLY_KEYS & set(action_shapes)
    if present != ACTION_ONLY_KEYS:
        raise ConversionError(
            f"the action tree must contain exactly the skip keys {sorted(ACTION_ONLY_KEYS)}; "
            f"missing {sorted(ACTION_ONLY_KEYS - present)}. A renamed module would otherwise "
            "be interpolated from a same-named video tensor of unrelated meaning."
        )

    out: dict[str, object] = {}
    copied = interpolated = 0
    for key in sorted(set(action_shapes) - ACTION_ONLY_KEYS):
        if key not in video_params:
            raise ConversionError(f"`{key}` is in the action tree but not in the video expert")
        target = tuple(action_shapes[key])
        source = np.asarray(video_params[key])
        if source.shape == target:
            out[key] = source.astype(np.float32)
            copied += 1
            continue

        torch_src = to_torch_layout(key, source)
        torch_target = tuple(to_torch_layout(key, np.zeros(target, np.float32)).shape)
        value = _resize_to_shape(torch_src, torch_target)
        if apply_alpha_scaling and torch_src.ndim >= 2 and torch_src.shape[-1] != torch_target[-1]:
            value = value * np.float32((float(torch_src.shape[-1]) / float(torch_target[-1])) ** 0.5)
        out[key] = from_torch_layout(key, value).astype(np.float32)
        if out[key].shape != target:
            raise ConversionError(f"{key}: produced {out[key].shape}, expected {target}")
        interpolated += 1

    if copied + interpolated != len(action_shapes) - len(ACTION_ONLY_KEYS):
        raise ConversionError("internal: the backbone key count does not add up")
    return out


# --------------------------------------------------------------------------- #
# M2: the official released checkpoint
# --------------------------------------------------------------------------- #
# `FastWAM.save_checkpoint` (fastwam.py:1088-1098) writes
#   {"mot": mot.state_dict(), "proprio_encoder": ..., "step": int, "torch_dtype": str}
# and MoT holds the two experts in an nn.ModuleDict called `mixtures`.
OFFICIAL_TOWER_PREFIXES = {"video": "mixtures.video.", "action": "mixtures.action."}
OFFICIAL_PROPRIO_KEYS = {"kernel": "weight", "bias": "bias"}


def official_torch_inventory(config) -> dict[str, dict[str, tuple[int, ...]]]:
    """The key/shape inventory the released `.pt` must have, derived from OUR model.

    Returns `{"mot": {torch key: shape}, "proprio_encoder": {torch key: shape}}` with
    torch's own layouts (Linear [out, in], Conv3d [out, in, t, h, w]).

    This is the cross-check target for the `key_inventory.json` dumped from the real
    12 GB file: if the two agree, the rename table and the layout flips are right, and
    the conversion below is a mechanical application of them.
    """
    video = video_dit_param_shapes(config.video_dit_config, scan_layers=False)
    action = action_dit_param_shapes(config.action_dit_config, scan_layers=False)

    mot: dict[str, tuple[int, ...]] = {}
    for tower, shapes in (("video", video), ("action", action)):
        prefix = OFFICIAL_TOWER_PREFIXES[tower]
        for key, shape in shapes.items():
            torch_key = prefix + jax_to_torch_key(key, tower)
            if torch_key in mot:
                raise ConversionError(f"two of our keys map onto the same torch key {torch_key!r}")
            mot[torch_key] = _torch_shape(key, shape)

    text_dim = config.video_dit_config.text_dim
    proprio = {"weight": (text_dim, config.proprio_dim), "bias": (text_dim,)}
    return {"mot": mot, "proprio_encoder": proprio}


def _torch_shape(key: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    """The torch-layout shape of one of our parameters, without allocating it."""
    if not key.endswith(".kernel"):
        return tuple(shape)
    if len(shape) == 2:
        return (shape[1], shape[0])
    if len(shape) == 5:
        return (shape[4], shape[3], shape[0], shape[1], shape[2])
    raise ConversionError(f"{key}: a .kernel must be 2-D or 5-D, got {shape}")


def _normalize_inventory(raw) -> dict[str, tuple[int, ...]]:
    """Accept the shapes a key dump might plausibly take, reject anything else.

    The dump is produced by a different agent on the v4-32, so this is deliberately
    forgiving about the container and strict about the content.
    """
    if isinstance(raw, dict):
        for candidate in ("keys", "state_dict", "inventory", "tensors"):
            if candidate in raw and isinstance(raw[candidate], dict):
                raw = raw[candidate]
                break
    if isinstance(raw, list):
        entries = {}
        for item in raw:
            if not isinstance(item, dict) or "shape" not in item:
                raise ConversionError(f"inventory list entries must be {{name, shape}}, got {item!r}")
            name = item.get("name") or item.get("key")
            if name is None:
                raise ConversionError(f"inventory entry has no name/key: {item!r}")
            entries[str(name)] = tuple(item["shape"])
        return entries
    if not isinstance(raw, dict):
        raise ConversionError(f"cannot read a key inventory out of a {type(raw).__name__}")

    entries = {}
    for name, value in raw.items():
        if isinstance(value, dict):
            if "shape" not in value:
                raise ConversionError(f"inventory entry {name!r} has no 'shape'")
            value = value["shape"]
        if not isinstance(value, (list, tuple)):
            raise ConversionError(f"inventory entry {name!r} must map to a shape, got {value!r}")
        entries[str(name)] = tuple(int(dim) for dim in value)
    return entries


def check_official_inventory(config, inventory_path: str | Path) -> int:
    """Compare a dumped `key_inventory.json` against :func:`official_torch_inventory`.

    The dump may be flat (`mot.mixtures.video.…`, `proprio_encoder.weight`, as the
    payload nests them) or already split per section; both are handled. Non-tensor
    payload entries (`step`, `torch_dtype`, `optimizer`) are ignored.
    """
    raw = json.loads(Path(inventory_path).read_text())
    if isinstance(raw, dict) and set(raw) & {"mot", "proprio_encoder"}:
        flat = {}
        for section in ("mot", "proprio_encoder"):
            if section in raw:
                flat.update(
                    {f"{section}.{k}": v for k, v in _normalize_inventory(raw[section]).items()}
                )
    else:
        flat = _normalize_inventory(raw)
    flat = {k: v for k, v in flat.items() if k.startswith(("mot.", "proprio_encoder."))}

    expected = official_torch_inventory(config)
    want = {f"mot.{k}": v for k, v in expected["mot"].items()}
    want.update({f"proprio_encoder.{k}": v for k, v in expected["proprio_encoder"].items()})
    return compare_param_shapes(want, flat, what=str(inventory_path))


def _tensor_fingerprint(array) -> dict[str, float]:
    """sum / mean / std / absmax in float64 — the probe's per-tensor fingerprint.

    All four are permutation-invariant, so this catches a tensor landing on the WRONG
    KEY, a dtype/precision loss, or corruption. It does NOT catch a transposed square
    matrix: that is settled structurally (torch Linear stores [out, in], flax [in, out],
    so a square kernel still has to be flipped) and pinned by
    `test_square_kernels_are_transposed_too`.
    """
    import numpy as np

    values = np.asarray(array, dtype=np.float64)
    return {
        "sum": float(values.sum()),
        "mean": float(values.mean()),
        # The probe dumped stats with pure torch, whose Tensor.std() is
        # Bessel-corrected (ddof=1) by default — numpy's default ddof=0 misses
        # every tensor by the sqrt(N/(N-1)) factor (~1.6e-4 on a 3072 bias).
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "absmax": float(np.abs(values).max()),
    }


def verify_official_fingerprints(towers: dict[str, dict], inventory_path: str | Path) -> int:
    """Compare converted arrays against the probe's float64 per-tensor fingerprints.

    `towers` is :func:`convert_official_checkpoint`'s return value. Every array is
    fingerprinted in the TORCH layout it came from, so the numbers line up with the dump
    without caring which way the conversion flipped it.

    Relative tolerance is 1e-9 on mean/std/absmax and scaled by the tensor's magnitude on
    the sum, which accumulates over up to 4.4e8 elements. Any tensor missing from the
    dump, or off beyond that, raises.
    """
    raw = json.loads(Path(inventory_path).read_text())
    entries = raw.get("tensors", raw)
    if not isinstance(entries, dict):
        raise ConversionError(f"{inventory_path}: no 'tensors' mapping to check against")

    bad: list[str] = []
    checked = 0
    for tower, flat in towers.items():
        prefix = (
            OFFICIAL_TOWER_PREFIXES[tower] if tower in OFFICIAL_TOWER_PREFIXES else ""
        )
        for key, array in flat.items():
            if tower in OFFICIAL_TOWER_PREFIXES:
                dumped_key = "mot." + prefix + jax_to_torch_key(key, tower)
            else:
                dumped_key = f"proprio_encoder.{OFFICIAL_PROPRIO_KEYS[key]}"
            if dumped_key not in entries:
                bad.append(f"{dumped_key}: absent from {inventory_path}")
                continue
            want = entries[dumped_key]["stats"]
            got = _tensor_fingerprint(to_torch_layout(key, array))
            scale = max(abs(want["sum"]), 1.0)
            for stat, tol in (
                ("sum", 1e-9 * scale + 1e-6),
                ("mean", 1e-9 * max(abs(want["mean"]), 1e-6)),
                ("std", 1e-9 * max(abs(want["std"]), 1e-6)),
                ("absmax", 0.0),
            ):
                if abs(got[stat] - want[stat]) > tol:
                    bad.append(f"{dumped_key}.{stat}: got {got[stat]!r} want {want[stat]!r}")
            checked += 1

    if bad:
        raise ConversionError(
            f"fingerprint mismatch against {inventory_path} on {len(bad)} value(s):\n  "
            + "\n  ".join(bad[:20])
            + (f"\n  ... and {len(bad) - 20} more" if len(bad) > 20 else "")
        )
    return checked


def convert_official_checkpoint(pt_path: str, config, *, verify: bool = True) -> dict[str, dict]:
    """Read the released `robotwin_uncond_3cam_384.pt` -> our flat parameter dicts.

    Returns `{"video": {...}, "action": {...}, "proprio_encoder": {...}}`, flax layout,
    float32. Every one of our parameters must be present in the payload with the right
    torch shape, and the payload must have no tensor we do not consume — the reference
    loads this file with `strict=False` twice over (fastwam.py:1103, and again in
    `load_wan22_ti2v_5b_components`), which is exactly how a half-loaded 6 B model gets
    to closed-loop before anyone notices.

    Needs torch (the `convert` extra) to unpickle. 12 GB in, ~24 GB of fp32 out: run it
    on the v4-32 host, not on a laptop.
    """
    import numpy as np

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ConversionError(
            "reading the official .pt needs torch: `uv sync --extra convert`. "
            "The file is a pickled torch archive; there is no safe torch-free reader."
        ) from exc

    payload = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "mot" not in payload:
        raise ConversionError(f"{pt_path}: not a FastWAM checkpoint (no 'mot' key)")
    mot_state = payload["mot"]
    proprio_state = payload.get("proprio_encoder")
    if proprio_state is None:
        raise ConversionError(f"{pt_path}: no 'proprio_encoder' — RoboTwin uncond always has one")

    expected = official_torch_inventory(config)
    if verify:
        compare_param_shapes(
            expected["mot"],
            {key: tuple(value.shape) for key, value in mot_state.items()},
            what=f"{pt_path}[mot]",
        )
        compare_param_shapes(
            expected["proprio_encoder"],
            {key: tuple(value.shape) for key, value in proprio_state.items()},
            what=f"{pt_path}[proprio_encoder]",
        )

    out: dict[str, dict] = {"video": {}, "action": {}, "proprio_encoder": {}}
    for tower, shapes in (
        ("video", video_dit_param_shapes(config.video_dit_config)),
        ("action", action_dit_param_shapes(config.action_dit_config)),
    ):
        prefix = OFFICIAL_TOWER_PREFIXES[tower]
        for key in shapes:
            tensor = mot_state[prefix + jax_to_torch_key(key, tower)]
            array = tensor.detach().to(torch.float32).numpy()
            out[tower][key] = np.ascontiguousarray(from_torch_layout(key, array))
    for ours, theirs in OFFICIAL_PROPRIO_KEYS.items():
        array = proprio_state[theirs].detach().to(torch.float32).numpy()
        out["proprio_encoder"][ours] = np.ascontiguousarray(
            from_torch_layout(f"proprio_encoder.{ours}", array)
        )
    return out


def _nest(flat: dict) -> dict:
    """{"blocks.0.attn1.query.kernel": arr} -> the nested pure-dict orbax wants."""
    tree: dict = {}
    for key, value in flat.items():
        node = tree
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return tree


def save_params(flat: dict, path: str) -> None:
    """Write a flat {dotted: array} dict as an Orbax tree (local or ``gs://``)."""
    import orbax.checkpoint as ocp

    target = path if path.startswith("gs://") else str(Path(path).expanduser().resolve())
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(target, _nest(flat))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check-transformer", help="key/shape audit, metadata only")
    check.add_argument("path", nargs="?", help="Orbax tree (local or gs://)")
    check.add_argument("--config", required=True)
    check.add_argument("--metadata-file", help="a copy of the tree's _METADATA json")

    load = sub.add_parser("load-transformer", help="full restore + re-verify (run on the TPU VM)")
    load.add_argument("path")
    load.add_argument("--config", required=True)
    load.add_argument("--scan-layers", action="store_true")

    inventory = sub.add_parser(
        "check-official", help="cross-check a dumped key_inventory.json against our model"
    )
    inventory.add_argument("inventory", help="key_inventory.json from the released .pt")
    inventory.add_argument("--config", required=True)

    dump = sub.add_parser(
        "dump-official-inventory", help="write the key/shape inventory our model expects"
    )
    dump.add_argument("output", help="json path")
    dump.add_argument("--config", required=True)

    init = sub.add_parser(
        "build-action-init", help="ActionDiT interpolation init from the video DiT (spec 5.1)"
    )
    init.add_argument("video_ckpt", help="Orbax tree of the converted Wan2.2-5B video DiT")
    init.add_argument("output", help="Orbax dir to write (local or gs://)")
    init.add_argument("--config", required=True)
    init.add_argument("--no-alpha-scaling", action="store_true")
    init.add_argument("--seed", type=int, default=0, help="rng for action_encoder / proj_out")

    official = sub.add_parser(
        "convert-official", help="released .pt -> Orbax (needs torch; run on the TPU VM)"
    )
    official.add_argument("pt_path")
    official.add_argument("output", help="Orbax dir root; three subtrees are written")
    official.add_argument("--config", required=True)
    official.add_argument(
        "--inventory",
        default="tests/fixtures/official_key_inventory.json",
        help="per-tensor float64 fingerprints to verify against; '' to skip",
    )

    args = parser.parse_args(argv)

    from fastwam_tpu import load_config

    full_config = load_config(args.config)
    config = full_config.model.video_dit_config

    if args.command == "check-transformer":
        if bool(args.path) == bool(args.metadata_file):
            parser.error("pass exactly one of PATH or --metadata-file")
        if args.metadata_file:
            actual = read_orbax_shapes_from_metadata_file(args.metadata_file)
            source = args.metadata_file
        else:
            actual = read_orbax_shapes(args.path)
            source = args.path
        count = check_video_dit(config, actual, what=source)
        print(f"OK: {count} parameters, every key and shape matches {source}")
        return

    if args.command == "load-transformer":
        params = load_video_dit_params(args.path, config, scan_layers=args.scan_layers)
        print(f"OK: restored {len(params)} parameter arrays from {args.path}")
        return

    if args.command == "check-official":
        count = check_official_inventory(full_config.model, args.inventory)
        print(f"OK: {count} tensors, every key and shape matches {args.inventory}")
        return

    if args.command == "dump-official-inventory":
        expected = official_torch_inventory(full_config.model)
        payload = {
            section: {key: list(shape) for key, shape in shapes.items()}
            for section, shapes in expected.items()
        }
        Path(args.output).write_text(json.dumps(payload, indent=1, sort_keys=True))
        total = sum(len(shapes) for shapes in expected.values())
        print(f"OK: wrote {total} expected key/shape entries to {args.output}")
        return

    if args.command == "build-action-init":
        from flax import nnx

        from fastwam_tpu.models.action_dit import ACTION_ONLY_KEYS, ActionDiT

        action_config = full_config.model.action_dit_config
        video_params = load_video_dit_params(args.video_ckpt, config)
        shapes = action_dit_param_shapes(action_config)
        backbone = build_action_dit_init(
            video_params, shapes, apply_alpha_scaling=not args.no_alpha_scaling
        )
        # action_encoder / proj_out stay at their random init, exactly as upstream's
        # `from_pretrained` leaves them (action_dit.py:201-224).
        random_model = ActionDiT.from_config(action_config, rngs=nnx.Rngs(args.seed))
        random_flat = {
            _dotted(key): value[...]
            for key, value in nnx.to_flat_state(nnx.state(random_model, nnx.Param))
        }
        merged = {**backbone, **{key: random_flat[key] for key in sorted(ACTION_ONLY_KEYS)}}
        compare_param_shapes(
            shapes, {k: tuple(v.shape) for k, v in merged.items()}, what="action init"
        )
        save_params(merged, args.output)
        print(
            f"OK: wrote {len(merged)} ActionDiT parameters to {args.output} "
            f"({len(backbone)} interpolated/copied, {len(ACTION_ONLY_KEYS)} random)"
        )
        return

    towers = convert_official_checkpoint(args.pt_path, full_config.model)
    if args.inventory:
        checked = verify_official_fingerprints(towers, args.inventory)
        print(f"OK: {checked} tensors match the float64 fingerprints in {args.inventory}")
    root = args.output.rstrip("/")
    for name, flat in towers.items():
        save_params(flat, f"{root}/{name}")
        print(f"OK: wrote {len(flat)} parameters to {root}/{name}")


if __name__ == "__main__":
    sys.exit(main())
