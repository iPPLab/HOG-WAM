"""Assembling a deployable FastWAM: weights, VAE, normalization constants, policy.

M3 implements this. It is deliberately separate from `models/` because everything here
is a *deployment* concern — where the bytes live, how much memory the restore peaks at,
which of gsutil/gcsfs can actually read this host's GCS — and none of it should be in
the model definition.

The converted official checkpoint is three sibling Orbax trees, written by
`tools/convert_weights.py convert-official`:

    <root>/video             825 arrays   -> model.mot.video
    <root>/action            824 arrays   -> model.mot.action
    <root>/proprio_encoder     2 arrays   -> model.proprio_encoder

Memory: the restore order matters and is not a style choice. The obvious spelling —
build the model, restore into a fresh tree, then swap — holds the random init and the
restored weights at the same time, i.e. 24 GB at bf16, which does not fit a 24.6 GB
3090 alongside a VAE. So the random parameter buffers are **deleted before** the restore
is issued (`restore_into` below, the pattern proven in
`tinywam-eval/scripts/tinywam_model_server.py:88-99`), and the peak stays at one copy.

`gs://` paths: orbax reads them through gcsfs, which needs Application Default
Credentials. `gcloud auth login` does not create those, so on the eval workstation gcsfs
reads anonymously and 401s while `gsutil` works fine (measured 2026-08-04 against this
checkpoint). On a TPU VM ADC comes from the metadata server and gs:// just works. The
practical consequence for M3: **download the checkpoint first** — which the server wants
anyway, because its babysitter restarts it on death. `read_bytes` below sidesteps the
whole problem for the small files by shelling out to gsutil.
"""

import json
import os
import subprocess

import jax
import jax.numpy as jnp
from flax import nnx

from ..config import Config
from ..data.normalize import OfficialStats, load_official_stats
from ..models.fastwam import FastWAM
from ..models.wan.tree_keys import restore_tree_key_types, stringify_tree_keys
from ..models.wan.vae import load_vae
from .policy import FastWAMPolicy

__all__ = [
    "LoadError",
    "DTYPES",
    "read_bytes",
    "load_stats",
    "restore_into",
    "load_fastwam",
    "build_policy",
    "describe_assets",
]

DTYPES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


class LoadError(ValueError):
    """A checkpoint or asset that cannot be loaded as a correct FastWAM deployment."""


def read_bytes(path: str) -> bytes:
    """Read a local file or a `gs://` object.

    `gs://` goes through the `gsutil` CLI rather than gcsfs on purpose: gcsfs uses
    Application Default Credentials, which `gcloud auth login` does not create, so on
    this workstation gcsfs reads anonymously and 401s while gsutil works fine. On a TPU
    VM both work (ADC comes from the metadata server), so this is the strictly more
    portable of the two.
    """
    if path.startswith("gs://"):
        result = subprocess.run(["gsutil", "cat", path], capture_output=True, check=False)
        if result.returncode != 0:
            raise LoadError(f"gsutil cat {path} failed:\n{result.stderr.decode()[-2000:]}")
        return result.stdout
    local = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(local):
        raise LoadError(f"no such file: {local}")
    with open(local, "rb") as handle:
        return handle.read()


def load_stats(path: str, *, expect_sha256: str | None = None) -> OfficialStats:
    """`robotwin_uncond_3cam_384_dataset_stats.json` -> the four 14-vectors we consume.

    Any missing key raises: the official loader is `strict=False` everywhere and this one
    is not, because a defaulted normalization constant is an affine misalignment of the
    action space that produces no error and no log line.
    """
    return load_official_stats(read_bytes(path), source=path, expect_sha256=expect_sha256)


def restore_into(
    module: nnx.Module, subtrees: dict[str, str], *, param_dtype=None, device=None
) -> nnx.Module:
    """Restore named sub-dicts of `module`'s parameter tree from Orbax dirs.

    Args:
        module: a freshly constructed model — its parameter buffers are DELETED here.
        subtrees: `{"mot.video": "<root>/video", ...}` — dotted path into the pure param
            dict -> the Orbax dir holding exactly that subtree.
        param_dtype: cast the restore target (e.g. bfloat16 for deployment). None keeps
            whatever the module was built with.
        device: where the restored arrays land. Defaults to `jax.devices()[0]`.

    Every array in the module must be covered by exactly one subtree, and every restored
    tree must match its slice key-for-key: a partial restore leaves deleted buffers
    behind and would fail later, somewhere unhelpful.

    THE `device` ARGUMENT IS LOAD-BEARING — do not drop it back to a bare
    `ShapeDtypeStruct(shape, dtype)`. Without a sharding, orbax materialises every array
    HOST-side first and the staging buffers are never released: measured 2026-08-04 on
    the real 6.02 B checkpoint, the server sat on **27.3 GB of host RSS (26.8 GB
    Private_Dirty)** with the weights also resident on the GPU. On this 62 GB box that
    pushed the co-resident UMT5 encoder into swap, its next encode went from 20 s to
    over 300 s, and the eval client hit its 300 s socket timeout on the very first
    `set_task` — a closed-loop failure three layers away from the cause.

    Small-model tests cannot see this: a few hundred MB of leaked staging looks like
    nothing. `tests/test_inference_cache.py::test_restore_into_places_arrays_on_the_device`
    asserts placement directly instead of trying to detect the leak by size.
    """
    import orbax.checkpoint as ocp

    device = jax.devices()[0] if device is None else device
    sharding = jax.sharding.SingleDeviceSharding(device)

    graphdef, state, rest = nnx.split(module, nnx.Param, ...)
    pure = nnx.to_pure_dict(state)

    def descend(tree: dict, dotted: str) -> dict:
        # "" means the WHOLE tree, i.e. one Orbax dir holds every parameter. That is the
        # layout a training run writes (`save_params_bf16` stores
        # `nnx.to_pure_dict(params)` as a single StandardCheckpointer tree), as opposed to
        # the released checkpoint which `tools/convert_weights.py` splits into
        # video/action/proprio_encoder. Both carry the same logical keys; only the number
        # of directories differs.
        if dotted == "":
            return tree
        node = tree
        for part in dotted.split("."):
            if part not in node:
                raise LoadError(f"the model has no parameter subtree {dotted!r} (missing {part!r})")
            node = node[part]
        return node

    slices = {name: descend(pure, name) for name in subtrees}
    covered = sum(len(jax.tree.leaves(node)) for node in slices.values())
    total = len(jax.tree.leaves(pure))
    if covered != total:
        raise LoadError(
            f"the named subtrees cover {covered} of {total} parameter arrays; a partial "
            f"restore is never acceptable. Named: {sorted(subtrees)}"
        )

    targets = {
        name: jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(
                x.shape, param_dtype if param_dtype is not None else x.dtype, sharding=sharding
            ),
            node,
        )
        for name, node in slices.items()
    }
    # Free the random init BEFORE asking orbax for the real weights, so the two never
    # coexist. See the module docstring.
    for leaf in jax.tree.leaves(pure):
        leaf.delete()

    checkpointer = ocp.StandardCheckpointer()
    for name, path in subtrees.items():
        location = path if path.startswith("gs://") else os.path.abspath(os.path.expanduser(path))
        node = descend(pure, name)
        # List indices: `tools/convert_weights.py::_nest` writes `blocks` children under
        # the STRING keys "0".."29" (flax 0.10's spelling, which is also what every
        # existing public tree uses), while flax 0.12's `to_pure_dict` gives ints.
        # The trees are identical; orbax sees a structure mismatch. Try both spellings
        # rather than pick one, and report every failure if neither works — a loader
        # that guesses here fails minutes into a restore with an unhelpful message.
        errors = {}
        restored = None
        for style, target in (
            ("flax 0.10 string keys", stringify_tree_keys(targets[name])),
            ("flax 0.12 int keys", targets[name]),
        ):
            try:
                restored = restore_tree_key_types(
                    checkpointer.restore(location, target=target), targets[name]
                )
            except Exception as exc:  # noqa: BLE001 - every cause is reported below
                errors[style] = exc
                continue
            break
        if restored is None:
            raise LoadError(
                f"restoring {name} from {location} failed under both key spellings. The tree "
                f"must match this module exactly — no partial restores, no strict=False.\n"
                + "\n".join(f"  {style}: {type(exc).__name__}: {exc}" for style, exc in errors.items())
            ) from next(iter(errors.values()))
        node.clear()
        node.update(restored)

    nnx.replace_by_pure_dict(state, pure)
    return nnx.merge(graphdef, state, rest)


def load_fastwam(
    ckpt_root: str,
    config: Config,
    *,
    dtype=jnp.bfloat16,
    param_dtype=None,
    scan_layers: bool = False,
    seed: int = 0,
) -> FastWAM:
    """The converted official checkpoint -> a ready `FastWAM`.

    `param_dtype` defaults to `dtype`: the released weights are bf16 on disk and there is
    no reason to inflate them to fp32 on a 24 GB card.

    Layer layout follows the CHECKPOINT, because the two on-disk forms differ:

        released ckpt   <root>/{video,action,proprio_encoder}, blocks UNSCANNED
                        (a plain list of 30) -> build unrolled
        training export <root>/ as one tree, blocks STACKED on a leading 30
                        (`save_params_bf16` of a scan_layers=True training model,
                        e.g. mot.action.blocks.attn1.key.kernel is (30,1024,3072))
                        -> build scanned, restore straight in

    Matching the model to the file is what keeps the restore free: a scanned model fed
    an unscanned tree has to stack 30 blocks on-device and briefly holds two copies of a
    12 GB tree (24 GB — more than this card has), and the reverse has to unstack. Neither
    conversion happens if the layouts already agree, which they do once the layout is read
    off the directory instead of assumed.

    `scan_layers` as an explicit argument is still refused for the released layout, for
    exactly the reason above.
    """
    scanned_ckpt = not os.path.isdir(os.path.join(os.path.expanduser(ckpt_root), "video"))
    if scan_layers and not scanned_ckpt:
        raise LoadError(
            "scan_layers=True is not supported for the released (unscanned) layout. Its "
            "on-disk trees are UNSCANNED (a plain list of 30 blocks), so a scanned model "
            "needs the blocks stacked after the restore, and doing that on-device holds "
            "two copies of a 12 GB tree — 24 GB, which does not fit the card this server "
            "runs on. Either run unrolled (the default; compile time is paid once, before "
            "listen()) or stack on the host first and device_put the result."
        )
    param_dtype = dtype if param_dtype is None else param_dtype
    root = ckpt_root.rstrip("/")
    model = FastWAM.from_config(
        config, rngs=nnx.Rngs(seed), dtype=dtype, param_dtype=param_dtype,
        scan_layers=scanned_ckpt,
    )
    # Two on-disk layouts, told apart by what is actually there rather than by a flag:
    #   released ckpt   <root>/{video,action,proprio_encoder}   (tools/convert_weights.py)
    #   training export <root>/                                 (save_params_bf16: one
    #                                                            StandardCheckpointer tree)
    # They are mutually exclusive, so sniffing cannot pick wrong; and a flag would be one
    # more thing to get wrong on the command line for something the directory already
    # states unambiguously.
    if scanned_ckpt:
        subtrees = {"": root}
    else:
        subtrees = {
            "mot.video": f"{root}/video",
            "mot.action": f"{root}/action",
            "proprio_encoder": f"{root}/proprio_encoder",
        }
    return restore_into(model, subtrees, param_dtype=param_dtype)


def build_policy(
    *,
    ckpt_root: str,
    vae_ckpt: str,
    stats_path: str,
    config: Config,
    dtype=jnp.bfloat16,
    stats_sha256: str | None = None,
    **policy_kwargs,
) -> FastWAMPolicy:
    """Everything a model server needs, in the order that keeps peak memory lowest.

    Stats first (88 KB, and it is the cheapest thing to get wrong), then the VAE (~1 GB),
    then the 12 GB of DiT — so a typo in a path costs seconds, not a full restore.
    """
    stats = load_stats(stats_path, expect_sha256=stats_sha256)
    vae = load_vae(vae_ckpt, dtype=dtype, weights_dtype=dtype)
    model = load_fastwam(ckpt_root, config, dtype=dtype)
    shapes = config.shapes
    return FastWAMPolicy(
        model=model,
        vae=vae,
        stats=stats,
        action_horizon=shapes.action_tokens,
        canvas_hw=tuple(config.data.train.video_size),
        context_len=shapes.context_tokens - 1,  # 129 total, minus the proprio token
        **policy_kwargs,
    )


def describe_assets(ckpt_root: str, vae_ckpt: str, stats_path: str) -> str:
    """One line per asset, for the server banner. Reading a wrong path is the class of
    mistake that only shows up as a score, so the paths get printed, always."""
    return json.dumps(
        {"ckpt": ckpt_root, "vae": vae_ckpt, "stats": stats_path}, indent=None, sort_keys=True
    )
