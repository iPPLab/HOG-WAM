"""The training loop. Entry point: `python -m fastwam_tpu.training.train <task> [k.p=v ...]`.

Ported from
    codebases/TinyWAM-v4/tinywam/train.py — our loop: fused gradient accumulation (the
        whole accumulation folded into ONE jit so each gradient is consumed into a
        ZeRO-sharded accumulator and the replicated copy never materializes), the
        Checkpointer, consensus resume, the NaN guards, the in-loop VAE.
    reference/maxdiffusion/src/maxdiffusion/trainers/wan_trainer.py — skeleton only: the
        jit signature with in/out_shardings and donate_argnums, StepTraceAnnotation
        windows. maxdiffusion's Wan trainer has NO gradient accumulation, so that part is
        ours.

Recipe (spec §3.2; user choices live in one training YAML, immutable contracts in config.py/specs)
    AdamW(lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2), grad clip 1.0,
    5% linear warmup then cosine to lr*0.01, bf16 compute with fp32 master weights,
    seed 42, no EMA, no CFG.
    total_train_steps = ceil(ceil(len(ds) / global_batch) / grad_accum) * num_epochs
                                                                  (trainer.py:204-253)
    warmup_steps      = int(total_train_steps * 0.05)
    Official RoboTwin global batch = 16 x 64 GPUs = 1024. This slice has 16 chips, so the
    same effective batch is reached with gradient accumulation, stated explicitly in
    tpu.global_batch_size / tpu.micro_batch_size rather than left implicit.

Two rules that outrank convenience
    * `tpu.decay_steps` defaults to the FULL derived horizon, never to `tpu.steps`. A short
      run is a PREFIX of the real schedule, not its own compressed one. Getting this
      backwards once left a 100K run at the LR floor for 80% of its life.
    * `run_name` carries no timestamp. The gs:// path has to be identical across restarts
      or resume cannot find the run it is resuming.

Deliberately NOT ported
    * `.item()` on the loss every step (a host sync per optimizer step, spec §9.5 #26).
      Metrics stay device-resident and are read only on the log cadence.
    * The dataset's blanket `except -> pick another random sample` retry (spec §9.4 #20).
"""

import argparse
import json
import math
import sys
import time

import numpy as np

__all__ = ["main", "train", "build_model", "flops_per_sample", "official_total_steps"]


def _log(*parts) -> None:
    import jax

    print(f"[train p{jax.process_index()}]", *parts, flush=True)


# --------------------------------------------------------------------------- #
# model assembly
# --------------------------------------------------------------------------- #
def restore_tree_to_host(path: str, template: dict) -> dict:
    """Restore an Orbax tree into HOST numpy arrays, tolerating either key style.

    Two things make the model loaders in `models/` unusable from a multi-process trainer,
    and both are fixed here rather than by changing them (they are also the eval path, and
    single-process is where they are correct):

    * Under `jax.distributed`, orbax refuses a target of bare `ShapeDtypeStruct`s —
      "sharding passed to deserialization should be specified, concrete and an instance of
      jax.sharding.Sharding. Got None" — because in a multi-process world it will not guess
      where an array belongs. Asking for `np.ndarray` sidesteps the question entirely: host
      memory has no sharding. The arrays are placed on the mesh later, deliberately, by
      `sharding.shard_params`.
    * `PyTreeCheckpointHandler` reads without consulting `commit_success.txt`. The shared
      Wan2.2 base trees were written by an older Orbax version and are restored through
      the compatibility loader below.
      and carry no marker; orbax 0.12 calls that "incomplete" and refuses. The team's ruling
      is not to rewrite those trees, since several projects read them.

    Both key spellings are tried — flax 0.10 wrote `blocks` children under "0".."29",
    flax 0.12 uses ints — and every failure is reported if neither works.
    """
    import jax
    import orbax.checkpoint as ocp
    from etils import epath

    from ..models.wan.tree_keys import restore_tree_key_types, stringify_tree_keys

    handler = ocp.PyTreeCheckpointHandler()
    location = epath.Path(path)
    errors = {}
    for style, target in (
        ("flax 0.10 string keys", stringify_tree_keys(template)),
        ("flax 0.12 int keys", template),
    ):
        restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), target)
        try:
            restored = handler.restore(
                location, args=ocp.args.PyTreeRestore(item=target, restore_args=restore_args)
            )
        except Exception as exc:  # noqa: BLE001 - every cause is reported below
            errors[style] = exc
            continue
        return restore_tree_key_types(restored, template)

    raise ValueError(
        f"restoring {path} to host memory failed under both key spellings. The tree must "
        "match the target exactly — no partial restores.\n"
        + "\n".join(f"  {style}: {type(exc).__name__}: {exc}" for style, exc in errors.items())
    ) from next(iter(errors.values()))


def build_model(
    config, *, dtype, param_dtype, scan_layers: bool, remat_policy: str, seed: int,
    load_weights: bool = True,
):
    """A FastWAM with the official initialisation, built on the HOST.

    video expert    <- the shared converted Wan2.2-TI2V-5B DiT (`weights.video_dit_orbax`)
    action expert   <- the interpolation init (`weights.action_dit_init`): the same Wan
                       weights 1-D linearly interpolated down to width 1024 with the
                       sqrt(3) alpha rescale. NOT random — spec, preprocess_action_dit_backbone.py.
    proprio_encoder <- random, as the official leaves it. 61,440 params.

    The tower checkpoints are UNSCANNED on disk (a plain list of 30 blocks), so the restore
    target is always the unscanned tree and the blocks are stacked afterwards when
    `scan_layers=True` — which is what `nnx.scan` slices.

    `load_weights=False` skips both restores and leaves the towers at their random init.
    That is a throughput/memory harness only — random weights have exactly the FLOPs and
    exactly the shapes of trained ones, and it cuts startup from ~20 minutes to ~2. Nothing
    it produces is a statement about learning. The CLI gates it behind a short run so it
    cannot be the thing that quietly trained a 29,365-step run from scratch.

    Returns `(graphdef, state, host_tree)`. The weights come back as HOST numpy and are
    deliberately NOT converted to device arrays here: the caller places them on the mesh in
    their ZeRO layout, and a round trip through CPU-resident `jax.Array`s would cost a
    second full 24 GB copy for nothing. `state` is the (emptied) nnx.State the caller
    refills with the placed arrays.
    """
    import jax
    from flax import nnx

    from ..models.action_dit import ActionDiT
    from ..models.fastwam import FastWAM
    from ..models.wan.transformer import WanVideoDiT

    cpu = jax.local_devices(backend="cpu")[0]
    # ``rgb_to_hog`` is deliberately a two-tower model: the ordinary video
    # expert receives HOG latents and predicts future HOG latents, while the
    # action expert and all MoT dimensions remain unchanged.  The existing
    # recipes retain the auxiliary HOG tower for backwards compatibility.
    include_hog = (config.model.visual_input_mode == "rgb"
                   and config.model.training_towers == "rgb_hog_action")
    with jax.default_device(cpu):
        model = FastWAM.from_config(
            config,
            rngs=nnx.Rngs(seed),
            dtype=dtype,
            param_dtype=param_dtype,
            include_hog=include_hog,
            scan_layers=scan_layers,
            remat_policy=remat_policy,
        )
        graphdef, state = nnx.split(model)
        pure = nnx.to_pure_dict(state)
        if config.data.format == "libero_hdf5":
            # Match the new proprio projection's random initialization across
            # baseline/HOG even though constructing the HOG expert consumes RNGs.
            projection = nnx.Linear(8, config.model.video_dit_config.text_dim,
                rngs=nnx.Rngs(seed + 102), dtype=dtype, param_dtype=param_dtype,
                kernel_init=nnx.initializers.xavier_uniform())
            old_projection = pure["proprio_encoder"]
            for leaf in jax.tree.leaves(old_projection):
                if hasattr(leaf, "delete"):
                    leaf.delete()
            pure["proprio_encoder"] = nnx.to_pure_dict(nnx.state(projection, nnx.Param))

        towers = [
            ("video", WanVideoDiT, config.model.weights.video_dit_orbax, config.model.video_dit_config),
            ("action", ActionDiT, config.model.weights.action_dit_init, config.model.action_dit_config),
        ]
        if include_hog and config.model.hog_dit_init == "copy_rgb":
            # The historical HOG branch has a separately trainable auxiliary
            # expert initialized as a copy of RGB.  It is not part of the new
            # rgb_to_hog ablation.
            towers.insert(
                1,
                ("hog", WanVideoDiT, config.model.weights.video_dit_orbax, config.model.video_dit_config),
            )
        elif include_hog:
            _log(
                f"HOG DiT RANDOM INIT (seed={seed}): retaining the independently "
                "initialized HOG expert; RGB and Action pretrained restores remain enabled"
            )
        if not load_weights:
            _log("RANDOM INIT: all three experts left at random init — timing harness only")
            towers = ()
        for name, tower_cls, path, tower_config in towers:
            _log(
                f"{'copying' if name == 'hog' else 'loading'} the {name} expert "
                f"{'from the resident RGB tree' if name == 'hog' else f'from {path}'}"
            )
            started = time.time()
            if name == "hog":
                # The HOG expert is initialised as an exact RGB copy.  Reuse the already
                # resident numpy leaves: this performs ONE GCS restore of the 5 B tree,
                # not a second expensive read. Device placement creates independent HOG
                # arrays, so the two towers train separately after step 0.
                loaded = pure["mot"]["video"]
            else:
                restore_config = tower_config
                if name == "action" and config.data.format == "libero_hdf5":
                    # The resident interpolation checkpoint has RoboTwin's 14-D
                    # random I/O layers. Restore its shared backbone strictly,
                    # then initialize LIBERO's 7-D I/O layers, never truncate it.
                    from dataclasses import replace
                    restore_config = replace(tower_config, action_dim=14)
                unscanned = nnx.eval_shape(
                    lambda cls=tower_cls, cfg=restore_config: cls.from_config(
                        cfg, rngs=nnx.Rngs(0), dtype=dtype, param_dtype=param_dtype, scan_layers=False
                    )
                )
                template = nnx.to_pure_dict(nnx.state(unscanned, nnx.Param))
                loaded = restore_tree_to_host(path, template)
                if name == "action" and config.data.format == "libero_hdf5":
                    for layer, n_in, n_out, layer_seed in (
                        ("action_encoder", 7, tower_config.hidden_dim, seed + 100),
                        ("proj_out", tower_config.hidden_dim, 7, seed + 101),
                    ):
                        linear = nnx.Linear(n_in, n_out, rngs=nnx.Rngs(layer_seed),
                                            dtype=dtype, param_dtype=param_dtype,
                                            kernel_init=nnx.initializers.xavier_uniform())
                        loaded[layer] = nnx.to_pure_dict(nnx.state(linear, nnx.Param))
                    _log("LIBERO: restored the full action backbone; independently initialized 7-D I/O")
                if scan_layers:
                    per_layer = [loaded["blocks"][index] for index in range(tower_config.num_layers)]
                    loaded = {
                        **{k: v for k, v in loaded.items() if k != "blocks"},
                        "blocks": jax.tree.map(lambda *layers: np.stack(layers, axis=0), *per_layer),
                    }
            target = pure["mot"][name]
            _assert_same_keys(target, loaded, f"mot.{name}", path)
            # Drop the random buffers as we go, so a 5 B tower's random init and its
            # restored copy are never both resident.
            for leaf in jax.tree.leaves(target):
                if hasattr(leaf, "delete"):
                    leaf.delete()
            target.clear()
            target.update(jax.tree.map(lambda x: np.asarray(x, np.dtype(param_dtype)), loaded))
            del loaded
            _log(f"  {name} expert loaded in {time.time() - started:.1f}s")

        # Whatever is still a CPU jax.Array (the proprio encoder always, both towers under
        # load_weights=False) becomes numpy, so the whole tree is one kind of thing before
        # it goes to the mesh.
        pure = jax.tree.map(lambda x: np.asarray(x, np.dtype(param_dtype)), pure)
        if include_hog and load_weights:
            # Small, real-parameter receipt: avoid hashing a second 5B checkpoint.
            # Verify a nonconstant matrix, not zero biases shared by random models.
            import hashlib
            def signature(tree):
                matrix = tree["patch_embedding"]["kernel"]
                return hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
            video_hash = signature(pure["mot"]["video"])
            hog_hash = signature(pure["mot"]["hog"])
            expected_equal = config.model.hog_dit_init == "copy_rgb"
            if (video_hash == hog_hash) != expected_equal:
                raise ValueError("HOG DiT initialization does not match the selected policy")
            _log("HOG_INIT_AUDIT " + json.dumps({
                "mode": config.model.hog_dit_init, "seed": seed,
                "parameter": "patch_embedding/kernel", "rgb_sha256": video_hash,
                "hog_sha256": hog_hash, "equal": video_hash == hog_hash,
                "rgb_action_pretrained": True,
            }, sort_keys=True))
            for index in range(config.model.video_dit_config.num_layers):
                def block_signature(tree):
                    blocks = tree["blocks"]
                    query = (blocks["attn1"]["query"]["kernel"][index] if scan_layers
                             else blocks[index]["attn1"]["query"]["kernel"])
                    return hashlib.sha256(np.ascontiguousarray(query).tobytes()).hexdigest()
                rgb_block = block_signature(pure["mot"]["video"])
                hog_block = block_signature(pure["mot"]["hog"])
                if (rgb_block == hog_block) != expected_equal:
                    raise ValueError(f"HOG block {index} violates the initialization policy")
            _log(f"HOG_INIT_AUDIT: all {config.model.video_dit_config.num_layers} "
                 f"DiT blocks verified ({config.model.hog_dit_init})")
    return graphdef, state, pure


def _assert_same_keys(target: dict, loaded: dict, where: str, path: str) -> None:
    """Same structure, same shapes, both directions.

    The official loader is `strict=False` everywhere; this one is not. A silently skipped
    tensor leaves part of a 5 B tower at its random init, and the only symptom is a score
    weeks later.
    """

    def flat(tree, prefix=""):
        out = {}
        for key, value in tree.items():
            here = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                out.update(flat(value, here))
            else:
                out[here] = tuple(value.shape)
        return out

    want, got = flat(target), flat(loaded)
    if want != got:
        missing = sorted(set(want) - set(got))
        extra = sorted(set(got) - set(want))
        mismatched = {k: (want[k], got[k]) for k in set(want) & set(got) if want[k] != got[k]}
        raise ValueError(
            f"{path} does not match this model's {where} subtree.\n"
            f"  missing from the checkpoint: {missing[:8]}{'...' if len(missing) > 8 else ''}\n"
            f"  present but unexpected:      {extra[:8]}{'...' if len(extra) > 8 else ''}\n"
            f"  shape mismatches:            {dict(list(mismatched.items())[:8])}"
        )


def load_frozen_vae(config, *, dtype):
    """The frozen Wan2.2 VAE, built on the host. Encoding happens inside the training step.

    Not `vae.py::load_vae`, for the same two reasons `build_model` does not use the tower
    loaders: that path goes through `StandardCheckpointer`, which refuses the shared
    wan22_5b VAE tree because it carries no `commit_success.txt`, and it would restore into
    unsharded jax arrays, which orbax rejects under `jax.distributed`. Everything else —
    the sidecar-driven config, the exact-tree requirement — is the same, and the sidecar is
    still what decides the geometry, so a wrong checkpoint cannot be loaded quietly.
    """
    import jax
    from flax import nnx

    from ..models.wan.vae import build_vae_from_sidecar, read_vae_sidecar

    path = config.model.weights.vae_orbax
    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        vae = build_vae_from_sidecar(read_vae_sidecar(path), dtype=dtype, weights_dtype=dtype)
        graphdef, state = nnx.split(vae)
        template = nnx.to_pure_dict(state)
        restored = restore_tree_to_host(path, template)
        _assert_same_keys(template, restored, "vae", path)
        # The restore returns the ON-DISK dtype, not the module's. Casting here is what
        # keeps the encode in bf16; leaving it would silently run the VAE in fp32, which is
        # slower and larger and would not show up as an error anywhere.
        restored = jax.tree.map(lambda x: np.asarray(x, np.dtype(dtype)), restored)
        nnx.replace_by_pure_dict(state, restored)
        return nnx.merge(graphdef, state)


# --------------------------------------------------------------------------- #
# FLOPs bookkeeping
# --------------------------------------------------------------------------- #
def flops_per_sample(config) -> float:
    """Analytic training FLOPs per sample (forward + backward), for MFU.

    2 FLOPs per multiply-accumulate, backward = 2x forward. Recomputation from remat is
    deliberately EXCLUDED: MFU measures useful model FLOPs, so remat lowers it by exactly
    what it costs, which is the number worth seeing. The VAE is not counted either — it is
    frozen and its cost shows up as wall clock, which is where the smoke measures it.

    The joint self-attention is charged ONCE over the concatenated sequence: both experts
    project into the shared 3072-wide attention space and attend together, so it belongs to
    neither tower's block.

    The action length is `train_action_tokens`, i.e. `flow.tau_copies` copies included —
    they are real work the chips do. Charging the data's 32 rows instead would report a
    copies run as *more* efficient than it is, which is backwards: the copies are the
    thing whose cost we want to see.
    """
    shapes = config.shapes
    video = config.model.video_dit_config
    action = config.model.action_dit_config
    inner = video.attn_inner_dim  # 3072, shared by both experts
    s_v, s_a = shapes.video_tokens, shapes.train_action_tokens
    s_ctx = shapes.context_tokens

    def tower(hidden: int, ffn: int, tokens: int) -> float:
        qkv = 3 * 2 * tokens * hidden * inner  # q/k/v into the shared attention width
        out = 2 * tokens * inner * hidden  # and back out
        cross_q = 2 * tokens * hidden * inner
        cross_kv = 2 * 2 * s_ctx * video.text_dim * inner
        cross_scores = 2 * 2 * tokens * s_ctx * inner
        cross_out = 2 * tokens * inner * hidden
        mlp = 2 * 2 * tokens * hidden * ffn
        return qkv + out + cross_q + cross_kv + cross_scores + cross_out + mlp

    has_hog = (config.model.visual_input_mode == "rgb"
               and config.model.training_towers == "rgb_hog_action")
    n_video_towers = 2 if has_hog else 1
    per_layer = n_video_towers * tower(video.hidden_dim, video.ffn_dim, s_v)
    per_layer += tower(action.hidden_dim, action.ffn_dim, s_a)
    joint_tokens = (2 * s_v + s_a) if has_hog else (s_v + s_a)
    per_layer += 2 * 2 * joint_tokens**2 * inner  # joint QK^T and (attn)V
    return 3.0 * video.num_layers * per_layer


_PEAK_BF16_TFLOPS = {"v4": 275.0, "v5 lite": 197.0, "v5e": 197.0, "v6 lite": 918.0, "v6e": 918.0}


def peak_tflops(config) -> float:
    """bf16 peak per chip. Auto-detected by device_kind unless the config pins it.

    Keep the override unset unless a known peak is needed for an unlisted device.
    Reusing a hardware-specific override across TPU generations distorts MFU.
    """
    import jax

    pinned = config.tpu.mfu_peak_tflops_per_chip
    if pinned is not None:
        return float(pinned)
    kind = jax.devices()[0].device_kind.lower()
    for pattern, value in _PEAK_BF16_TFLOPS.items():
        if pattern in kind:
            return value
    _log(f"WARNING: unknown device_kind {kind!r}; assuming the v4 bf16 peak (275 TFLOP/s)")
    return 275.0


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def official_total_steps(train_windows: int, config) -> int:
    """`ceil(ceil(len(train_ds) / global_batch) / grad_accum) * num_epochs` (trainer.py:204-253).

    `grad_accum` here is the OFFICIAL `gradient_accumulation_steps` (1), not ours. Theirs
    splits an already-global batch across steps and therefore changes the step count; ours
    splits one global batch across microsteps inside a step and does not. Feeding our 16
    into this formula would divide the whole schedule by 16.
    """
    per_epoch = math.ceil(train_windows / config.tpu.global_batch_size)
    per_epoch = math.ceil(per_epoch / config.gradient_accumulation_steps)
    return per_epoch * config.num_epochs


# The rank of every per-sample tensor a microbatch carries, so the batch PartitionSpecs
# can be built without re-deriving shapes. Keys and ranks match `models/fastwam.py`'s
# REQUIRED_BATCH_KEYS + OPTIONAL_BATCH_KEYS; `build_inputs` raises on anything else.
MICRO_RANKS = (
    ("latents", 5),
    ("hog_latents", 5),
    ("action", 3),
    ("proprio", 3),
    ("context", 3),
    ("context_mask", 2),
    ("image_is_pad", 2),
    ("action_is_pad", 2),
)


def micro_ranks(include_hog: bool) -> tuple[tuple[str, int], ...]:
    """Batch ranks for the selected objective.

    The two-tower ``rgb_to_hog`` path has no auxiliary HOG latent key; keeping
    that key out of the jit pytree is important because ``training_loss`` rejects
    unknown batch fields for the classic model.  The legacy three-tower recipes
    retain the historical constant above for compatibility with existing tests.
    """
    if include_hog:
        return MICRO_RANKS
    return tuple(item for item in MICRO_RANKS if item[0] != "hog_latents")


def _microstep_scalars(loss, aux):
    import jax.numpy as jnp

    return {
        "loss": loss,
        "loss_video": aux["loss_video"],
        "loss_hog": aux["loss_hog"],
        "loss_action": aux["loss_action"],
        "t_video": jnp.mean(aux["t_video"]),
        "t_hog": jnp.mean(aux["t_hog"]),
        "t_action": jnp.mean(aux["t_action"]),
    }


def make_microstep(graphdef, zero_n: int | None, *, copy_lengths=None, jit_kwargs=None):
    """One accumulation microstep: gradient computed and CONSUMED into a donated accumulator.

    This is the default path, and the reason is memory. The fused alternative below folds
    all `accum` microsteps into one program so XLA can coalesce and overlap the cross-chip
    reductions; at the official batch that is 16 unrolled copies of a 6 B forward+backward
    in a single HLO module, and it was measured on v4-32 needing **32.16 G of temporaries
    against 30.75 G of HBM** — 4.6% over, and no config knob closes that gap. Here each
    microstep is its own small program, so only one gradient tree is ever live beside the
    accumulator.

    The gradient is consumed INSIDE the jit rather than returned. A grad-returning step
    forces XLA to materialise the full replicated tree (24 GB at 6 B params) before the
    caller can reduce it; folding it into the donated, ZeRO-sharded accumulator lowers the
    backward as a reduce-scatter straight into the 1.5 GB/chip layout.

    `reset` (True on the first microstep) overwrites instead of adding, so the accumulator
    never needs zeroing between optimizer steps.

    Returns `microstep(grad_acc, params, key, micro, reset) -> (grad_acc, scalars)`.
    """
    import jax
    from flax import nnx

    from . import sharding

    def body(grad_acc, params, key, micro, reset):
        def loss_and_aux(param_tree, k, m):
            return nnx.merge(graphdef, param_tree).training_loss(m, k, copy_lengths=copy_lengths)

        (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(params, key, micro)
        grads = sharding.shard_grads(grads, zero_n)
        grad_acc = jax.lax.cond(
            reset,
            lambda _: grads,
            lambda _: jax.tree.map(lambda a, g: a + g, grad_acc, grads),
            operand=None,
        )
        return grad_acc, _microstep_scalars(loss, aux)

    # params are NOT donated: they are read by every microstep of the accumulation.
    return jax.jit(body, donate_argnums=(0,), **(jit_kwargs or {}))


def make_apply_step(tx, accum: int, *, jit_kwargs=None):
    """The optimizer half of the non-fused path: mean the accumulator, then update once.

    The division by `accum` happens inside the jit; as an eager tree.map between the loop
    and this call it was ~700 extra tiny dispatches per optimizer step. The accumulator is
    returned so its donated buffer is reused — the next step's first microstep overwrites
    it rather than allocating a second 1.5 GB/chip tree.
    """
    import jax
    import optax

    inv = 1.0 / accum

    def body(params, opt, grad_acc):
        grads = jax.tree.map(lambda g: g * inv, grad_acc)
        updates, opt2 = tx.update(grads, opt, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt2, {"grad_norm": optax.global_norm(grads)}, grads

    return jax.jit(body, donate_argnums=(0, 1, 2), **(jit_kwargs or {}))


def make_fused_step(
    graphdef, tx, accum: int, zero_n: int | None, *, copy_lengths=None, jit_kwargs=None
):
    """All `accum` microsteps plus the optimizer apply, UNROLLED inside ONE jit.

    Same math as a Python loop over a per-microstep jit, but XLA sees the entire gradient
    add-tree in one program, so it can coalesce the cross-chip reductions and overlap
    microstep j's reduce with microstep j+1's compute — a per-microstep jit boundary is a
    sync point that forbids both. It stays in GSPMD auto (no shard_map), which is what lets
    remat compose; the shard_map-based deferral TinyWAM also carries does NOT compose with
    jax.checkpoint and is a documented dead end (`project_v6e_gradaccum_defer`).

    Not `lax.scan` over the microsteps: XLA's AllReduceReassociate does not cross scan
    iterations, so scanning would give back exactly the coalescing this exists for. (The 30
    MoT layers are still scanned — that is a different loop, inside the model.)

    Every gradient is reduce-scattered into the ZeRO accumulator by `shard_grads`, so the
    replicated gradient tree is never materialized. Verify the emitted reduction count on
    the LOWERED HLO rather than assuming XLA coalesced.

    Returns `step(params, opt_state, keys, micros) -> (params, opt_state, metrics)`.
    """
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    from . import sharding

    inv = 1.0 / accum

    def loss_and_aux(param_tree, key, micro):
        return nnx.merge(graphdef, param_tree).training_loss(micro, key, copy_lengths=copy_lengths)

    def body(param_tree, opt, keys, micros):
        grad_acc = None
        loss_sum = None
        for j in range(accum):
            (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
                param_tree, keys[j], micros[j]
            )
            grads = sharding.shard_grads(grads, zero_n)
            grad_acc = grads if grad_acc is None else jax.tree.map(jnp.add, grad_acc, grads)
            scalars = {
                "loss": loss,
                "loss_video": aux["loss_video"],
                "loss_hog": aux["loss_hog"],
                "loss_action": aux["loss_action"],
                "t_video": jnp.mean(aux["t_video"]),
                "t_hog": jnp.mean(aux["t_hog"]),
                "t_action": jnp.mean(aux["t_action"]),
            }
            loss_sum = scalars if loss_sum is None else {k: loss_sum[k] + v for k, v in scalars.items()}
        grad_acc = jax.tree.map(lambda g: g * inv, grad_acc)
        updates, opt2 = tx.update(grad_acc, opt, param_tree)
        params2 = optax.apply_updates(param_tree, updates)
        metrics = {k: v * inv for k, v in loss_sum.items()}
        metrics["grad_norm"] = optax.global_norm(grad_acc)
        # The NaN guard runs on the checkpoint cadence (`check_finite`), not here: a
        # per-step full-tree finiteness reduction would undo the hot-path savings.
        return params2, opt2, metrics

    return jax.jit(body, donate_argnums=(0, 1), **(jit_kwargs or {}))


def build_schedule(config, decay_steps: int):
    """5% linear warmup then cosine to lr*0.01 (trainer.py:204-253). -> (schedule, warmup).

    optax's `decay_steps` is the TOTAL including warmup and it runs the cosine over
    `decay_steps - warmup_steps`, which is exactly the reference's
    `CosineAnnealingLR(T_max=total - warmup)`.

    `init_value` is `lr/warmup_steps`, not 0: the reference warms up with
    `LinearLR(start_factor=1/warmup_steps)`, so step 0 already sees one warmup unit.
    """
    import optax

    warmup = max(1, int(decay_steps * config.optimizer.warmup_ratio))
    peak = float(config.learning_rate)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=peak / warmup,
        peak_value=peak,
        warmup_steps=warmup,
        decay_steps=decay_steps,
        end_value=peak * config.optimizer.min_lr_ratio,
    )
    return schedule, warmup


# --------------------------------------------------------------------------- #
# the loop
# --------------------------------------------------------------------------- #
def train(config, *, dry_run: bool = False, load_weights: bool = True,
          benchmark_steps: int | None = None, benchmark_output: str | None = None,
          prefetch_host_batch: bool = False) -> None:
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    from ..data.loader import RobotwinTwpDataset
    from ..data.pool import open_filesystem, stable_seed
    from ..models.hog import hog_image
    from . import checkpoint as ckpt_mod
    from . import remat as remat_mod
    from . import sharding

    if benchmark_steps is not None:
        if not 12 <= benchmark_steps <= 100:
            raise ValueError("benchmark_steps must be between 12 and 100")
        if not config.tpu.resume_from or not config.tpu.resume or not load_weights:
            raise ValueError("A benchmark requires an explicit, real resumable checkpoint")
        if not benchmark_output:
            raise ValueError("A benchmark requires a separate JSON output path")
    if prefetch_host_batch and config.data.format != 'libero_hdf5':
        raise ValueError('Host prefetch currently supports LIBERO only')

    t_start = time.time()
    is_chief = jax.process_index() == 0
    n_dev = jax.device_count()
    n_hosts = jax.process_count()
    _log(
        f"jax {jax.__version__} process {jax.process_index()}/{n_hosts} "
        f"devices={n_dev} ({jax.devices()[0].device_kind})"
    )

    dtypes = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}
    compute_dtype = dtypes[config.tpu.compute_dtype]
    param_dtype = dtypes[config.tpu.param_dtype]
    policy = remat_mod.resolve(config.tpu.remat_policy)

    # ---- mesh ---------------------------------------------------------------
    mesh = sharding.make_mesh(config)
    sharding.install(
        mesh,
        constraints=config.tpu.sharding_constraints,
        activations=config.tpu.shard_activations,
        axis_policy=config.tpu.param_axis_policy,
    )
    zero_n = sharding.zero_degree(mesh) if config.tpu.shard_params else None
    _log(sharding.describe_mesh(mesh))
    if not config.tpu.sharding_constraints:
        _log(
            "WARNING: tpu.sharding_constraints=false — gradients will be all-reduced into a "
            "REPLICATED tree instead of reduce-scattered. At 6 B params that is 24 GB/chip of "
            "gradient alone. Debug only."
        )

    # ---- batch geometry -----------------------------------------------------
    global_bs = config.tpu.global_batch_size
    micro_bs = config.tpu.micro_batch_size
    accum = config.tpu.grad_accum
    if micro_bs % n_dev:
        raise ValueError(f"tpu.micro_batch_size {micro_bs} must divide over {n_dev} devices")
    if micro_bs % n_hosts:
        raise ValueError(f"tpu.micro_batch_size {micro_bs} must divide over {n_hosts} hosts")
    host_bs = micro_bs // n_hosts
    _log(
        f"batch: global {global_bs} = {micro_bs} micro x {accum} accum "
        f"({host_bs}/host, {micro_bs // n_dev}/chip)"
    )
    official_batch = 1024  # released recipe: 16 per process x 64 GPU processes
    if global_bs != official_batch:
        _log(
            f"RECIPE DEVIATION: global batch {global_bs} != the official {official_batch}. "
            "Every loss number from this run is at a different effective batch than the "
            "reference, and the LR was tuned for the reference. Deliberate here; do not "
            "compare it to an official-batch run without saying so."
        )

    # ---- action tau copies ---------------------------------------------------
    copy_lengths = config.shapes.copy_lengths
    if len(copy_lengths) > 1:
        _log(
            f"RECIPE DEVIATION: flow.tau_copies — the action stream carries "
            f"{len(copy_lengths)} independent copies of the chunk, lengths {copy_lengths}, "
            f"{config.shapes.train_action_tokens} rows against the data's "
            f"{config.shapes.action_tokens}. The action objective is sampled at "
            f"{len(copy_lengths)}x the flow times per step, and the loss is averaged over "
            "copies. The reference draws ONE tau per sample; its action loss is not this "
            "run's action loss, so do not put the two in one table."
        )

    # ---- data ---------------------------------------------------------------
    node = config.data.train
    l3 = config.data.l3
    fs = open_filesystem(l3.prefix)
    if config.data.format == "libero_hdf5":
        from ..data.libero import LiberoHdf5Dataset
        dataset_cls = LiberoHdf5Dataset
    else:
        dataset_cls = RobotwinTwpDataset
    dataset = dataset_cls(
        l3.prefix,
        stats_blob=fs.cat_file(l3.stats_path.removeprefix("gs://")),
        stats_source=l3.stats_path,
        stats_sha256=l3.stats_sha256,
        num_frames=node.num_frames,
        action_video_freq_ratio=node.action_video_freq_ratio,
        video_size=node.video_size,
        action_shift=l3.action_shift,
        source_color_order=l3.source_color_order,
        text_mode=l3.text_mode,
        text_dir=l3.text_dir,
        context_len=node.context_len,
        task_filter=l3.task_filter,
        val_set_proportion=node.val_set_proportion,
        split="train",
        episode_weighting=l3.episode_weighting,
        pool_size=l3.pool_size,
        pool_rate=l3.pool_rate,
        pool_prefetch_threads=l3.pool_prefetch_threads,
        decode_threads=l3.decode_threads,
        seed=config.seed,
        host_id=jax.process_index(),
        n_hosts=n_hosts,
        fs=fs,
        expect_index_fingerprint=l3.index_fingerprint,
    )
    if l3.text_mode == "baked":
        _log(
            "WARNING: data.l3.text_mode='baked' — the conditioning is the BARE instruction "
            "through Wan2.1's UMT5, not the official template through Wan2.2's. Acceptable for "
            "a smoke; a B-line run must use 'sidetable'."
        )

    # ---- schedule -----------------------------------------------------------
    train_windows = int(dataset.index.num_windows)
    derived_total = official_total_steps(train_windows, config)
    steps = config.tpu.steps or derived_total
    decay_steps = config.tpu.decay_steps or derived_total
    schedule, warmup_steps = build_schedule(config, decay_steps)
    _log(
        f"schedule: {train_windows:,} train windows -> official total {derived_total:,} steps "
        f"({config.num_epochs} epochs at global batch {global_bs}); running {steps:,} of them; "
        f"decay_steps={decay_steps:,} warmup={warmup_steps:,}; "
        f"lr {config.learning_rate:g} -> {config.learning_rate * config.optimizer.min_lr_ratio:g}"
    )
    if steps != decay_steps:
        _log(
            f"NOTE: this run stops at step {steps:,} of a {decay_steps:,}-step cosine, so it is a "
            "PREFIX of the full schedule and the LR never reaches the floor. That is the point — "
            "a short run must see the same LR at step N that the full run sees at step N."
        )

    # ---- run identity -------------------------------------------------------
    # No timestamp, ever: resume needs the same gs:// path on every relaunch.
    run_name = config.tpu.run_name or f"fastwam-uncond-rt3cam-b{global_bs}-lr{config.learning_rate:g}"
    run_root = f"{config.tpu.gcs.run_dir.rstrip('/')}/{run_name}"
    _log(f"run_name={run_name} -> {run_root}")

    # ---- model + optimizer --------------------------------------------------
    stage = time.time()
    graphdef, param_state, host_tree = build_model(
        config,
        dtype=compute_dtype,
        param_dtype=param_dtype,
        scan_layers=config.tpu.scan_layers,
        remat_policy=policy.model_policy(),
        seed=config.seed,
        load_weights=load_weights,
    )
    # Everything in this model is a trainable Param — there are no batch statistics and no
    # stored RNG state. Asserting it here means an integer key can never reach AdamW, where
    # it would be "updated" into nonsense without anything raising.
    non_float = [
        str(x.dtype) for x in jax.tree.leaves(host_tree) if not jnp.issubdtype(x.dtype, jnp.floating)
    ]
    if non_float:
        raise ValueError(
            f"the parameter tree contains non-floating leaves {sorted(set(non_float))}; the "
            "optimizer would silently corrupt them. Split them out before tx.init."
        )
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(host_tree))
    tower_label = ("RGB→HOG + Action (two-tower)" if config.model.visual_input_mode == "rgb_to_hog"
                   else "RGB + HOG + Action" if config.model.training_towers == "rgb_hog_action"
                   else "RGB + Action (baseline)")
    _log(
        f"model: {n_params / 1e9:.4f} B params ({tower_label}), "
        f"built + loaded in {time.time() - stage:.1f}s"
    )

    tx = optax.chain(
        optax.clip_by_global_norm(float(config.max_grad_norm)),
        optax.adamw(
            schedule,
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            eps=config.optimizer.eps,
            weight_decay=config.weight_decay,
        ),
    )
    # ---- placement ----------------------------------------------------------
    # Parameters go to the mesh first, then the optimizer state is initialised FROM the
    # placed tree, so `tx.init` produces sharded moments directly instead of building a
    # 48 GB host copy that then has to be moved.
    stage = time.time()
    nnx.replace_by_pure_dict(
        param_state, sharding.shard_params(host_tree, mesh, shard=config.tpu.shard_params)
    )
    params = param_state
    del host_tree
    replicated = NamedSharding(mesh, P())
    p_shard = jax.tree.map(lambda x: x.sharding, params)

    # `tx.init` is zeros_like plus a scalar count. Its output shardings MUST be given
    # explicitly: `zeros_like` reads only shape and dtype, so there is no data dependence
    # for GSPMD to propagate a sharding along, and left to itself XLA builds both Adam
    # moments REPLICATED — 2 x 24 GB of temporaries against 30.75 GB of HBM, which is
    # exactly how this failed the first time it ran on the pod. Deriving the spec from each
    # leaf's shape reproduces the parameter layout, because mu and nu are shaped like the
    # parameters; the scalar count falls out as replicated.
    def opt_spec(leaf):
        if zero_n is None:
            return replicated
        return NamedSharding(mesh, sharding.largest_axis_spec(leaf, zero_n))

    o_shard = jax.tree.map(opt_spec, jax.eval_shape(tx.init, params))
    opt_state = jax.jit(tx.init, out_shardings=o_shard)(params)
    resident_gb = 4 * n_params * (3 if config.tpu.param_dtype == "float32" else 2)
    _log(
        f"params + optimizer state placed in {time.time() - stage:.1f}s "
        f"(~{resident_gb / 1e9 / max(1, zero_n or 1):.1f} G/chip resident at ZeRO "
        f"{zero_n or 1}-way)"
    )

    vae = load_frozen_vae(config, dtype=compute_dtype)
    vae_graphdef, vae_state = nnx.split(vae)
    nnx.replace_by_pure_dict(vae_state, sharding.replicate(nnx.to_pure_dict(vae_state), mesh))
    vae = nnx.merge(vae_graphdef, vae_state)
    _log(f"frozen VAE replicated ({config.tpu.compute_dtype}) from {config.model.weights.vae_orbax}")

    # ---- the jitted pieces --------------------------------------------------
    @jax.jit
    def encode_pixels(pixels):
        """[B,3,9,384,320] in [-1,1] -> normalized latents [B,48,3,24,20] fp32.

        Inside the training step, as the official does. The VAE is frozen and per-sample
        independent, so this is elementwise over the batch: batch-sharded pixels encode to
        batch-sharded latents with no cross-chip collectives.
        """
        # `encode(normalize=True)` already returns (z - mean)/std — normalization happens
        # HERE and nowhere else. Wrapping this call in `normalize_latents` again is the
        # double-normalization that cost run v6e-official-30k its closed-loop score
        # (2026-08-06): 29,365 steps trained in a latent space scaled 1.85x away from the
        # one inference feeds. Guarded by the train/inference latent contract test.
        latents = vae.encode(pixels, normalize=True, channels_first=True)
        return sharding.shard_batch(latents.astype(jnp.float32))

    splits = config.tpu.vae_encode_splits
    vae_benchmark_compiled = False

    def encode_checked(pixels):
        nonlocal encode_pixels, vae_benchmark_compiled
        if benchmark_steps is not None and not vae_benchmark_compiled:
            encode_pixels = encode_pixels.lower(pixels).compile()
            analysis = encode_pixels.memory_analysis()
            if analysis is not None:
                planned = (analysis.argument_size_in_bytes + analysis.output_size_in_bytes
                           - analysis.alias_size_in_bytes + analysis.temp_size_in_bytes)
                _log(f'BENCHMARK_VAE_PLANNED_MEMORY {planned / 1e9:.2f} GB/card')
                if planned > 115e9:
                    raise MemoryError('VAE benchmark exceeds conservative 115 GB/card cap')
            vae_benchmark_compiled = True
        return encode_pixels(pixels)

    def vae_encode(pixels):
        if splits == 1:
            return encode_checked(pixels)
        # Split STRIDED, not contiguous. A contiguous slice px[0:B/K] lives on B/K/per_dev
        # chips and leaves the rest idle, turning K passes into a K-times slowdown; the
        # [B/K, K, ...] reshape puts one sample per chip per pass, so each pass does 1/K of
        # the work on ALL chips and only the activation peak changes.
        b = pixels.shape[0]
        reshaped = pixels.reshape(b // splits, splits, *pixels.shape[1:])
        outs = [encode_checked(reshaped[:, i]) for i in range(splits)]
        stacked = jnp.stack(outs, axis=1)
        return sharding.shard_batch(stacked.reshape(b, *stacked.shape[2:]))

    @jax.jit
    def make_hog_pixels(pixels):
        return sharding.shard_batch(hog_image(pixels))

    include_hog = (config.model.visual_input_mode == "rgb"
                   and config.model.training_towers == "rgb_hog_action")
    micro_spec = {
        key: NamedSharding(mesh, sharding.batch_spec(rank))
        for key, rank in micro_ranks(include_hog)
    }
    # Which accumulation shape to run. `accum == 1` needs no accumulator at all; beyond
    # that the default is the per-microstep path, because the fused one does not fit (see
    # make_microstep). `tpu.fused_accum` opts back in for configurations where it does.
    fused_step = microstep = apply_step = None
    grad_acc = None
    if accum == 1 or config.tpu.fused_accum:
        fused_step = make_fused_step(
            graphdef,
            tx,
            accum,
            zero_n,
            copy_lengths=copy_lengths,
            jit_kwargs=dict(
                in_shardings=(p_shard, o_shard, [replicated] * accum, [micro_spec] * accum),
                out_shardings=(p_shard, o_shard, replicated),
            ),
        )
        _log(f"accumulation: ONE fused jit over {accum} microstep(s)")
    else:
        microstep = make_microstep(
            graphdef,
            zero_n,
            copy_lengths=copy_lengths,
            jit_kwargs=dict(
                in_shardings=(p_shard, p_shard, replicated, micro_spec, replicated),
                out_shardings=(p_shard, replicated),
            ),
        )
        apply_step = make_apply_step(
            tx,
            accum,
            jit_kwargs=dict(
                in_shardings=(p_shard, o_shard, p_shard),
                out_shardings=(p_shard, o_shard, replicated, p_shard),
            ),
        )
        # Allocated once, with its sharding PINNED. `zeros_like` reads only shape and
        # dtype, so without out_shardings XLA would build this replicated — the same trap
        # that made tx.init ask for 44.9 G. Its contents are irrelevant: microstep 0 of
        # every optimizer step overwrites rather than adds.
        grad_acc = jax.jit(
            lambda tree: jax.tree.map(jnp.zeros_like, tree), out_shardings=p_shard
        )(params)
        reset_true = jax.device_put(jnp.asarray(True), replicated)
        reset_false = jax.device_put(jnp.asarray(False), replicated)
        _log(f"accumulation: {accum} microsteps into a donated ZeRO accumulator, then one apply")

    @jax.jit
    def check_finite(param_tree, opt):
        """One replicated bool: are ALL floating leaves finite?

        On the checkpoint cadence only. It refuses to persist a NaN state (which resume
        would then reload) without paying a full-tree reduction every step.
        """

        def ok(x):
            x = jnp.asarray(x)
            if jnp.issubdtype(x.dtype, jnp.floating):
                return jnp.all(jnp.isfinite(x))
            return jnp.array(True)

        return jax.tree.reduce(
            lambda acc, x: jnp.logical_and(acc, ok(x)), (param_tree, opt), jnp.array(True)
        )

    # ---- resume -------------------------------------------------------------
    full_root = f"{run_root}/full"
    start_step = 1
    if config.tpu.resume:
        resume_root = config.tpu.resume_from or full_root
        restored_box: dict = {}

        def build_target():
            def abstract(x):
                return jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)

            target = {
                "params": jax.tree.map(abstract, nnx.to_pure_dict(params)),
                "opt_state": jax.tree.map(abstract, opt_state),
                "step": np.asarray(0, dtype=np.int64),
            }
            # Free the from-scratch arrays BEFORE the restore is issued, or the fresh state
            # and the restored state sit in HBM together and the first step OOMs.
            for leaf in jax.tree.leaves(nnx.to_pure_dict(params)):
                leaf.delete()
            for leaf in jax.tree.leaves(opt_state):
                leaf.delete()
            return target

        start_step = ckpt_mod.resume(
            resume_root,
            build_target,
            apply_restored=restored_box.update,
            allow_fresh=config.tpu.resume_from is None,
        )
        if restored_box:
            nnx.replace_by_pure_dict(params, restored_box["params"])
            opt_state = restored_box["opt_state"]
        _log(f"resume: starting at step {start_step}")

    # ---- checkpointer -------------------------------------------------------
    full_every = config.tpu.ckpt.full_every or config.save_every
    bf16_every = config.tpu.ckpt.bf16_every or config.save_every
    checkpointer = ckpt_mod.Checkpointer(run_root, keep_last=config.tpu.ckpt.keep_last)
    _log(f"ckpt: full every {full_every}, bf16 every {bf16_every}")

    # ---- data: seek to the resume point, THEN prefill (overlaps the cold compile) ----
    # The pool counts `sample_batch` calls, and there are `accum` of those per optimizer
    # step, so the resume point has to be scaled. Seeking to `start_step - 1` instead would
    # rewind the pool by a factor of `accum` and quietly retrain on episodes the run had
    # already moved past.
    dataset.seek((start_step - 1) * accum).start()
    base_key = jax.random.key(config.seed)

    def data_rng(step: int, micro: int) -> np.random.Generator:
        """A fresh generator per (host, step, microstep), not one long stateful stream.

        A stream advanced by consumption restarts from the beginning after a preemption:
        the pool lands on the right episodes but every window offset, instruction and draw
        replays differently, so a resumed run silently diverges from an uninterrupted one
        and neither is reproducible. Deriving per step is what makes "the batch is a pure
        function of (seed, step)" — the property the loader's whole resume story rests on —
        actually true.
        """
        return np.random.default_rng(
            stable_seed(config.seed, "batch", jax.process_index(), step, micro)
        )

    if is_chief:
        spec = dataset.data_spec()
        spec.update(
            {
                "run_name": run_name,
                "run_root": run_root,
                "global_batch_size": global_bs,
                "micro_batch_size": micro_bs,
                "grad_accum": accum,
                "steps": steps,
                "decay_steps": decay_steps,
                "warmup_steps": warmup_steps,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "max_grad_norm": config.max_grad_norm,
                "remat_policy": policy.value,
                "vae_encode_splits": splits,
                "prefetch_host_batch": prefetch_host_batch,
                "scan_layers": config.tpu.scan_layers,
                # A copies run trains a different objective from a single-copy one at the
                # same step count. A run whose spec does not say so cannot be compared.
                "copy_lengths": list(copy_lengths),
                "train_action_tokens": config.shapes.train_action_tokens,
                # Both change the emitted collectives, and shard_activations changes the loss
                # at the last fp32 bit. A run whose spec does not say which it used cannot be
                # compared to any other run.
                "shard_activations": config.tpu.shard_activations,
                "param_axis_policy": config.tpu.param_axis_policy,
                "mesh": {name: int(mesh.shape[name]) for name in mesh.axis_names},
                "compute_dtype": config.tpu.compute_dtype,
                "param_dtype": config.tpu.param_dtype,
                "n_params": n_params,
                "video_dit_init": config.model.weights.video_dit_orbax,
                "hog_dit_init": (
                    ("random" if config.model.hog_dit_init == "random"
                     else config.model.weights.video_dit_orbax)
                    if include_hog
                    else None
                ),
                "hog_dit_init_mode": config.model.hog_dit_init,
                "hog_preprocess": "cell8-bin9-l2hys-3moment-v1",
                "visual_input_mode": config.model.visual_input_mode,
                "hog_condition_mode": config.model.hog_condition_mode,
                "training_towers": config.model.training_towers,
                "data_format": config.data.format,
                "hog_inference": (
                    "RGB->HOG before VAE; classic video/action path"
                    if config.model.visual_input_mode == "rgb_to_hog"
                    else "removed; RGB/Action queries never attend HOG keys"
                ),
                "action_dit_init": config.model.weights.action_dit_init,
            }
        )
        _log("run spec:\n" + json.dumps(spec, indent=1, default=str))

    # ---- MFU ----------------------------------------------------------------
    per_sample_flops = flops_per_sample(config)
    peak = peak_tflops(config)
    _log(
        f"MFU model: {per_sample_flops / 1e12:.3f} TFLOP/sample (fwd+bwd, remat recompute "
        f"excluded), peak {peak:.0f} TFLOP/s/chip, {config.shapes.train_joint_seq} joint "
        f"tokens/sample"
    )

    host_prefetch = None
    if prefetch_host_batch and not dry_run:
        from .host_prefetch import HostBatchPrefetch
        host_prefetch = HostBatchPrefetch(dataset.sample_batch, data_rng, host_bs, start_step, accum)
        _log('host prefetch: one deterministic LIBERO microbatch; GPU operations stay on the main thread')

    def sample_micro(step: int, j: int):
        """One microbatch: host-local numpy -> global arrays -> latents.

        Pixels are encoded immediately and dropped, so the accumulation holds `accum` sets
        of latents (~280 KB per sample) instead of `accum` sets of pixels (~13 MB each).
        """
        if host_prefetch is None:
            batch = dataset.sample_batch(data_rng(step, j), host_bs)
        else:
            batch = host_prefetch.get(step, j)
        pixels = sharding.global_from_host_local(batch["video"], mesh)
        hog_pixels = make_hog_pixels(pixels)
        # In the new ablation the HOG image is the *only* visual stream.  The
        # classic full/HOG recipes still encode RGB and carry a separate HOG
        # target for their historical auxiliary objective.
        visual_pixels = hog_pixels if config.model.visual_input_mode == "rgb_to_hog" else pixels
        micro = {
            "latents": vae_encode(visual_pixels),
            "action": sharding.global_from_host_local(batch["action"], mesh),
            "proprio": sharding.global_from_host_local(batch["proprio"], mesh),
            "context": sharding.global_from_host_local(
                batch["context"].astype(np.float32), mesh
            ).astype(compute_dtype),
            "context_mask": sharding.global_from_host_local(batch["context_mask"], mesh),
            "image_is_pad": sharding.global_from_host_local(batch["image_is_pad"], mesh),
            "action_is_pad": sharding.global_from_host_local(batch["action_is_pad"], mesh),
        }
        if include_hog:
            micro["hog_latents"] = vae_encode(hog_pixels)
        del pixels, hog_pixels, visual_pixels
        # The noise key is SHARED across hosts (the jitted step must be identical everywhere)
        # and DISTINCT per microstep.
        return jax.random.fold_in(base_key, step * accum + j), micro

    if dry_run:
        _log("dry_run: everything is built; exiting before step 1")
        dataset.close()
        return

    def hbm_gb() -> tuple[float, float]:
        try:
            stats = jax.local_devices()[0].memory_stats() or {}
        except Exception:  # noqa: BLE001 - observability must never take the run down
            return (0.0, 0.0)
        return ((stats.get("bytes_in_use") or 0) / 1e9, (stats.get("peak_bytes_in_use") or 0) / 1e9)

    in_use, peak_hbm = hbm_gb()
    _log(f"HBM before the loop: in use {in_use:.1f}G / peak {peak_hbm:.1f}G")
    _log(f"startup took {time.time() - t_start:.1f}s; the next gap is the cold compile")

    profile = config.tpu.profile
    window_time, window_step = None, 0
    first_done = False
    checkpoint_metrics = []
    benchmark_records = []
    benchmark_memory = None
    benchmark_warm_time = None
    benchmark_end = min(steps, start_step + benchmark_steps - 1) if benchmark_steps else steps
    t_loop = time.time()

    def require_finite(at_step: int) -> None:
        if not bool(check_finite(params, opt_state)):
            raise FloatingPointError(
                f"step {at_step}: non-finite params/optimizer state AFTER the update — refusing "
                "to checkpoint (resume would reload it). The last good checkpoint is preserved."
            )

    try:
        for step in range(start_step, benchmark_end + 1):
            if profile.enabled and is_chief and step == profile.start_step:
                jax.profiler.start_trace(profile.dir)
            with jax.profiler.StepTraceAnnotation("train_step", step_num=step):
                if fused_step is not None:
                    keys, micros = [], []
                    for j in range(accum):
                        key, micro = sample_micro(step, j)
                        keys.append(key)
                        micros.append(micro)
                    if benchmark_steps is not None and not first_done:
                        fused_step = fused_step.lower(params, opt_state, keys, micros).compile()
                        analysis = fused_step.memory_analysis()
                        if analysis is not None:
                            benchmark_memory = {k: int(getattr(analysis, k)) for k in (
                                'argument_size_in_bytes', 'output_size_in_bytes',
                                'alias_size_in_bytes', 'temp_size_in_bytes')}
                            planned = (analysis.argument_size_in_bytes + analysis.output_size_in_bytes
                                       - analysis.alias_size_in_bytes + analysis.temp_size_in_bytes)
                            _log('BENCHMARK_PLANNED_MEMORY ' + json.dumps(benchmark_memory))
                            if planned > 125e9:
                                raise MemoryError('Benchmark executable exceeds conservative 125 GB/card cap')
                    params, opt_state, metrics = fused_step(params, opt_state, keys, micros)
                    del keys, micros
                else:
                    loss_acc = None
                    for j in range(accum):
                        key, micro = sample_micro(step, j)
                        if benchmark_steps is not None and not first_done and j == 0:
                            microstep = microstep.lower(grad_acc, params, key, micro, reset_true).compile()
                            analysis = microstep.memory_analysis()
                            if analysis is not None:
                                benchmark_memory = {k: int(getattr(analysis, k)) for k in (
                                    'argument_size_in_bytes', 'output_size_in_bytes',
                                    'alias_size_in_bytes', 'temp_size_in_bytes')}
                                planned = (analysis.argument_size_in_bytes + analysis.output_size_in_bytes
                                           - analysis.alias_size_in_bytes + analysis.temp_size_in_bytes)
                                _log('BENCHMARK_MICRO_MEMORY ' + json.dumps(benchmark_memory))
                                if planned > 115e9:
                                    raise MemoryError('Microstep exceeds conservative 115 GB/card cap')
                        grad_acc, scalars = microstep(
                            grad_acc, params, key, micro, reset_true if j == 0 else reset_false
                        )
                        loss_acc = (
                            dict(scalars)
                            if loss_acc is None
                            else {k: loss_acc[k] + v for k, v in scalars.items()}
                        )
                        del micro
                    params, opt_state, norms, grad_acc = apply_step(params, opt_state, grad_acc)
                    metrics = {k: v * (1.0 / accum) for k, v in loss_acc.items()}
                    metrics.update(norms)

            if not first_done:
                jax.block_until_ready(metrics["loss"])
                _log(f"step {step} done (including the cold compile) at {time.time() - t_start:.1f}s")
                first_done = True
            if profile.enabled and is_chief and step == profile.start_step + profile.num_steps - 1:
                jax.block_until_ready(metrics["loss"])
                jax.profiler.stop_trace()
                _log(f"profiler trace -> {profile.dir}")

            if step == start_step + 1:  # the first post-compile step anchors steady-state timing
                jax.block_until_ready(metrics["loss"])
                window_time, window_step = time.time(), step
                benchmark_warm_time = window_time

            if (step % config.log_every == 0 or step == benchmark_end
                    or (benchmark_steps is not None and step == start_step)):
                # Block on THIS step's loss so the throughput below is anchored to device
                # completion; without it the async dispatch runs ahead and samples/s is fiction.
                jax.block_until_ready(metrics["loss"])
                values = {k: float(v) for k, v in metrics.items()}
                if not np.isfinite(values["loss"]):
                    raise FloatingPointError(f"non-finite loss at step {step}: {values['loss']}")
                learning_rate = float(schedule(step - 1))
                parts = [
                    f"step {step}/{steps}",
                    f"loss {values['loss']:.4f}",
                    f"v {values['loss_video']:.4f}",
                    f"h {values['loss_hog']:.4f}",
                    f"a {values['loss_action']:.4f}",
                    f"|g| {values['grad_norm']:.3f}",
                    f"lr {learning_rate:.3e}",
                ]
                if window_time is not None and step > window_step:
                    steps_per_s = (step - window_step) / (time.time() - window_time)
                    samples_per_s = steps_per_s * global_bs
                    tflops = per_sample_flops * (samples_per_s / n_dev) / 1e12
                    parts += [
                        f"{1.0 / steps_per_s:.2f}s/step",
                        f"{samples_per_s:.1f}smp/s",
                        f"MFU {tflops / peak * 100:.1f}%",
                    ]
                    window_time, window_step = time.time(), step
                parts.append(f"HBM {hbm_gb()[1]:.1f}G")
                _log("  ".join(parts))
                if is_chief:
                    checkpoint_metrics.append(
                        {"step": int(step), "learning_rate": learning_rate, **values}
                    )
                    if benchmark_steps is not None:
                        benchmark_records.append({"step": int(step), **values})

            # An isolated speed test restores a real optimizer state but NEVER
            # publishes checkpoints, metrics sidecars, or a final inference export.
            # Its sample keys, schedule and update code are the production path.
            if benchmark_steps is not None:
                if step == benchmark_end:
                    jax.block_until_ready((params, opt_state))
                    benchmark_elapsed = time.time() - benchmark_warm_time
                continue

            # Every host validates the post-update state on the save cadence and raises in
            # lockstep BEFORE any save. A chief-only raise would hang the peers.
            if step % full_every == 0 or step == config.tpu.ckpt.early_verify_step:
                require_finite(step)
                fails = ckpt_mod.broadcast_int(checkpointer.save_fail_count if is_chief else 0)
                if fails >= config.tpu.ckpt.save_fail_limit:
                    raise RuntimeError(
                        f"the durable checkpoint has failed {fails} times in a row (last: "
                        f"{checkpointer.last_save_error}) — aborting rather than burning chips on "
                        "progress that is not landing. Fix GCS auth/quota, then relaunch with "
                        "tpu.resume=true."
                    )
                if checkpointer.save(
                    step,
                    nnx.to_pure_dict(params),
                    opt_state,
                    metrics=checkpoint_metrics if is_chief else None,
                ) and is_chief:
                    checkpoint_metrics.clear()
            if step % bf16_every == 0:
                checkpointer.save_params_bf16(step, nnx.to_pure_dict(params))

        # The cadences only fire on their multiples, so a `steps` that is not one would lose
        # the final state on exit. That is how a 20k run once lost its weights.
        if benchmark_steps is None and steps >= start_step:
            if steps % full_every:
                require_finite(steps)
                if checkpointer.save(
                    steps,
                    nnx.to_pure_dict(params),
                    opt_state,
                    metrics=checkpoint_metrics if is_chief else None,
                ) and is_chief:
                    checkpoint_metrics.clear()
            if steps % bf16_every:
                checkpointer.save_params_bf16(steps, nnx.to_pure_dict(params))
    finally:
        if host_prefetch is not None:
            host_prefetch.close()
        dataset.close()

    if benchmark_steps is not None:
        jax.block_until_ready((params, opt_state))
        if not bool(check_finite(params, opt_state)):
            raise FloatingPointError("Benchmark ended with non-finite parameters/optimizer")
        from pathlib import Path
        result = {
            "start_step": start_step, "end_step": benchmark_end,
            "warm_steps": benchmark_end - (start_step + 1),
            "warm_seconds": benchmark_elapsed,
            "peak_hbm_gb": hbm_gb()[1], "metrics": benchmark_records,
            "micro_batch_size": micro_bs, "grad_accum": accum,
            "global_batch_size": global_bs, "remat_policy": policy.value,
            "vae_encode_splits": splits, "resume_from": config.tpu.resume_from,
            "checkpoint_writes": False, "all_state_finite": True,
            "executable_memory_analysis": benchmark_memory,
            "prefetch_host_batch": prefetch_host_batch,
        }
        result["seconds_per_step"] = result["warm_seconds"] / result["warm_steps"]
        Path(benchmark_output).write_text(json.dumps(result, indent=2) + "\n")
        _log("BENCHMARK_COMPLETE " + json.dumps(result))
        return

    _log(f"done: {steps - start_step + 1} steps in {time.time() - t_loop:.1f}s")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> None:
    from ..config import load_config

    parser = argparse.ArgumentParser(
        prog="python -m fastwam_tpu.training.train",
        description="Train FastWAM (uncond) on a TPU slice.",
    )
    parser.add_argument(
        "training_config",
        help="one local recipe under configs/training/, e.g. configs/training/my_experiment.yaml",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="dotted config assignments, e.g. tpu.steps=50 tpu.micro_batch_size=32. "
        "A key that does not already exist raises.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build everything (mesh, weights, optimizer, data) and exit before step 1",
    )
    parser.add_argument(
        "--benchmark-steps", type=int, default=None,
        help="12-100 isolated real-checkpoint steps; no checkpoint/export writes",
    )
    parser.add_argument("--benchmark-output", default=None, help="Separate benchmark JSON path")
    parser.add_argument('--prefetch-host-batch', action='store_true',
                        help='Bounded deterministic CPU look-ahead for LIBERO microbatches')
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="skip the checkpoint restores and train from random weights. A throughput and "
        "memory harness ONLY: startup drops from ~20 min to ~2, the FLOPs are identical, and "
        "nothing it produces says anything about learning. Refused unless tpu.steps is set "
        "and small, so it cannot become the run that trained from scratch by accident.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    config = load_config(args.training_config, overrides=args.overrides)

    # The compilation cache MUST be configured before any jax op, i.e. before
    # jax.distributed.initialize(). A cold compile for this model is minutes and all four
    # hosts pay it; a relaunch with a warm cache skips the whole thing. Same region as the
    # chips, which the config's region guard already enforces.
    import jax

    cache = config.tpu.gcs.jax_cache_dir
    if cache:
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
        print(f"[train] compilation cache: {cache}", flush=True)

    if config.tpu.multihost:
        jax.distributed.initialize()

    if args.random_init:
        limit = 1000
        if config.tpu.steps is None or config.tpu.steps > limit:
            raise SystemExit(
                f"--random-init needs tpu.steps set to at most {limit} (got {config.tpu.steps}). "
                "It exists to time the hardware, not to train a model; a long run started this "
                "way would look normal and learn nothing from the pretrained towers."
            )
        print("[train] --random-init: NOT a real run, the experts are random", flush=True)

    train(config, dry_run=args.dry_run, load_weights=not args.random_init,
          benchmark_steps=args.benchmark_steps, benchmark_output=args.benchmark_output,
          prefetch_host_batch=args.prefetch_host_batch)


if __name__ == "__main__":
    main()
