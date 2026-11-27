"""M5: the training harness on CPU — mesh, schedule, fused accumulation, checkpoints.

Everything here runs on a shrunken model (widths only; the mask modes, schedulers, loss
lambdas and the per-token timestep path are the released values), so the pipe that the
v4-32 run uses is exercised end to end without 6 B parameters.

Multi-device tests need more than one CPU device. Get them with

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest tests/test_train_step.py

and they skip otherwise rather than pretending a 1-device mesh proved anything about
sharding.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from fastwam_tpu.config import ConfigError, load_config, parse_overrides
from fastwam_tpu.models.fastwam import FastWAM
from fastwam_tpu.training import checkpoint as ckpt_mod
from fastwam_tpu.training import remat as remat_mod
from fastwam_tpu.training import sharding
from fastwam_tpu.training.train import (
    MICRO_RANKS,
    build_schedule,
    flops_per_sample,
    make_apply_step,
    make_fused_step,
    make_microstep,
    official_total_steps,
)

TASK = "training/full.example.yaml"

LAYERS = 2
HEADS = 2
HEAD_DIM = 8
VIDEO_DIM = 16
ACTION_HIDDEN = 8
TEXT_DIM = 12
FREQ_DIM = 16
IN_CH = 4
ACTION_DIM = 3
PROPRIO_DIM = 3
LATENT_T, LATENT_H, LATENT_W = 3, 4, 4
ACTION_TOKENS = 8
CONTEXT_LEN = 5


def small_model_config():
    config = load_config(TASK)
    video = dataclasses.replace(
        config.model.video_dit_config,
        hidden_dim=VIDEO_DIM,
        ffn_dim=2 * VIDEO_DIM,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        in_dim=IN_CH,
        out_dim=IN_CH,
        action_dim=ACTION_DIM,
    )
    action = dataclasses.replace(
        config.model.action_dit_config,
        action_dim=ACTION_DIM,
        hidden_dim=ACTION_HIDDEN,
        ffn_dim=2 * ACTION_HIDDEN,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
    )
    return dataclasses.replace(
        config.model, video_dit_config=video, action_dit_config=action, proprio_dim=PROPRIO_DIM
    )


def make_micro(seed: int, batch: int = 2) -> dict:
    """A microbatch with exactly the keys `training_loss` accepts."""
    rng = np.random.default_rng(seed)

    def draw(*shape):
        return jnp.asarray(rng.standard_normal(shape), jnp.float32)

    image_frames = 4 * (LATENT_T - 1) + 1
    return {
        "latents": draw(batch, IN_CH, LATENT_T, LATENT_H, LATENT_W),
        "hog_latents": draw(batch, IN_CH, LATENT_T, LATENT_H, LATENT_W),
        "action": draw(batch, ACTION_TOKENS, ACTION_DIM),
        "proprio": draw(batch, ACTION_TOKENS, PROPRIO_DIM),
        "context": draw(batch, CONTEXT_LEN, TEXT_DIM),
        "context_mask": jnp.ones((batch, CONTEXT_LEN), bool),
        "image_is_pad": jnp.zeros((batch, image_frames), bool),
        "action_is_pad": jnp.zeros((batch, ACTION_TOKENS), bool),
    }


def build_state(scan_layers: bool = True, remat_policy: str = "full", lr: float = 1e-3):
    model = FastWAM(
        small_model_config(),
        rngs=nnx.Rngs(0),
        include_hog=True,
        scan_layers=scan_layers,
        remat_policy=remat_policy,
    )
    graphdef, params = nnx.split(model)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr, b1=0.9, b2=0.95, eps=1e-8))
    return graphdef, params, tx, tx.init(params)


# --------------------------------------------------------------------------- #
# the fused accumulation
# --------------------------------------------------------------------------- #
def test_one_fused_step_runs_and_moves_the_parameters():
    graphdef, params, tx, opt_state = build_state()
    step = make_fused_step(graphdef, tx, accum=2, zero_n=None)
    keys = [jax.random.key(1), jax.random.key(2)]
    micros = [make_micro(11), make_micro(12)]
    before = jax.tree.map(np.asarray, nnx.to_pure_dict(params))

    params, opt_state, metrics = step(params, opt_state, keys, micros)

    assert set(metrics) == {
        "loss", "loss_video", "loss_hog", "loss_action",
        "t_video", "t_hog", "t_action", "grad_norm",
    }
    for name, value in metrics.items():
        assert np.isfinite(float(value)), f"{name} is not finite"
    after = jax.tree.map(np.asarray, nnx.to_pure_dict(params))
    assert jax.tree.leaves(before), "the parameter tree is empty"
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)
    ), "no parameter changed: the optimizer apply did not run"


def test_accumulating_two_identical_microbatches_equals_one_big_step():
    """The whole reason accumulation is allowed to stand in for a bigger batch.

    Two microsteps over the SAME batch with the same key must give the same update as one
    microstep over that batch: the gradients are averaged, not summed, and nothing else in
    the step depends on `accum`. If this drifts, the effective batch is a fiction and the
    B line is not training at the official 1024.
    """
    micro = make_micro(21)
    key = jax.random.key(5)

    graphdef, params_a, tx, opt_a = build_state()
    one = make_fused_step(graphdef, tx, accum=1, zero_n=None)
    params_a, _, metrics_a = one(params_a, opt_a, [key], [micro])

    graphdef, params_b, tx, opt_b = build_state()
    two = make_fused_step(graphdef, tx, accum=2, zero_n=None)
    params_b, _, metrics_b = two(params_b, opt_b, [key, key], [micro, micro])

    assert float(metrics_a["loss"]) == pytest.approx(float(metrics_b["loss"]), rel=1e-5)
    for a, b in zip(
        jax.tree.leaves(nnx.to_pure_dict(params_a)),
        jax.tree.leaves(nnx.to_pure_dict(params_b)),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-6)


def test_the_microstep_path_matches_the_fused_one_bit_for_bit():
    """The default (per-microstep) accumulation and the fused one must be the same update.

    They are two schedules for the same arithmetic, chosen on memory: the fused program
    needs every microstep live at once, which at the official batch is 32.16 G against
    30.75 G of HBM on v4-32. If they ever diverge, one of them is wrong and the throughput
    numbers measured with one do not describe the other.
    """
    micro = make_micro(71)
    keys = [jax.random.key(4), jax.random.key(5)]

    graphdef, params_f, tx, opt_f = build_state()
    fused = make_fused_step(graphdef, tx, accum=2, zero_n=None)
    params_f, _, metrics_f = fused(params_f, opt_f, keys, [micro, micro])

    graphdef, params_m, tx, opt_m = build_state()
    step = make_microstep(graphdef, zero_n=None)
    apply_step = make_apply_step(tx, accum=2)
    grad_acc = jax.tree.map(jnp.zeros_like, params_m)
    totals = None
    for index in range(2):
        grad_acc, scalars = step(grad_acc, params_m, keys[index], micro, index == 0)
        totals = dict(scalars) if totals is None else {k: totals[k] + v for k, v in scalars.items()}
    params_m, _, norms, _ = apply_step(params_m, opt_m, grad_acc)

    assert float(metrics_f["loss"]) == pytest.approx(float(totals["loss"]) / 2, rel=1e-6)
    assert float(metrics_f["grad_norm"]) == pytest.approx(float(norms["grad_norm"]), rel=1e-6)
    for a, b in zip(
        jax.tree.leaves(nnx.to_pure_dict(params_f)),
        jax.tree.leaves(nnx.to_pure_dict(params_m)),
        strict=True,
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_repeated_steps_on_one_batch_overfit_it():
    """The real check on the training math: a fixed batch and a fixed noise draw must be
    driven down. A loss that stalls here means the gradient never reaches the weights."""
    graphdef, params, tx, opt_state = build_state(lr=3e-3)
    step = make_fused_step(graphdef, tx, accum=1, zero_n=None)
    micro = make_micro(31)
    key = jax.random.key(7)

    losses = []
    for _ in range(40):
        params, opt_state, metrics = step(params, opt_state, [key], [micro])
        losses.append(float(metrics["loss"]))

    assert all(np.isfinite(losses))
    assert losses[-1] < 0.4 * losses[0], f"loss did not drop: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_the_step_runs_under_both_layer_layouts():
    """`scan_layers` is a layout choice; the v4-32 run needs it on, the parity tools run
    unrolled, and both have to survive a real optimizer step.

    Numerical equivalence of the two layouts is NOT retested here — comparing two freshly
    constructed models would only compare two different random inits, since the two layouts
    consume the RNG differently. `tests/test_mot_parity.py::test_scan_matches_unrolled`
    proves it properly, by restacking one model's weights into the other's layout.
    """
    for scan_layers in (True, False):
        graphdef, params, tx, opt_state = build_state(scan_layers=scan_layers)
        step = make_fused_step(graphdef, tx, accum=1, zero_n=None)
        _, _, metrics = step(params, opt_state, [jax.random.key(3)], [make_micro(41)])
        assert np.isfinite(float(metrics["loss"])), f"scan_layers={scan_layers}"


# --------------------------------------------------------------------------- #
# schedule
# --------------------------------------------------------------------------- #
def test_official_total_steps_matches_the_reference_formula():
    config = load_config(TASK)
    # trainer.py:204-253 over the full RoboTwin transition count at global batch 1024.
    assert official_total_steps(6_075_103, config) == 29_665
    assert official_total_steps(1024, config) == config.num_epochs
    assert official_total_steps(1025, config) == 2 * config.num_epochs


def test_schedule_warms_up_then_decays_to_one_percent():
    config = load_config(TASK)
    total = 29_663
    schedule, warmup = build_schedule(config, total)
    assert warmup == int(total * 0.05)
    # Tolerances are fp32-sized: optax evaluates the schedule in float32.
    assert float(schedule(warmup)) == pytest.approx(config.learning_rate, rel=1e-5)
    assert float(schedule(0)) == pytest.approx(config.learning_rate / warmup, rel=1e-4)
    assert float(schedule(total - 1)) == pytest.approx(
        config.learning_rate * config.optimizer.min_lr_ratio, rel=1e-3
    )
    assert float(schedule(warmup // 2)) < config.learning_rate


def test_a_short_run_sees_the_full_runs_lr_at_the_same_step():
    """The codex rule made mechanical: shortening `steps` must not shorten the cosine.

    A 50-step smoke tied to its own 50-step cosine would be at the LR FLOOR by step 49 —
    every observation it produced would come from a model that had effectively stopped
    learning, and the numbers would not predict the full run's at all. With `decay_steps`
    defaulting to the derived total, step 49 of the smoke sits where step 49 of the full
    run sits: still inside warmup.
    """
    config = load_config(TASK)
    floor = config.learning_rate * config.optimizer.min_lr_ratio
    full, warmup = build_schedule(config, 29_663)
    short_by_mistake, _ = build_schedule(config, 1000)

    # A 1000-step run tied to its own cosine has finished decaying by its last step; the
    # same 1000 steps as a prefix of the real schedule are still climbing through warmup,
    # 60x higher. Every number the smoke produced would come from a different LR regime
    # than the run it is supposed to predict.
    assert float(short_by_mistake(999)) == pytest.approx(floor, rel=0.05)
    assert float(full(999)) > 50 * float(short_by_mistake(999))

    # And the shapes are opposite, which is the part that no single point can show.
    assert float(short_by_mistake(999)) < float(short_by_mistake(500))
    assert float(full(999)) > float(full(500))
    assert 999 < warmup, "step 999 of the full schedule should still be warming up"


def test_config_defaults_leave_the_schedule_horizon_derived():
    config = load_config(TASK)
    assert config.tpu.steps is None
    assert config.tpu.decay_steps is None


# --------------------------------------------------------------------------- #
# FLOPs / MFU bookkeeping
# --------------------------------------------------------------------------- #
def test_flops_per_sample_is_in_the_expected_band():
    config = load_config(TASK)
    tflops = flops_per_sample(config) / 1e12
    # RGB + HOG + Action attend over 752 tokens.  The exact analytic count is ~21.4
    # TFLOP/sample; a broad guard catches a dropped tower or a double-counted backward.
    assert 18.0 < tflops < 26.0, tflops


def test_flops_scale_linearly_with_layers():
    config = load_config(TASK)
    doubled = dataclasses.replace(
        config,
        model=dataclasses.replace(
            config.model,
            video_dit_config=dataclasses.replace(config.model.video_dit_config, num_layers=60),
            action_dit_config=dataclasses.replace(config.model.action_dit_config, num_layers=60),
        ),
    )
    assert flops_per_sample(doubled) == pytest.approx(2 * flops_per_sample(config))


# --------------------------------------------------------------------------- #
# remat
# --------------------------------------------------------------------------- #
def test_only_none_and_full_are_offered():
    assert remat_mod.resolve("none") is remat_mod.RematPolicy.NONE
    assert remat_mod.resolve("full") is remat_mod.RematPolicy.FULL
    for tier in ("save_qkv", "save_attn_out", "offload_hidden", "custom"):
        with pytest.raises(NotImplementedError, match="checkpoint_name"):
            remat_mod.resolve(tier)
    with pytest.raises(ValueError, match="unknown policy"):
        remat_mod.resolve("aggressive")


def test_a_selective_policy_cannot_silently_save_nothing():
    with pytest.raises(ValueError, match="NOTHING"):
        remat_mod.check_checkpoint_names("joint_q")


# --------------------------------------------------------------------------- #
# mesh + sharding
# --------------------------------------------------------------------------- #
def mesh_config(**tpu_overrides):
    config = load_config(TASK)
    return dataclasses.replace(config, tpu=dataclasses.replace(config.tpu, **tpu_overrides))


def test_mesh_fills_the_minus_one_axis():
    config = mesh_config(ici_data_parallelism=1, ici_fsdp_parallelism=-1)
    mesh = sharding.make_mesh(config)
    assert mesh.shape["data"] == 1
    assert mesh.shape["fsdp"] == jax.device_count()
    assert sharding.zero_degree(mesh) == jax.device_count()
    sharding.uninstall()


def test_mesh_refuses_a_dcn_axis():
    with pytest.raises(sharding.ShardingError, match="ONE slice"):
        sharding.make_mesh(mesh_config(dcn_fsdp_parallelism=2))


def test_mesh_refuses_a_product_that_is_not_the_device_count():
    with pytest.raises(sharding.ShardingError, match="multiplies out"):
        sharding.make_mesh(
            mesh_config(ici_data_parallelism=3, ici_fsdp_parallelism=5)
        )


def test_mesh_refuses_two_minus_ones():
    with pytest.raises(sharding.ShardingError, match="at most one"):
        sharding.make_mesh(mesh_config(ici_data_parallelism=-1, ici_fsdp_parallelism=-1))


def test_mesh_devices_are_ordered_by_process_then_id():
    mesh = sharding.make_mesh(mesh_config())
    order = [(d.process_index, d.id) for d in mesh.devices.flat]
    assert order == sorted(order)
    sharding.uninstall()


def test_largest_axis_spec_picks_a_divisible_axis_or_replicates():
    spec = sharding.largest_axis_spec(jax.ShapeDtypeStruct((16, 4), jnp.float32), 4)
    assert spec == jax.sharding.PartitionSpec("fsdp", None)
    spec = sharding.largest_axis_spec(jax.ShapeDtypeStruct((5, 3), jnp.float32), 4)
    assert spec == jax.sharding.PartitionSpec()
    assert sharding.largest_axis_spec(jax.ShapeDtypeStruct((), jnp.float32), 4) == (
        jax.sharding.PartitionSpec()
    )


def test_constraints_are_the_identity_when_not_installed():
    sharding.uninstall()
    x = jnp.ones((4, 3))
    assert sharding.shard_batch(x) is x
    tree = {"a": jnp.ones((4,))}
    assert sharding.shard_grads(tree, 2) is tree


@pytest.mark.skipif(jax.device_count() < 2, reason="needs >1 device; see the module docstring")
def test_params_land_in_the_zero_layout():
    n_dev = jax.device_count()
    mesh = sharding.make_mesh(mesh_config())
    tree = {"big": np.zeros((4 * n_dev, 8), np.float32), "scalar": np.zeros((), np.float32)}
    placed = sharding.shard_params(tree, mesh, shard=True)
    assert placed["big"].sharding.spec == jax.sharding.PartitionSpec("fsdp", None)
    assert placed["scalar"].sharding.spec == jax.sharding.PartitionSpec()
    replicated = sharding.shard_params(tree, mesh, shard=False)
    assert replicated["big"].sharding.spec == jax.sharding.PartitionSpec()
    np.testing.assert_array_equal(np.asarray(placed["big"]), tree["big"])
    sharding.uninstall()


@pytest.mark.skipif(jax.device_count() < 2, reason="needs >1 device; see the module docstring")
def test_a_fused_step_on_a_real_mesh_matches_the_unsharded_one():
    """Sharding is a placement decision, not a numerics one."""
    mesh = sharding.make_mesh(mesh_config())
    micro = make_micro(51, batch=jax.device_count())
    key = jax.random.key(9)

    sharding.uninstall()
    graphdef, params, tx, opt_state = build_state()
    plain = make_fused_step(graphdef, tx, accum=1, zero_n=None)
    params_plain, _, metrics_plain = plain(params, opt_state, [key], [micro])

    sharding.install(mesh, constraints=True)
    graphdef, params, tx, opt_state = build_state()
    params = sharding.shard_params(nnx.to_pure_dict(params), mesh, shard=True)
    model = FastWAM(
        small_model_config(), rngs=nnx.Rngs(0), include_hog=True,
        scan_layers=True, remat_policy="full"
    )
    graphdef, state = nnx.split(model)
    nnx.replace_by_pure_dict(state, params)
    opt_state = sharding.shard_params(tx.init(state), mesh, shard=True)
    sharded = make_fused_step(graphdef, tx, accum=1, zero_n=sharding.zero_degree(mesh))
    _, _, metrics_sharded = sharded(state, opt_state, [key], [micro])

    assert float(metrics_sharded["loss"]) == pytest.approx(float(metrics_plain["loss"]), rel=1e-4)
    assert params_plain is not None
    sharding.uninstall()


# --------------------------------------------------------------------------- #
# the ZeRO axis policy and the in-model activation pins
#
# The v4-32 sharding fix. Background in `training/sharding.py`'s module docstring; the
# numbers quoted in these tests were measured by compiling the REAL 6.02 B shapes on a
# 16-device CPU mesh with abstract parameters, which is also what
# `test_pinning_activations_leaves_no_activation_collective_inside_the_scan` re-does.
# --------------------------------------------------------------------------- #
def abstract_real_model(policy: str, mesh):
    """The real 6.02 B FastWAM as ShapeDtypeStructs on `mesh` — no weights, ~0.5 s.

    `nnx.eval_shape` never allocates, so the full-size tree is available to a CPU test.
    That matters: every shape that makes the bug (3072, 14336, 30 scanned layers) is
    absent from the shrunken model the rest of this file uses.
    """
    from jax.sharding import NamedSharding

    config = load_config(TASK)
    zero_n = sharding.zero_degree(mesh)
    abstract = nnx.eval_shape(
        lambda: FastWAM(
            config.model,
            rngs=nnx.Rngs(0),
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            include_hog=True,
            scan_layers=True,
            remat_policy="full",
        )
    )
    graphdef, state = nnx.split(abstract)
    pure = nnx.to_pure_dict(state)
    placed = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(
            x.shape,
            jnp.float32,
            sharding=NamedSharding(mesh, sharding.largest_axis_spec(x, zero_n, policy=policy)),
        ),
        pure,
    )
    nnx.replace_by_pure_dict(state, placed)
    return config, graphdef, state


def leaf_names(tree):
    """path -> leaf, with the path flattened to the dotted name the checkpoint uses."""
    out = {}
    for path, leaf in jax.tree.leaves_with_path(tree):
        name = ".".join(str(getattr(k, "key", getattr(k, "idx", k))) for k in path)
        out[name] = leaf
    return out


def full_spec(leaf) -> tuple:
    """A leaf's PartitionSpec padded to its rank.

    jax canonicalises a spec by dropping trailing `None`s, so a rank-2 array that came back
    out of a jit reports `P('fsdp')` where `largest_axis_spec` produced `P('fsdp', None)`.
    Same placement, different repr; comparing the padded forms compares the placement.
    """
    spec = tuple(leaf.sharding.spec)
    return spec + (None,) * (len(leaf.shape) - len(spec))


def test_the_axis_policy_menu_is_closed():
    with pytest.raises(sharding.ShardingError, match="axis_policy"):
        sharding.largest_axis_spec(jax.ShapeDtypeStruct((16, 4), jnp.float32), 4, policy="widest")
    mesh = sharding.make_mesh(mesh_config())
    with pytest.raises(sharding.ShardingError, match="axis_policy"):
        sharding.install(mesh, axis_policy="widest")
    sharding.uninstall()


def test_the_default_axis_policy_still_places_every_parameter_where_gcs_has_it():
    """The shipped layout must not move: existing checkpoints were written under it.

    Orbax does re-shard on restore, so a change here would not corrupt anything — it would
    silently make one run's checkpoints laid out differently from the next one's, which is
    the sort of difference that only surfaces as a confusing restore months later.
    """
    assert sharding.axis_policy() == "largest", "uninstalled default drifted"
    for shape, expected in (
        ((30, 3072, 3072), (None, "fsdp", None)),  # attention q/k/v/o: tie -> the FIRST 3072
        ((30, 3072, 14336), (None, None, "fsdp")),  # FFN up: the widest axis IS the output
        ((30, 14336, 3072), (None, "fsdp", None)),  # FFN down: the widest axis is CONTRACTING
        ((30, 1024, 3072), (None, None, "fsdp")),  # action q/k/v
        ((30, 3072, 1024), (None, "fsdp", None)),  # action o
    ):
        spec = sharding.largest_axis_spec(jax.ShapeDtypeStruct(shape, jnp.float32), 16)
        assert spec == jax.sharding.PartitionSpec(*expected), shape


def test_the_last_policy_never_shards_an_axis_a_matmul_contracts_over():
    """What "last" means, on the real parameter tree rather than on invented shapes.

    Every kernel in this model — `nnx.Linear` and the reshape-and-dense patch embedding
    alike — is laid out with the OUTPUT features last and everything the matmul sums over
    before it. So "no contracting axis carries fsdp" is exactly "fsdp is on the last axis",
    with one legitimate escape: when the last axis is not divisible by the ZeRO degree the
    walk falls back inward rather than giving up and replicating the leaf.

    This documents the knob; it is not the shipped configuration. See
    `config.py:tpu.param_axis_policy` for the selected policy.
    """
    mesh = sharding.make_mesh(mesh_config())
    n_shards = sharding.zero_degree(mesh)
    _, _, state = abstract_real_model("last", mesh)

    checked = 0
    for name, leaf in leaf_names(nnx.to_pure_dict(state)).items():
        if not name.endswith("kernel"):
            continue
        checked += 1
        spec = full_spec(leaf)
        if all(axis is None for axis in spec):
            continue
        if spec[-1] == "fsdp":
            continue
        assert leaf.shape[-1] % n_shards, (
            f"{name} {leaf.shape} put 'fsdp' on a contracting axis ({spec}) even though its "
            f"output axis {leaf.shape[-1]} divides by {n_shards}"
        )
    assert checked > 20, f"only {checked} kernels found; the tree walk is not seeing the model"
    sharding.uninstall()


def test_the_shipped_policy_shards_contracting_axes_which_is_the_bug():
    """The defect, pinned as a fact so the fix cannot be quietly reverted into a no-op.

    Under the shipped "largest" policy a majority of this model's kernels carry 'fsdp' on an
    axis their matmul sums over. That is legal and costs no memory — it only decides which
    lowering XLA is tempted by, which is why the fix is `shard_activations`, not a change
    here.
    """
    mesh = sharding.make_mesh(mesh_config())
    _, _, state = abstract_real_model("largest", mesh)
    kernels = {n: v for n, v in leaf_names(nnx.to_pure_dict(state)).items() if n.endswith("kernel")}
    contracting = {
        name
        for name, leaf in kernels.items()
        if "fsdp" in full_spec(leaf) and full_spec(leaf)[-1] != "fsdp"
    }
    assert "mot.video.blocks.ffn.proj_out.kernel" in contracting
    assert "mot.video.blocks.attn1.proj_attn.kernel" in contracting
    assert "mot.video.blocks.ffn.act_fn.proj.kernel" not in contracting  # FFN up is fine
    assert len(contracting) > len(kernels) / 3, sorted(contracting)
    sharding.uninstall()


@pytest.mark.skipif(jax.device_count() < 2, reason="needs >1 device; see the module docstring")
def test_gradient_shardings_are_the_parameter_shardings():
    """A gradient pinned to a different layout than its parameter is a reshard every step.

    Both trees go through `largest_axis_spec`, so this can only break by one of them being
    given a different policy — which is exactly why the policy is process-wide state rather
    than an argument. Tree STRUCTURE is checked too: `shard_grads` tree.maps over whatever
    it is handed, so a structural mismatch would not raise here, it would surface as an
    optimizer that silently sees the wrong tree.
    """
    mesh = sharding.make_mesh(mesh_config())
    for policy in sharding.AXIS_POLICIES:
        sharding.install(mesh, constraints=True, axis_policy=policy)
        n_dev = jax.device_count()
        tree = {"kernel": np.zeros((4 * n_dev, 8), np.float32), "bias": np.zeros((8,), np.float32)}
        params = sharding.shard_params(tree, mesh, shard=True)
        pinned = jax.jit(lambda t: sharding.shard_grads(t, sharding.zero_degree(mesh)))(params)

        assert jax.tree.structure(pinned) == jax.tree.structure(params)
        for name in tree:
            assert full_spec(pinned[name]) == full_spec(params[name]), (policy, name)
    sharding.uninstall()


def test_the_activation_pins_leave_the_graph_untouched_when_the_switch_is_off():
    """`shard_activations=false` must be the byte-for-byte old harness.

    The pins are python-level identity when off, so the jaxpr the old code produced is the
    jaxpr this code produces — which is what lets the in-flight v4-32 run be resumed from
    this checkout without its numerics moving.
    """
    graphdef, params, tx, opt_state = build_state()
    micro, key = make_micro(81), jax.random.key(6)

    def constraint_count():
        step = make_fused_step(graphdef, tx, accum=1, zero_n=None)
        jaxpr = jax.make_jaxpr(step)(params, opt_state, [key], [micro])
        return str(jaxpr).count("sharding_constraint")

    mesh = sharding.make_mesh(mesh_config())
    sharding.install(mesh, constraints=True, activations=False)
    off = constraint_count()
    sharding.install(mesh, constraints=True, activations=True)
    on = constraint_count()
    sharding.uninstall()

    # The old harness emitted exactly one constraint per gradient leaf, from `shard_grads`,
    # and nothing else — no constraint lives inside the model. That is the count `off` has
    # to reproduce; comparing against an uninstalled mesh would compare against zero, since
    # uninstalling silences `shard_grads` too.
    grads_only = len(jax.tree.leaves(nnx.to_pure_dict(params)))
    assert off == grads_only, (
        f"shard_activations=false emitted {off} constraints, not the {grads_only} that "
        "`shard_grads` alone accounts for — something in the model is pinning unconditionally"
    )
    if jax.device_count() > 1:
        assert on > off, "shard_activations=true emitted no constraints"


@pytest.mark.skipif(jax.device_count() < 2, reason="needs >1 device; see the module docstring")
def test_pinning_activations_moves_the_loss_by_about_one_ulp():
    """The pins are semantically the identity, and NOT bit-exact. Both halves matter.

    `with_sharding_constraint` changes no arithmetic, but it does change which of two equal
    matmul partitionings XLA emits — one sums a contraction as `n_shards` partial sums and
    adds them, the other sums it in a single pass. Different association, different fp32
    rounding. Measured on this mesh: the loss moves by exactly one ulp and the gradients by
    ~4e-7 RMS-relative, and that is the whole story; anything larger means the pins are not
    where they were thought to be. Do not compare loss curves across this switch.
    """
    mesh = sharding.make_mesh(mesh_config())

    def grads_of(activations: bool):
        sharding.install(mesh, constraints=True, activations=activations)
        model = FastWAM(
            small_model_config(), rngs=nnx.Rngs(0), include_hog=True,
            scan_layers=True, remat_policy="full"
        )
        graphdef, state = nnx.split(model)
        nnx.replace_by_pure_dict(
            state, sharding.shard_params(nnx.to_pure_dict(state), mesh, shard=True)
        )
        zero_n = sharding.zero_degree(mesh)

        @jax.jit
        def run(params, key, micro):
            def loss_fn(tree, k, m):
                return nnx.merge(graphdef, tree).training_loss(m, k)

            (loss, _), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, micro)
            return loss, sharding.shard_grads(grads, zero_n)

        micro = make_micro(91, batch=2 * jax.device_count())
        loss, grads = run(state, jax.random.key(13), micro)
        return float(loss), jax.tree.map(np.asarray, nnx.to_pure_dict(grads))

    loss_off, grads_off = grads_of(False)
    loss_on, grads_on = grads_of(True)
    sharding.uninstall()

    ulp = float(np.spacing(np.float32(loss_off)))
    assert abs(loss_on - loss_off) <= 4 * ulp, (
        f"loss moved {abs(loss_on - loss_off):.3e}, more than 4 ulp ({4 * ulp:.3e}); the pins "
        "are supposed to change the summation order, not the arithmetic"
    )
    for (name, a), b in zip(
        leaf_names(grads_off).items(), jax.tree.leaves(grads_on), strict=True
    ):
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        rms = np.sqrt(np.mean(a**2))
        if rms == 0:
            continue
        assert np.max(np.abs(a - b)) / rms < 1e-4, name


@pytest.mark.skipif(jax.device_count() < 2, reason="needs >1 device; see the module docstring")
def test_pinning_activations_leaves_no_activation_collective_inside_the_scan():
    """The regression guard for the whole fix, at the shapes that made it matter.

    A collective inside the scan body runs once per MoT layer, i.e. 30 times per microstep,
    so a batch-shaped one there is the difference between 50 G and 367 G of traffic per
    microstep. Compiling with abstract parameters costs ~25 s and no memory, and it is
    the only check that sees the real 3072/14336/360-token geometry — on the shrunken model
    the activation-to-weight ratio is 40x smaller and XLA makes different choices.

    The microbatch is pinned at the 512 the v4-32 trace was taken at, not derived from the
    device count, because the whole question is a cost comparison between an activation and
    a weight and the batch is one side of it. Below ~128 the ACTION tower's hidden state
    ([B,32,1024]) is genuinely smaller than its kernels, XLA gathers it instead, and it is
    right to: that is the cost model working, not the bug coming back. The video tower
    ([B,360,3072]) and `t_mod` ([B,360,6,3072]) are on the other side of the comparison at
    every batch that fits.
    """
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = sharding.make_mesh(mesh_config())
    sharding.install(mesh, constraints=True, activations=True)
    config, graphdef, state = abstract_real_model("largest", mesh)

    micro_bs = 512
    assert micro_bs % jax.device_count() == 0, "pick a device count that divides 512"
    shapes = config.shapes
    p_shard = jax.tree.map(lambda x: x.sharding, state)
    o_struct = jax.eval_shape(lambda t: tx_for_real().init(t), state)
    o_shard = jax.tree.map(
        lambda x: NamedSharding(mesh, sharding.largest_axis_spec(x, sharding.zero_degree(mesh))),
        o_struct,
    )
    opt_abs = jax.tree.map(
        lambda x, s: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=s), o_struct, o_shard
    )
    replicated = NamedSharding(mesh, P())
    micro_spec = {key: NamedSharding(mesh, sharding.batch_spec(rank)) for key, rank in MICRO_RANKS}
    boolean = {"context_mask", "image_is_pad", "action_is_pad"}
    micro_shapes = {
        "latents": (micro_bs, *shapes.latent_shape),
        "action": (micro_bs, shapes.action_tokens, config.model.action_dit_config.action_dim),
        "proprio": (micro_bs, shapes.action_tokens, config.model.proprio_dim),
        "context": (micro_bs, shapes.context_tokens - 1, config.model.video_dit_config.text_dim),
        "context_mask": (micro_bs, shapes.context_tokens - 1),
        "image_is_pad": (micro_bs, shapes.video_frames),
        "action_is_pad": (micro_bs, shapes.action_tokens),
    }
    micro_abs = {
        name: jax.ShapeDtypeStruct(
            shape, jnp.bool_ if name in boolean else jnp.float32, sharding=micro_spec[name]
        )
        for name, shape in micro_shapes.items()
    }
    key_struct = jax.eval_shape(lambda: jax.random.key(0))
    key_abs = jax.ShapeDtypeStruct(key_struct.shape, key_struct.dtype, sharding=replicated)

    step = make_fused_step(
        graphdef,
        tx_for_real(),
        1,
        sharding.zero_degree(mesh),
        jit_kwargs=dict(
            in_shardings=(p_shard, o_shard, [replicated], [micro_spec]),
            out_shardings=(p_shard, o_shard, replicated),
        ),
    )
    hlo = step.lower(state, opt_abs, [key_abs], [micro_abs]).compile().as_text()
    sharding.uninstall()

    in_scan = in_scan_collectives(hlo)
    offenders = {k: n for k, n in in_scan.items() if k[1].startswith(f"f32[{micro_bs},")}
    assert not offenders, (
        "activation-shaped collectives are back inside the 30-layer scan body — every one of "
        f"these runs 30x per microstep: {offenders}"
    )
    # Without this the test would also pass if `in_scan_collectives` silently stopped finding
    # anything (an HLO format change, a scan that stopped being a while loop). Full-slice FSDP
    # gathers a weight per layer by construction, so the scan body always has SOME collective;
    # an empty result means the guard is looking at nothing.
    assert in_scan, (
        "no collective at all inside the scan body — the ZeRO weight all-gathers should be "
        "there, so this is the detector failing, not the traffic being gone"
    )


@pytest.mark.skipif(jax.device_count() < 4, reason="needs >=4 devices; see the module docstring")
def test_hsdp_splits_the_batch_over_everything_but_zeroes_only_over_fsdp():
    """HSDP is a config override, not a code path — this is what it buys and costs.

    `ici_data_parallelism>1` replicates the parameters across `data` groups and ZeRO-shards
    only inside a group, so the ZeRO degree falls to `ici_fsdp_parallelism` and each chip
    holds proportionally more optimizer state. The batch is still split over EVERY chip
    (`batch_axes` names both axes), which is the part that makes it a throughput knob rather
    than a smaller run: no chip does more work, they just gather weights in smaller rings.
    """
    n_dev = jax.device_count()
    mesh = sharding.make_mesh(
        mesh_config(ici_data_parallelism=n_dev // 4, ici_fsdp_parallelism=4)
    )
    sharding.install(mesh, constraints=True)
    assert sharding.zero_degree(mesh) == 4
    assert sharding.batch_axes(mesh) == ("data", "fsdp")

    tree = {"kernel": np.zeros((16, 8), np.float32)}
    placed = sharding.shard_params(tree, mesh, shard=True)
    assert full_spec(placed["kernel"]) == ("fsdp", None)
    assert placed["kernel"].sharding.shard_shape((16, 8)) == (4, 8), "ZeRO degree is not 4"

    batch = jax.jit(sharding.shard_batch)(jnp.zeros((n_dev, 4, 4)))
    assert batch.sharding.shard_shape(batch.shape)[0] == 1, "the batch is not split over all chips"
    sharding.uninstall()


def tx_for_real():
    config = load_config(TASK)
    return optax.chain(
        optax.clip_by_global_norm(float(config.max_grad_norm)),
        optax.adamw(
            float(config.learning_rate),
            b1=config.optimizer.beta1,
            b2=config.optimizer.beta2,
            eps=config.optimizer.eps,
            weight_decay=config.weight_decay,
        ),
    )


def in_scan_collectives(hlo: str) -> dict:
    """{(op, result shape): count} for every collective OUTSIDE the ENTRY computation.

    "Outside ENTRY" is "inside the while body that `nnx.scan` lowers to", i.e. executed once
    per MoT layer — 30x whatever the count says. Collectives never appear inside fusion
    computations, so the ENTRY test is enough to separate per-step traffic from per-layer
    traffic, and it does not depend on XLA's generated computation names.
    """
    import re

    shape_re = re.compile(r"\w+\[[\d,]*\]")
    collectives = ("all-gather", "all-reduce", "reduce-scatter", "all-to-all", "collective-permute")
    found: dict = {}
    computation = ""
    for line in hlo.splitlines():
        if line and not line.startswith(" ") and "{" in line:
            computation = line.split("{")[0].strip()
        if computation.startswith("ENTRY") or " = " not in line:
            continue
        rhs = line.strip().split(" = ", 1)[1]
        op = rhs.split("(", 1)[0].split(" ")[-1]
        if op not in collectives:
            continue
        for shape in shape_re.findall(rhs.split(" ", 1)[0]):
            found[(op, shape)] = found.get((op, shape), 0) + 1
    return found


# --------------------------------------------------------------------------- #
# checkpointing
# --------------------------------------------------------------------------- #
def test_step_number_rejects_orbax_temp_dirs():
    assert ckpt_mod.step_number("gs://b/run/full/step_000000100") == 100
    assert ckpt_mod.step_number("step_7/") == 7
    assert ckpt_mod.step_number("step_000000100.orbax-checkpoint-tmp-12345") is None
    assert ckpt_mod.step_number("params_bf16_step0000100") is None
    assert ckpt_mod.step_number("full") is None


def test_finalized_steps_skips_directories_without_the_marker(tmp_path):
    import fsspec

    fs = fsspec.filesystem("file")
    root = tmp_path / "full"
    for step, commit in ((10, True), (20, False), (30, True)):
        directory = root / f"step_{step:09d}"
        directory.mkdir(parents=True)
        (directory / "data").write_text("x")
        if commit:
            (directory / "commit_success.txt").write_text(json.dumps({"step": step}))
    found = ckpt_mod.finalized_steps(fs, str(root))
    assert [step for step, _ in found] == [30, 10]


def test_finalized_steps_on_a_missing_root_is_a_fresh_start_not_an_error(tmp_path):
    import fsspec

    assert ckpt_mod.finalized_steps(fsspec.filesystem("file"), str(tmp_path / "nope")) == []


def test_save_restore_round_trip_and_retention(tmp_path):
    """The full contract on a local root: data first, marker last, keep_last enforced, and
    the restored tree equal to what was saved."""
    graphdef, params, tx, opt_state = build_state()
    checkpointer = ckpt_mod.Checkpointer(str(tmp_path / "run"), keep_last=2)

    for step in (1, 2, 3):
        assert checkpointer.save(step, nnx.to_pure_dict(params), opt_state)
        params, opt_state, _ = make_fused_step(graphdef, tx, accum=1, zero_n=None)(
            params, opt_state, [jax.random.key(step)], [make_micro(60 + step)]
        )

    finalized = checkpointer.finalized()
    assert [step for step, _ in finalized] == [3, 2], "keep_last=2 did not prune"
    saved_pure = nnx.to_pure_dict(params)

    assert checkpointer.save(4, saved_pure, opt_state)
    target = {
        "params": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), saved_pure),
        "opt_state": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), opt_state),
        "step": np.asarray(0, np.int64),
    }
    restored = checkpointer.restore(checkpointer.step_dir(4), target)
    assert int(restored["step"]) == 4
    for a, b in zip(
        jax.tree.leaves(saved_pure), jax.tree.leaves(restored["params"]), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_resume_picks_the_newest_finalized_step(tmp_path):
    graphdef, params, tx, opt_state = build_state()
    root = str(tmp_path / "run")
    checkpointer = ckpt_mod.Checkpointer(root, keep_last=5)
    pure = nnx.to_pure_dict(params)
    for step in (5, 9):
        assert checkpointer.save(step, pure, opt_state)

    box: dict = {}
    next_step = ckpt_mod.resume(
        f"{root}/full",
        lambda: {
            "params": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), pure),
            "opt_state": jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), opt_state),
            "step": np.asarray(0, np.int64),
        },
        apply_restored=box.update,
        allow_fresh=True,
    )
    assert next_step == 10
    assert int(box["step"]) == 9


def test_resume_on_an_empty_home_is_a_fresh_start(tmp_path):
    assert (
        ckpt_mod.resume(
            str(tmp_path / "run" / "full"),
            lambda: pytest.fail("the target must not be built when nothing is finalized"),
            apply_restored=lambda restored: None,
            allow_fresh=True,
        )
        == 1
    )


def test_an_explicit_resume_from_that_resolves_to_nothing_raises(tmp_path):
    with pytest.raises(ckpt_mod.CheckpointError, match="Refusing"):
        ckpt_mod.resume(
            str(tmp_path / "elsewhere" / "full"),
            lambda: {},
            apply_restored=lambda restored: None,
            allow_fresh=False,
        )


# --------------------------------------------------------------------------- #
# config surface
# --------------------------------------------------------------------------- #
def test_overrides_must_name_an_existing_key():
    assert parse_overrides(["tpu.steps=50", "tpu.resume=true"]) == {
        "tpu": {"steps": 50, "resume": True}
    }
    assert parse_overrides(["learning_rate=1e-4"])["learning_rate"] == pytest.approx(1e-4)
    config = load_config(TASK, overrides=["tpu.steps=50"])
    assert config.tpu.steps == 50
    with pytest.raises(ConfigError, match="not present in the base config"):
        load_config(TASK, overrides=["tpu.stpes=50"])


def test_batch_geometry_must_divide():
    with pytest.raises(ConfigError, match="must divide by"):
        load_config(TASK, overrides=["tpu.micro_batch_size=48"])


def test_deviating_from_the_official_global_batch_is_allowed_but_announced():
    """Leaving the official effective batch is a decision, not an error — but a loud one.

    This test used to assert that `tpu.global_batch_size=512` raised, and it was wrong
    about the design: `config.py:747-752` says in as many words that deviating from the
    official 16 x 64 = 1024 "is a real change to the recipe, but it is sometimes the right
    call — a run that must fit in a fixed wall clock trades batch for steps. So this is not
    a hard stop; instead train.py prints the comparison at startup, loudly, every time."

    So the contract has two halves, and both are asserted here:

      * the config layer LETS you deviate (no exception), because a run under a wall-clock
        constraint is a legitimate use;
      * `train()` ANNOUNCES it, so no run's loss numbers can be silently compared against a
        reference trained at a different effective batch.

    What actually is a hard stop is geometry, and that lives in its own tests:
    `test_batch_geometry_must_divide` (global must be a whole number of micros) plus the
    per-device and per-host divisibility raised inside `train()`. Together those mean the
    effective batch is always a number somebody chose, which was the real invariant.
    """
    config = load_config(TASK, overrides=["tpu.global_batch_size=512"])
    assert config.tpu.global_batch_size == 512

    # The announcement is emitted inside `train()`, well past the point a unit test can
    # reach without a mesh and a dataset, so it is pinned at the source level -- the same
    # technique `test_latent_space_contract` uses for the latent normalisation, and for the
    # same reason: a warning nobody can see disappearing is exactly the kind of regression
    # that only shows up as a confusing number months later.
    import pathlib

    src = (
        pathlib.Path(__file__).resolve().parents[1]
        / "src"
        / "fastwam_tpu"
        / "training"
        / "train.py"
    ).read_text()
    assert "RECIPE DEVIATION" in src, (
        "train.py no longer announces a global batch that differs from the official "
        "recipe. Deviating is allowed on purpose (config.py:747-752), which is precisely "
        "why the announcement is the only thing standing between a deliberate deviation "
        "and a loss curve nobody can interpret."
    )
    assert "official_batch = 1024" in src, (
        "the official effective batch reference is no longer pinned to 1024; "
        "if the reference recipe changed, update this test deliberately rather than "
        "letting the comparison drift."
    )


def test_scan_layers_cannot_be_turned_off():
    with pytest.raises(ConfigError, match="not a supported configuration"):
        load_config(TASK, overrides=["tpu.scan_layers=false"])
