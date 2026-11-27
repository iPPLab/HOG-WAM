"""D5 + colour order: settle the two open data conventions against the official checkpoint.

Two switches in the internal `specs/robotwin.yaml` are conventions, not facts, and neither one
raises when it is wrong:

  * `data.l3.action_shift` — 1 means `action[j] = qpos[start+j+1]`, i.e. the action leads
    the state by one frame, which is what the official's T-1 rows per episode imply.
    0 means they are aligned. One frame of slip cost 48 points of success rate once
    (`project_anchor_gap_fix`), and it is invisible offline unless you go looking.
  * `data.l3.source_color_order` — "bgr" reverses the decoded channels. RoboTwin baked its
    JPEGs through cv2, which encodes BGR, so a plain decode returns BGR; the released
    checkpoint trained on RGB. A missing flip and a doubled flip look identical in a
    histogram and produce the same shapes.

The discriminator: a checkpoint's training loss is low only under the convention it was
trained with. So run the official checkpoint's own training loss over the same windows
under all four combinations, with the SAME timesteps and the SAME noise draw in every
one, and read off which corner is lowest. Everything except the two switches is held
fixed, so the four numbers differ for exactly one reason.

This is also the first end-to-end join of the official checkpoint with our loss pipeline:
if the numbers come out uniformly enormous, the problem is upstream of the conventions.

    python -m tools.discriminate_conventions --config configs/training/full.example.yaml \
        --ckpt <orbax-root> --windows 128 data.l3.pool_size=256

Run it the way the trainer runs (all hosts of the slice, jax.distributed), because that is
the code path the B line will use.
"""

import argparse
import dataclasses
import itertools
import json
import sys
import time

import numpy as np

def log(*parts) -> None:
    import jax

    print(f"[probe p{jax.process_index()}]", *parts, flush=True)


def load_official(root: str, config, *, dtype, scan_layers: bool):
    """The converted official checkpoint -> a FastWAM, on the host.

    Three sibling Orbax trees written by `tools/convert_weights.py convert-official`:
    `<root>/video`, `<root>/action`, `<root>/proprio_encoder`. Unlike the training init,
    the proprio encoder is a TRAINED tensor here and must be restored — leaving it random
    would make every number below meaningless.

    `inference/load.py::load_fastwam` refuses `scan_layers=True` because stacking 30 blocks
    on a 24 GB card holds two copies of a 12 GB tree. A TPU host has hundreds of GB, so
    that constraint does not apply and the towers' own loaders (which stack after the
    restore) are used instead.
    """
    import jax
    import numpy as np
    from flax import nnx

    from fastwam_tpu.models.fastwam import FastWAM
    from fastwam_tpu.training.train import build_model

    root = root.rstrip("/")
    weights = dataclasses.replace(
        config.model.weights, video_dit_orbax=f"{root}/video", action_dit_init=f"{root}/action"
    )
    official = dataclasses.replace(config, model=dataclasses.replace(config.model, weights=weights))
    _, _, host_tree = build_model(
        official,
        dtype=dtype,
        param_dtype=dtype,
        scan_layers=scan_layers,
        remat_policy="none",
        seed=0,
    )

    from fastwam_tpu.training.train import restore_tree_to_host

    with jax.default_device(jax.local_devices(backend="cpu")[0]):
        blank = FastWAM.from_config(
            official, rngs=nnx.Rngs(0), dtype=dtype, param_dtype=dtype,
            scan_layers=scan_layers, remat_policy="none",
        )
        graphdef, state = nnx.split(blank)
        pure = host_tree
        # Same host restore as the towers: under jax.distributed orbax will not accept an
        # unsharded target, and the trained proprio encoder must not be left random.
        restored = restore_tree_to_host(f"{root}/proprio_encoder", pure["proprio_encoder"])
        if sorted(restored) != sorted(pure["proprio_encoder"]):
            raise ValueError(
                f"{root}/proprio_encoder has keys {sorted(restored)}, the model wants "
                f"{sorted(pure['proprio_encoder'])}"
            )
        pure["proprio_encoder"] = jax.tree.map(lambda x: np.asarray(x, np.dtype(dtype)), restored)
        nnx.replace_by_pure_dict(state, pure)
    log(f"official checkpoint loaded from {root} (proprio_encoder restored, not random)")
    return graphdef, state, pure


def build_dataset(config, *, action_shift: int, color_order: str, host_id: int, n_hosts: int, fs):
    from fastwam_tpu.data.loader import RobotwinTwpDataset

    node = config.data.train
    l3 = config.data.l3
    return RobotwinTwpDataset(
        l3.prefix,
        stats_blob=fs.cat_file(l3.stats_path.removeprefix("gs://")),
        stats_source=l3.stats_path,
        stats_sha256=l3.stats_sha256,
        num_frames=node.num_frames,
        action_video_freq_ratio=node.action_video_freq_ratio,
        video_size=node.video_size,
        action_shift=action_shift,
        source_color_order=color_order,
        text_mode=l3.text_mode,
        text_dir=l3.text_dir,
        context_len=node.context_len,
        val_set_proportion=node.val_set_proportion,
        split="train",
        episode_weighting=l3.episode_weighting,
        pool_size=l3.pool_size,
        pool_rate=l3.pool_rate,
        pool_prefetch_threads=l3.pool_prefetch_threads,
        decode_threads=l3.decode_threads,
        seed=config.seed,
        host_id=host_id,
        n_hosts=n_hosts,
        fs=fs,
        expect_index_fingerprint=l3.index_fingerprint,
    ).start()


def main(argv=None) -> None:
    from fastwam_tpu.config import load_config

    parser = argparse.ArgumentParser(prog="python -m tools.discriminate_conventions")
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted config assignments")
    parser.add_argument("--ckpt", required=True, help="the converted official checkpoint root")
    parser.add_argument("--windows", type=int, default=128, help="global windows per combination")
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="evaluate at this FIXED noise level instead of sampling from the training "
        "distribution. Use a small value (0.1-0.3). With shift=5 the training draw "
        "concentrates at high sigma, where the target is nearly pure noise and the loss "
        "barely responds to the data convention at all — which is what a first run of this "
        "probe measured: 1%% between colour orders. At low sigma the model has to "
        "reconstruct, so a wrong convention has somewhere to show up.",
    )
    parser.add_argument("--out", default=None, help="write the table as json here (chief only)")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    config = load_config(args.config, overrides=args.overrides)

    import jax

    cache = config.tpu.gcs.jax_cache_dir
    if cache:
        jax.config.update("jax_compilation_cache_dir", cache)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 5)
    if config.tpu.multihost:
        jax.distributed.initialize()

    import jax.numpy as jnp
    from flax import nnx

    from fastwam_tpu.data.pool import open_filesystem, stable_seed
    from fastwam_tpu.training import sharding
    from fastwam_tpu.training.train import load_frozen_vae

    dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[config.tpu.compute_dtype]
    mesh = sharding.make_mesh(config)
    sharding.install(mesh, constraints=config.tpu.sharding_constraints)
    n_hosts, host_id = jax.process_count(), jax.process_index()
    is_chief = host_id == 0
    log(sharding.describe_mesh(mesh))

    batch = config.tpu.micro_batch_size
    if batch % n_hosts:
        raise ValueError(f"tpu.micro_batch_size {batch} must divide over {n_hosts} hosts")
    host_batch = batch // n_hosts
    rounds = max(1, -(-args.windows // batch))
    log(f"{rounds} round(s) x {batch} windows = {rounds * batch} windows per combination")

    graphdef, state, host_tree = load_official(
        args.ckpt, config, dtype=dtype, scan_layers=config.tpu.scan_layers
    )
    nnx.replace_by_pure_dict(
        state, sharding.shard_params(host_tree, mesh, shard=config.tpu.shard_params)
    )
    del host_tree

    vae = load_frozen_vae(config, dtype=dtype)
    vae_graphdef, vae_state = nnx.split(vae)
    nnx.replace_by_pure_dict(vae_state, sharding.replicate(nnx.to_pure_dict(vae_state), mesh))
    vae = nnx.merge(vae_graphdef, vae_state)

    @jax.jit
    def encode(pixels):
        return sharding.shard_batch(
            vae.normalize_latents(vae.encode(pixels, channels_first=True)).astype(jnp.float32)
        )

    @jax.jit
    def evaluate(param_tree, micro, timesteps, noise):
        """The training loss with the timesteps and the noise SUPPLIED, not sampled.

        That is what makes the four combinations comparable: identical t and identical
        epsilon, so the only thing that differs between them is the data convention.
        """
        merged = nnx.merge(graphdef, param_tree)
        _, metrics = merged.training_loss(micro, jax.random.key(0), timesteps=timesteps, noise=noise)
        return metrics["loss_video"], metrics["loss_action"]

    shapes = config.shapes
    fs = open_filesystem(config.data.l3.prefix)

    # One draw of (t_video, t_action, noise_video, noise_action) per round, reused by every
    # combination. Sampled from the training distribution so the numbers are on the same
    # scale as a real training loss rather than at some arbitrary noise level.
    from fastwam_tpu.schedulers import flow_match

    fixed = []
    for r in range(rounds):
        key = jax.random.fold_in(jax.random.key(20260804), r)
        k_tv, k_ta, k_nv, k_na = jax.random.split(key, 4)
        if args.sigma is None:
            t_video = flow_match.sample_training_t(
                k_tv, batch, config.model.video_scheduler.train_shift,
                config.model.video_scheduler.num_train_timesteps,
            )
            t_action = flow_match.sample_training_t(
                k_ta, batch, config.model.action_scheduler.train_shift,
                config.model.action_scheduler.num_train_timesteps,
            )
        else:
            t_video = jnp.full(
                (batch,), args.sigma * config.model.video_scheduler.num_train_timesteps, jnp.float32
            )
            t_action = jnp.full(
                (batch,), args.sigma * config.model.action_scheduler.num_train_timesteps, jnp.float32
            )
        noise_video = jax.random.normal(k_nv, (batch, *shapes.latent_shape), jnp.float32)
        noise_action = jax.random.normal(
            k_na, (batch, shapes.action_tokens, config.model.action_dit_config.action_dim), jnp.float32
        )
        fixed.append(((t_video, t_action), (noise_video, noise_action)))

    log(
        "noise level: "
        + (
            f"FIXED sigma={args.sigma}"
            if args.sigma is not None
            else "sampled from the training distribution (shift=5, so mostly HIGH noise)"
        )
    )
    table = {}
    for action_shift, color_order in itertools.product((1, 0), ("bgr", "rgb")):
        started = time.time()
        dataset = build_dataset(
            config,
            action_shift=action_shift,
            color_order=color_order,
            host_id=host_id,
            n_hosts=n_hosts,
            fs=fs,
        )
        # The same seed in every combination, so every combination sees the same episodes
        # and the same window starts. Neither switch touches the draw.
        rng = np.random.default_rng(stable_seed(config.seed, "probe", host_id))
        video_losses, action_losses = [], []
        for r in range(rounds):
            sample = dataset.sample_batch(rng, host_batch)
            micro = {
                "latents": encode(sharding.global_from_host_local(sample["video"], mesh)),
                "action": sharding.global_from_host_local(sample["action"], mesh),
                "proprio": sharding.global_from_host_local(sample["proprio"], mesh),
                "context": sharding.global_from_host_local(
                    sample["context"].astype(np.float32), mesh
                ).astype(dtype),
                "context_mask": sharding.global_from_host_local(sample["context_mask"], mesh),
                "image_is_pad": sharding.global_from_host_local(sample["image_is_pad"], mesh),
                "action_is_pad": sharding.global_from_host_local(sample["action_is_pad"], mesh),
            }
            loss_video, loss_action = evaluate(state, micro, *fixed[r])
            video_losses.append(float(loss_video))
            action_losses.append(float(loss_action))
        dataset.close()
        key = f"action_shift={action_shift},color={color_order}"
        table[key] = {
            "loss_video": float(np.mean(video_losses)),
            "loss_action": float(np.mean(action_losses)),
            "loss": float(np.mean(video_losses) + np.mean(action_losses)),
            "windows": rounds * batch,
            "seconds": round(time.time() - started, 1),
        }
        log(f"{key}: {json.dumps(table[key])}")

    if is_chief:
        ranked = sorted(table.items(), key=lambda kv: kv[1]["loss"])
        print("\n=== D5 + colour discriminator (official checkpoint, teacher-forced) ===", flush=True)
        print(f"{'combination':34s} {'loss':>9s} {'video':>9s} {'action':>9s}", flush=True)
        for name, row in ranked:
            print(
                f"{name:34s} {row['loss']:9.4f} {row['loss_video']:9.4f} {row['loss_action']:9.4f}",
                flush=True,
            )
        best, second = ranked[0], ranked[1]
        margin = second[1]["loss"] / max(best[1]["loss"], 1e-9)
        print(
            f"\nlowest: {best[0]}  ({margin:.2f}x better than the runner-up {second[0]})",
            flush=True,
        )
        if margin < 1.05:
            print(
                "WARNING: the margin is under 5%. That is not a decision — it means the two "
                "switches barely move this loss, so do NOT lock a convention on it.",
                flush=True,
            )
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(table, handle, indent=1)
            print(f"written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
