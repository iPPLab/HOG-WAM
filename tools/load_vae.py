"""CLI to inspect / smoke-check a converted Wan2.2 VAE checkpoint.

Deliberately NOT part of tools/convert_weights.py. That script's job is torch ->
Orbax conversion, and the VAE loader consumes the published converted Wan2.2 VAE Orbax
tree, whose keys map 1:1 onto this module. So the VAE
side of M1 is a *loader*, and the loader itself lives in the library
(``fastwam_tpu.models.wan.vae.load_vae``) because M3 inference and M4 materialization
both import it. This file is only the human-facing wrapper.

    uv run python tools/load_vae.py --ckpt /path/to/vae_orbax
    uv run python tools/load_vae.py --ckpt <dir> --smoke --frames 9 --height 384 --width 320

``--smoke`` runs a real encode/decode: fine on TPU, minutes per call on a laptop CPU
at the RoboTwin canvas, so start small.
"""

import argparse
import sys
import time


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt", required=True, help="Orbax dir holding the VAE (local or gs://)")
    parser.add_argument("--smoke", action="store_true", help="run a real encode/decode on random pixels")
    parser.add_argument("--frames", type=int, default=9, help="pixel frames for --smoke (must be 1+4m)")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--config", help="optional training recipe for geometry checking")
    args = parser.parse_args(argv)

    import jax
    import jax.numpy as jnp

    from fastwam_tpu.models.wan.vae import load_vae, read_vae_sidecar

    sidecar = read_vae_sidecar(args.ckpt)
    print(f"sidecar   {args.ckpt.rstrip('/')}/vae_config.json")
    for key in sorted(sidecar):
        value = sidecar[key]
        if isinstance(value, list) and len(value) > 8:
            value = f"[{value[0]}, {value[1]}, ... ] ({len(value)} values)"
        print(f"  {key:22s} {value}")

    started = time.time()
    vae = load_vae(args.ckpt)
    print(f"loaded    in {time.time() - started:.1f}s")
    print(
        f"  z_dim {vae.z_dim}  spatial /{vae.spatial_downsample_factor}  "
        f"temporal /{vae.temporal_downsample_factor}"
    )

    _, state = __import__("flax").nnx.split(vae)
    leaves = jax.tree.leaves(__import__("flax").nnx.to_pure_dict(state))
    print(f"  {len(leaves)} tensors, {sum(int(x.size) for x in leaves) / 1e6:.2f} M params")

    print(f"  latents for [B,3,{args.frames},{args.height},{args.width}] -> "
          f"[B,{','.join(str(v) for v in vae.latent_shape(args.frames, args.height, args.width))}]")

    if args.config:
        from fastwam_tpu import load_config

        vae.check_geometry(load_config(args.config).shapes)
        print("  geometry agrees with the selected training recipe")

    if args.smoke:
        video = jax.random.uniform(
            jax.random.key(0), (1, 3, args.frames, args.height, args.width), minval=-1.0, maxval=1.0
        )
        started = time.time()
        latents = vae.encode(video)
        print(f"encode    {video.shape} -> {latents.shape} in {time.time() - started:.1f}s "
              f"(mean {float(latents.mean()):+.4f}, std {float(latents.std()):.4f})")

        started = time.time()
        frame = vae.encode_frame(video[:, :, :1])
        same = float(jnp.abs(frame - latents[:, :, :1]).max())
        print(f"frame     -> {frame.shape} in {time.time() - started:.1f}s; "
              f"max|delta| vs latent frame 0 = {same:.3e} (must be 0)")

        started = time.time()
        pixels = vae.decode(latents)
        print(f"decode    {latents.shape} -> {pixels.shape} in {time.time() - started:.1f}s "
              f"(range [{float(pixels.min()):.3f}, {float(pixels.max()):.3f}])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
