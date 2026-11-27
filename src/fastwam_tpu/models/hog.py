"""TPU-native HOG image construction for the auxiliary video tower.

The data loader already emits RGB canvases in ``[-1, 1]``.  Building HOG after the
batch reaches the accelerator keeps the GCS format unchanged and, importantly, does
not create a second collection of tiny objects in the bucket.  The representation is
a three-channel, cell-normalised HOG image with the same geometry as the RGB canvas:

* channel 0: normalised unsigned gradient energy;
* channels 1/2: the cosine/sine second angular moments of the 9-bin cell histogram.

Using ``2 * theta`` makes unsigned orientations (theta and theta + pi) identical.  A
2x2-cell L2-Hys normalisation is applied before the three moments are rasterised back
to pixels.  The result is in ``[-1, 1]`` and can therefore be encoded by the same frozen
Wan VAE weights that initialise the RGB path.
"""

import jax
import jax.numpy as jnp

__all__ = ["hog_image"]


def _l2_hys(hist: jax.Array, eps: float) -> jax.Array:
    """2x2-cell L2-Hys, averaged back onto the contributing cells."""
    blocks = jnp.stack(
        (
            hist[:, :, :-1, :-1],
            hist[:, :, 1:, :-1],
            hist[:, :, :-1, 1:],
            hist[:, :, 1:, 1:],
        ),
        axis=-2,
    )
    norm = jnp.sqrt(jnp.sum(blocks * blocks, axis=(-2, -1), keepdims=True) + eps)
    blocks = jnp.minimum(blocks / norm, 0.2)
    blocks = blocks / jnp.sqrt(jnp.sum(blocks * blocks, axis=(-2, -1), keepdims=True) + eps)

    out = jnp.zeros_like(hist)
    count = jnp.zeros(hist.shape[:-1] + (1,), hist.dtype)
    out = out.at[:, :, :-1, :-1].add(blocks[..., 0, :])
    out = out.at[:, :, 1:, :-1].add(blocks[..., 1, :])
    out = out.at[:, :, :-1, 1:].add(blocks[..., 2, :])
    out = out.at[:, :, 1:, 1:].add(blocks[..., 3, :])
    count = count.at[:, :, :-1, :-1].add(1)
    count = count.at[:, :, 1:, :-1].add(1)
    count = count.at[:, :, :-1, 1:].add(1)
    count = count.at[:, :, 1:, 1:].add(1)
    return out / jnp.maximum(count, 1)


def hog_image(
    video: jax.Array,
    *,
    cell_size: int = 8,
    num_bins: int = 9,
    eps: float = 1.0e-6,
) -> jax.Array:
    """Convert ``[B,3,T,H,W]`` RGB video in ``[-1,1]`` to a HOG image.

    The operation is deterministic, contains no trainable state, and is intended to
    be called inside a JIT on TPU.  Height and width must be exact cell multiples; the
    RoboTwin 384x320 canvas satisfies this for the standard 8-pixel cell.
    """
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError(f"video must be [B,3,T,H,W], got {video.shape}")
    if cell_size <= 0 or num_bins <= 1:
        raise ValueError(f"cell_size and num_bins must be positive/useful, got {cell_size}, {num_bins}")
    height, width = video.shape[-2:]
    if height % cell_size or width % cell_size:
        raise ValueError(
            f"HOG canvas {height}x{width} must divide by cell_size={cell_size}"
        )

    x = video.astype(jnp.float32)
    # Channels-first RGB -> [B,T,H,W].  The affine [-1,1] -> [0,1] conversion does not
    # change orientation; it only gives gradient magnitudes an intuitive range.
    rgb = (jnp.transpose(x, (0, 2, 3, 4, 1)) + 1.0) * 0.5
    gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]

    # Centred differences with edge replication.  This is the same [-1,0,1] derivative
    # used by common HOG implementations, without introducing a CPU/OpenCV dependency.
    padded_x = jnp.pad(gray, ((0, 0), (0, 0), (0, 0), (1, 1)), mode="edge")
    padded_y = jnp.pad(gray, ((0, 0), (0, 0), (1, 1), (0, 0)), mode="edge")
    gx = padded_x[..., 2:] - padded_x[..., :-2]
    gy = padded_y[:, :, 2:, :] - padded_y[:, :, :-2, :]
    magnitude = jnp.sqrt(jnp.maximum(gx * gx + gy * gy, 0.0))
    angle = jnp.mod(jnp.arctan2(gy, gx), jnp.pi)

    # Linear interpolation between neighbouring unsigned orientation bins.
    bin_pos = angle * (num_bins / jnp.pi)
    lower = jnp.floor(bin_pos).astype(jnp.int32) % num_bins
    frac = bin_pos - jnp.floor(bin_pos)
    upper = (lower + 1) % num_bins
    votes = magnitude[..., None] * (
        jax.nn.one_hot(lower, num_bins, dtype=jnp.float32) * (1.0 - frac[..., None])
        + jax.nn.one_hot(upper, num_bins, dtype=jnp.float32) * frac[..., None]
    )

    b, t = video.shape[0], video.shape[2]
    hc, wc = height // cell_size, width // cell_size
    votes = votes.reshape(b, t, hc, cell_size, wc, cell_size, num_bins)
    hist = jnp.sum(votes, axis=(3, 5))
    hist = _l2_hys(hist, eps)

    centres = (jnp.arange(num_bins, dtype=jnp.float32) + 0.5) * (jnp.pi / num_bins)
    energy = jnp.sum(hist, axis=-1)
    cos2 = jnp.sum(hist * jnp.cos(2.0 * centres), axis=-1)
    sin2 = jnp.sum(hist * jnp.sin(2.0 * centres), axis=-1)
    scale = jnp.maximum(jnp.max(energy, axis=(-2, -1), keepdims=True), eps)
    cells = jnp.stack((2.0 * energy / scale - 1.0, cos2 / scale, sin2 / scale), axis=2)
    pixels = jnp.repeat(jnp.repeat(cells, cell_size, axis=-2), cell_size, axis=-1)
    pixels = jnp.transpose(pixels, (0, 2, 1, 3, 4))
    return jnp.clip(pixels, -1.0, 1.0).astype(video.dtype)
