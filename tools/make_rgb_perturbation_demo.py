"""Create a clearly labelled synthetic RoboTwin-shaped RGB canvas for QA demos.

This is not an evaluation frame.  It mirrors the official 384x320 canvas
layout so the perturbation visualization can be inspected when no simulator
frame was archived locally.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_canvas() -> np.ndarray:
    h, w = 384, 320
    y, x = np.mgrid[:h, :w]
    image = np.zeros((h, w, 3), np.float32)
    # head-camera tile: tabletop gradient and a few high-contrast objects
    head = y < 256
    image[..., 0] = 112 + 18 * y / h
    image[..., 1] = 92 + 14 * y / h
    image[..., 2] = 68 + 9 * y / h
    image[head & (x > 35) & (x < 82) & (y > 100) & (y < 205)] = (190, 42, 34)
    image[head & (x > 180) & (x < 235) & (y > 65) & (y < 165)] = (46, 116, 182)
    image[head & (x > 245) & (x < 285) & (y > 135) & (y < 220)] = (225, 188, 64)
    # wrist tiles: different viewpoints at the bottom of the official canvas
    image[256:384, :160] = (58, 74, 82)
    image[256:384, 160:] = (70, 64, 82)
    image[285:355, 42:102] = (184, 52, 44)
    image[272:343, 209:280] = (54, 136, 180)
    image += 3.0 * np.sin(x[..., None] / 13.0)
    return np.clip(image, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(make_canvas(), "RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 319, 383), outline=(245, 245, 245), width=1)
    image.save(args.output)


if __name__ == "__main__":
    main()
