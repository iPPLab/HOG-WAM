"""Controlled RGB-input perturbations for FastWAM closed-loop ablations.

The evaluator should call :func:`perturb_rgb` immediately after composing the
384x320 RGB canvas and immediately before ``policy.infer_action``.  The clean
canvas is never modified in place.  A fixed ``seed`` makes a run reproducible;
the evaluator can derive it from ``episode_seed`` and ``replan_index``.

This module intentionally operates on uint8 RGB pixels.  Applying noise after
normalisation would test a different failure mode and would not reproduce
sensor/render noise at the model input boundary.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

PerturbationMode = Literal[
    "gaussian",
    "speckle",
    "mixed",
    "exposure",
    "color_filter",
    "fog",
    "contrast",
]


@dataclass(frozen=True)
class RGBPerturbation:
    """A reproducible, pixel-space perturbation profile.

    ``gaussian_sigma`` is in uint8 levels.  ``speckle_std`` is a multiplicative
    standard deviation (0.04 means roughly 4% intensity variation).  The
    default ``mixed`` profile is intentionally mild enough for an ablation,
    not a claim that it is a calibrated OIDN failure distribution.
    """

    mode: PerturbationMode = "mixed"
    gaussian_sigma: float = 6.0
    speckle_std: float = 0.025
    exposure_ev: float = 0.0
    filter_rgb: tuple[float, float, float] = (1.20, 0.82, 0.68)
    filter_strength: float = 0.0
    fog_density: float = 0.0
    fog_rgb: tuple[float, float, float] = (235.0, 240.0, 245.0)
    contrast_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in (
            "gaussian", "speckle", "mixed", "exposure", "color_filter", "fog", "contrast"
        ):
            raise ValueError(f"unsupported perturbation mode: {self.mode}")
        if self.gaussian_sigma < 0 or self.speckle_std < 0:
            raise ValueError("noise scales must be non-negative")
        if not 0.0 <= self.filter_strength <= 1.0:
            raise ValueError("filter_strength must be in [0, 1]")
        if not 0.0 <= self.fog_density <= 1.0:
            raise ValueError("fog_density must be in [0, 1]")
        if self.contrast_factor < 0:
            raise ValueError("contrast_factor must be non-negative")
        if len(self.filter_rgb) != 3 or len(self.fog_rgb) != 3:
            raise ValueError("filter_rgb and fog_rgb must have three channels")


def perturb_rgb(
    canvas: np.ndarray,
    *,
    seed: int,
    profile: RGBPerturbation = RGBPerturbation(),
) -> np.ndarray:
    """Return a perturbed copy of a uint8 ``H x W x 3`` RGB canvas.

    The operation is deterministic for a fixed ``canvas``, ``seed`` and
    profile.  It preserves shape, dtype and the [0, 255] range.  The original
    array is never changed.
    """

    image = np.asarray(canvas)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected uint8 RGB [H,W,3], got {image.dtype} {image.shape}")

    rng = np.random.default_rng(seed)
    work = image.astype(np.float32, copy=True)
    if profile.mode in ("gaussian", "mixed") and profile.gaussian_sigma:
        work += rng.normal(0.0, profile.gaussian_sigma, size=work.shape).astype(np.float32)
    if profile.mode in ("speckle", "mixed") and profile.speckle_std:
        work *= 1.0 + rng.normal(0.0, profile.speckle_std, size=work.shape).astype(np.float32)
    if profile.mode == "exposure":
        # EV is a photographic exposure stop: -1 means half the light and +1
        # means twice the light. Clipping is intentional because saturation is
        # part of the exposure failure mode.
        work *= float(2.0 ** profile.exposure_ev)
    elif profile.mode == "color_filter":
        tint = np.asarray(profile.filter_rgb, dtype=np.float32).reshape(1, 1, 3)
        work *= (1.0 - profile.filter_strength) + profile.filter_strength * tint
    elif profile.mode == "fog":
        veil = np.asarray(profile.fog_rgb, dtype=np.float32).reshape(1, 1, 3)
        work = (1.0 - profile.fog_density) * work + profile.fog_density * veil
    elif profile.mode == "contrast":
        work = (work - 127.5) * profile.contrast_factor + 127.5
    return np.clip(np.rint(work), 0.0, 255.0).astype(np.uint8)


PAPER_PROFILES = {
    "mixed": RGBPerturbation("mixed", gaussian_sigma=10.0, speckle_std=0.05),
    "warm": RGBPerturbation("color_filter", filter_strength=0.25),
    "underexposure": RGBPerturbation("exposure", exposure_ev=-0.5),
    "contrast": RGBPerturbation("contrast", contrast_factor=0.80),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="input RGB image")
    parser.add_argument("output", type=Path, help="perturbed RGB image")
    parser.add_argument("--comparison", type=Path, help="optional side-by-side PNG")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument(
        "--mode",
        choices=("gaussian", "speckle", "mixed", "exposure", "color_filter", "fog", "contrast"),
        default="mixed",
    )
    parser.add_argument("--sigma", type=float, default=6.0)
    parser.add_argument("--speckle-std", type=float, default=0.025)
    parser.add_argument("--exposure-ev", type=float, default=0.0)
    parser.add_argument("--filter-strength", type=float, default=0.0)
    parser.add_argument("--fog-density", type=float, default=0.0)
    parser.add_argument("--contrast-factor", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    # Pillow is kept as a CLI-only dependency; the evaluator integration only
    # imports perturb_rgb and therefore remains numpy-only.
    from PIL import Image, ImageDraw

    args = _parse_args()
    clean = np.asarray(Image.open(args.input).convert("RGB"), dtype=np.uint8)
    profile = RGBPerturbation(
        args.mode,
        args.sigma,
        args.speckle_std,
        exposure_ev=args.exposure_ev,
        filter_strength=args.filter_strength,
        fog_density=args.fog_density,
        contrast_factor=args.contrast_factor,
    )
    noisy = perturb_rgb(clean, seed=args.seed, profile=profile)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(noisy, "RGB").save(args.output)
    if args.comparison:
        comparison = Image.new("RGB", (clean.shape[1] * 2, clean.shape[0] + 28), "white")
        left = Image.fromarray(clean, "RGB")
        right = Image.fromarray(noisy, "RGB")
        comparison.paste(left, (0, 28))
        comparison.paste(right, (clean.shape[1], 28))
        draw = ImageDraw.Draw(comparison)
        draw.text((8, 7), "clean RGB", fill="black")
        draw.text((clean.shape[1] + 8, 7), f"{args.mode}, seed={args.seed}", fill="black")
        args.comparison.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(args.comparison)


if __name__ == "__main__":
    main()
