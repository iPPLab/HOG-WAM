"""Continuous-time flow matching, shift = 5.0. Every formula is copied, not adapted.

M2 implements this (training path); M3 uses build_inference_schedule.

Semantics from
    reference/FastWAM/src/fastwam/models/wan22/schedulers/scheduler_continuous.py
        class WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0, eps=1e-10)
Shape/API reference (not semantics)
    reference/maxdiffusion/src/maxdiffusion/schedulers/scheduling_flow_match_flax.py

Formulas (spec §2.9)
    phi(u, s)               = s*u / (1 + (s-1)*u)                     # s = 5.0
    sample_training_t()     : u ~ U(0,1); sigma = phi(u, 5); t = sigma * 1000
    add_noise(x0, n, t)     : sigma = t/1000; x_t = (1-sigma)*x0 + sigma*n
    training_target(x0, n)  : n - x0                                  # velocity
    training_weight(t)      : y = exp(-2*((t-500)/1000)**2)
                              w = (y - Y_MIN) / (WEIGHT_NORM_CONST + 1e-10)
    inference_schedule(N)   : u = linspace(1, 0, N+1); sigma = phi(u, shift)
                              timesteps = sigma[:-1]*1000; deltas = sigma[1:]-sigma[:-1]  # negative
    step(v, delta, x)       : x + v*delta

The two constants below are evaluated on the reference's 1000-point grid. Hardcode
them rather than recomputing, so the two implementations agree bit-for-bit
(spec §9.3 item 15). Sanity: w(500) = 2.076404, w(0) = w(1000) = 0 — the weight is a
normalized Gaussian bump that zeroes out both noise extremes.

Everything here is a free function on jax arrays: the reference "scheduler" carries no
state beyond (num_train_timesteps, shift) and the two constants, and threading a
stateful object through a jitted train step buys nothing. Both callers pass the config
values explicitly, which also makes the video/action pair — two schedulers with
identical settings but INDEPENDENT timestep draws — impossible to accidentally share.

`sample_training_t` draws u in fp32 exactly like the reference (`torch.rand` in fp32,
then phi, then *1000). The stream itself cannot match torch's, so anything comparing
against the golden passes its own timesteps in rather than sampling.
"""

import jax
import jax.numpy as jnp

__all__ = [
    "Y_MIN",
    "WEIGHT_NORM_CONST",
    "WEIGHT_EPS",
    "phi",
    "sample_training_t",
    "add_noise",
    "training_target",
    "training_weight",
    "build_inference_schedule",
    "step",
]

Y_MIN = 0.6065306597  # exp(-0.5), attained at t=1000
WEIGHT_NORM_CONST = 0.1894955536
WEIGHT_EPS = 1e-10  # scheduler_continuous.py's `eps`, only ever used in this denominator


class SchedulerError(ValueError):
    """A scheduler argument that cannot be a correct FastWAM call."""


def phi(u: jax.Array, shift: float) -> jax.Array:
    """s*u / (1 + (s-1)*u) — the shift reparameterisation, `_phi` in the reference."""
    if shift <= 0:
        raise SchedulerError(f"shift must be positive, got {shift}")
    return shift * u / (1.0 + (shift - 1.0) * u)


def sample_training_t(
    rng: jax.Array, batch: int, shift: float, num_train_timesteps: int, copies: int = 1
) -> jax.Array:
    """-> [B] float32 in [0, num_train_timesteps). Video and action draw INDEPENDENTLY.

    `sample_training_t` (scheduler_continuous.py:31-37): u ~ U(0,1) in fp32, sigma =
    phi(u, shift), t = sigma * num_train_timesteps.

    `copies > 1` returns [B, copies] instead: one INDEPENDENT flow time per action copy
    (`copies.py`). The default draws `(batch,)`, i.e. the reference's line untouched.
    """
    if batch <= 0:
        raise SchedulerError(f"batch must be positive, got {batch}")
    if copies < 1:
        raise SchedulerError(f"copies must be positive, got {copies}")
    shape = (batch,) if copies == 1 else (batch, copies)
    u = jax.random.uniform(rng, shape, dtype=jnp.float32)
    return phi(u, shift) * float(num_train_timesteps)


def _broadcast_right(sigma: jax.Array, rank: int) -> jax.Array:
    """[B, ...] -> [B, ..., 1, 1] with `rank` total dims (the reference's `.view(-1, 1...)`).

    Rank-1 sigma reproduces the reference exactly; rank 2 ([B, S], one flow time per row)
    is the per-copy path and lands the trailing axes on the feature dims.
    """
    return sigma.reshape(*sigma.shape, *([1] * (rank - sigma.ndim)))


def add_noise(x0: jax.Array, noise: jax.Array, t: jax.Array, num_train_timesteps: int) -> jax.Array:
    """x0/noise: [B, ...]; t: [B] (or [B, S], one per row) -> same shape as x0.

    sigma = t / num_train_timesteps; x_t = (1 - sigma) * x0 + sigma * noise.
    """
    if x0.shape != noise.shape:
        raise SchedulerError(f"x0 {x0.shape} and noise {noise.shape} must match")
    if t.ndim not in (1, 2) or t.ndim > x0.ndim or t.shape != x0.shape[: t.ndim]:
        raise SchedulerError(
            f"t must be [B] or [B, S] agreeing with x0's leading dims {x0.shape}, got {t.shape}"
        )
    sigma = _broadcast_right((t / float(num_train_timesteps)).astype(x0.dtype), x0.ndim)
    return (1 - sigma) * x0 + sigma * noise


def training_target(x0: jax.Array, noise: jax.Array) -> jax.Array:
    """noise - x0, independent of t (`training_target`, scheduler_continuous.py:58-61)."""
    if x0.shape != noise.shape:
        raise SchedulerError(f"x0 {x0.shape} and noise {noise.shape} must match")
    return noise - x0


def training_weight(t: jax.Array, num_train_timesteps: int = 1000) -> jax.Array:
    """[B] -> [B] loss weight, computed in fp32 like the reference.

    `num_train_timesteps` only enters through the Gaussian's centre and width; Y_MIN /
    WEIGHT_NORM_CONST are the reference's values for the 1000-step grid and are the
    reason this is not parameterised further — see the module docstring.
    """
    if num_train_timesteps != 1000:
        raise SchedulerError(
            f"Y_MIN / WEIGHT_NORM_CONST are hardcoded for num_train_timesteps=1000, "
            f"got {num_train_timesteps}; recompute them before changing this"
        )
    steps = float(num_train_timesteps)
    y = jnp.exp(-2.0 * ((t.astype(jnp.float32) - steps / 2.0) / steps) ** 2)
    return (y - Y_MIN) / (WEIGHT_NORM_CONST + WEIGHT_EPS)


def build_inference_schedule(
    num_steps: int, shift: float, num_train_timesteps: int
) -> tuple[jax.Array, jax.Array]:
    """-> (timesteps [N], deltas [N]); deltas are negative.

    u = linspace(1, 0, N+1); sigma = phi(u, shift); timesteps = sigma[:-1]*T;
    deltas = sigma[1:] - sigma[:-1]. Built in fp32, matching the reference.
    """
    if num_steps <= 0:
        raise SchedulerError(f"num_steps must be positive, got {num_steps}")
    u = jnp.linspace(1.0, 0.0, num_steps + 1, dtype=jnp.float32)
    sigma = phi(u, shift)
    return sigma[:-1] * float(num_train_timesteps), sigma[1:] - sigma[:-1]


def step(velocity: jax.Array, delta: jax.Array, x: jax.Array) -> jax.Array:
    """x + velocity * delta. `delta` is a scalar (one schedule shared by the batch)."""
    if velocity.shape != x.shape:
        raise SchedulerError(f"velocity {velocity.shape} and x {x.shape} must match")
    return x + velocity * delta.astype(x.dtype)
