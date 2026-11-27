"""The training latent space and the deployment latent space must be the SAME space.

This file exists because of a bug that every other test in this repo was structurally
incapable of catching (found 2026-08-06, B-line closed loop 0/9):

    training/train.py:729   latents = vae.normalize_latents(vae.encode(pixels, ...))
    inference/policy.py     vae.encode_frame(tensor, normalize=True, ...)

`encode`'s `normalize` argument DEFAULTS TO TRUE, so the training line normalises twice
and the deployment line once. Both sides were internally consistent -- training trained
in the doubled space, inference served the single space, each with its own passing tests
-- and the only symptom was a checkpoint that scored 0.2421 open-loop instead of 0.0143.

The class of bug is "each side is correct, the pair is not", and the only kind of test
that sees it is one that spans the boundary. Everything below compares the two sides
against each other rather than each against its own expectation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from fastwam_tpu.models.wan import vae as vae_mod


@pytest.fixture(scope="module")
def tiny_vae():
    from flax import nnx

    from tests.test_vae import TINY

    return vae_mod.WanVAE38(TINY, nnx.Rngs(0))


def _training_side(vae, pixels):
    """EXACTLY what `training/train.py::encode_pixels` computes, called the same way.

    This mirrors train.py by hand -- the real one is a closure built inside the training
    loop and cannot be imported -- so it has to be edited whenever that line is. It is
    written the way the FIXED line reads (a single `encode`, whose `normalize` defaults to
    True); the pre-fix line was `normalize_latents(encode(...))`, and restoring that here
    is what turns this test red.
    """
    return vae.encode(pixels, normalize=True, channels_first=True)


def _deployment_side(vae, frame):
    """EXACTLY what `inference/policy.py::_encode_frame` computes."""
    return vae.encode_frame(frame, normalize=True, channels_first=True)


def test_training_and_deployment_agree_on_the_first_frame_latent(tiny_vae):
    """The conditioning latent the model TRAINS on must equal the one it is SERVED.

    The action tower only ever attends to latent frame 0, and at deployment that frame
    comes from `encode_frame` on the single observed image. If the two disagree the model
    is served a space it never saw, which is invisible in every per-side test and costs
    the entire closed-loop score.
    """
    rng = np.random.default_rng(0)
    t = 1 + 4 * 2
    pixels = jnp.asarray(rng.uniform(-1, 1, (1, 3, t, 64, 64)), jnp.float32)

    train_frame0 = _training_side(tiny_vae, pixels)[:, :, 0:1]
    deploy_frame0 = _deployment_side(tiny_vae, pixels[:, :, 0])

    assert train_frame0.shape == deploy_frame0.shape
    gap = float(jnp.max(jnp.abs(train_frame0 - deploy_frame0)))
    scale = float(jnp.max(jnp.abs(deploy_frame0))) + 1e-12
    assert gap / scale < 1e-5, (
        f"the training conditioning latent and the deployment conditioning latent differ "
        f"by {gap / scale:.3e} relative. They are supposed to be the same tensor. Check "
        f"whether one side normalises and the other normalises twice -- `encode`'s "
        f"`normalize` argument defaults to True, so wrapping it in `normalize_latents` "
        f"applies the affine a second time (train.py:729 did exactly this)."
    )


def test_normalize_latents_is_not_idempotent(tiny_vae):
    """Guards the assumption the bug rode in on.

    `normalize_latents(normalize_latents(z)) != normalize_latents(z)` -- so an accidental
    second application is a real change of space, not a no-op. If this ever became
    idempotent the test above would stop being able to see the bug and would need
    rewriting rather than deleting.
    """
    rng = np.random.default_rng(1)
    z = jnp.asarray(rng.normal(size=(1, tiny_vae.config.z_dim, 1, 4, 4)), jnp.float32)
    once = tiny_vae.normalize_latents(z)
    twice = tiny_vae.normalize_latents(once)
    assert float(jnp.max(jnp.abs(twice - once))) > 1e-3


def test_train_py_does_not_normalize_the_latents_twice():
    """Read the REAL training source and check the spelling that caused the incident.

    The numeric test above mirrors `train.py` by hand, because `encode_pixels` is a
    closure inside `train()` and cannot be imported -- which means a regression in
    `train.py` alone would not move it. This one has no such gap: it parses the actual
    file and asserts that the call producing the latents is a bare `encode(...)` and not
    wrapped in `normalize_latents(...)`.

    Together the two cover both halves: the numeric one says the two spaces coincide, and
    this one says the training source still spells it the way that makes them coincide.
    """
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src" / "fastwam_tpu" / "training" / "train.py"
    tree = ast.parse(src.read_text())

    encoders = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "encode_pixels"
    ]
    assert len(encoders) == 1, f"expected exactly one encode_pixels in {src}, found {len(encoders)}"

    offenders = [
        node
        for node in ast.walk(encoders[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "normalize_latents"
    ]
    assert not offenders, (
        f"{src.name}:{offenders[0].lineno} calls `normalize_latents` inside `encode_pixels`. "
        "`encode`'s `normalize` argument already defaults to True, so this applies the "
        "affine a SECOND time and every checkpoint trained from here learns a latent space "
        "inference does not serve. That is exactly what cost run v6e-official-30k its "
        "closed-loop score on 2026-08-06 (0.2421 open-loop vs 0.0143 in the space it was "
        "actually trained in)."
    )


def test_encode_normalize_defaults_to_true():
    """The default that made the bug possible, pinned so a change is a deliberate one.

    Callers read `vae.encode(pixels, channels_first=True)` as "encode", not as "encode and
    normalise", which is why wrapping it in `normalize_latents` looked right.
    """
    import inspect

    sig = inspect.signature(vae_mod.WanVAE38.encode)
    assert sig.parameters["normalize"].default is True
