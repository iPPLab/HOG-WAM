"""Numerical and structural gates for the removable HOG auxiliary tower."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from test_mot_parity import (
    BATCH,
    CONTEXT_LEN,
    build_action,
    build_video,
    load_flat_params,
    sample_inputs,
)

from fastwam_tpu.models.hog import hog_image
from fastwam_tpu.models.mot import MoT, build_hog_mot_attention_mask, build_mot_attention_mask


def test_hog_image_shape_range_and_flat_image():
    flat = jnp.zeros((2, 3, 3, 32, 24), jnp.float32)
    got = hog_image(flat)
    assert got.shape == flat.shape
    assert got.dtype == flat.dtype
    assert np.isfinite(np.asarray(got)).all()
    assert float(got.min()) >= -1.0 and float(got.max()) <= 1.0
    # No gradient energy: channel 0 is background and angular moments are zero.
    np.testing.assert_allclose(np.asarray(got[:, 0]), -1.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(got[:, 1:]), 0.0, atol=1e-5)


def test_hog_image_distinguishes_horizontal_and_vertical_edges():
    horizontal = jnp.zeros((1, 3, 1, 32, 24), jnp.float32).at[:, :, :, 16:].set(1.0)
    vertical = jnp.zeros((1, 3, 1, 32, 24), jnp.float32).at[:, :, :, :, 12:].set(1.0)
    h = np.asarray(hog_image(horizontal))
    v = np.asarray(hog_image(vertical))
    assert not np.allclose(h, v)
    # cos(2 theta) changes sign between 0 and pi/2 orientations.
    assert np.sign(h[:, 1].mean()) != np.sign(v[:, 1].mean())


def test_three_tower_mask_keeps_rgb_and_action_hog_blind():
    sv, sa, per_frame = 36, 8, 12
    mask = np.asarray(build_hog_mot_attention_mask(sv, sv, sa, per_frame))
    action = 2 * sv
    assert mask.shape == (2 * sv + sa, 2 * sv + sa)
    assert not mask[:sv, sv:].any()  # RGB sees neither HOG nor Action.
    assert not mask[action:, sv:action].any()  # Action never sees HOG.
    assert mask[action:, :per_frame].all()
    assert mask[action:, action:].all()
    # HOG sees causally equivalent RGB and HOG blocks.
    np.testing.assert_array_equal(mask[sv:action, :sv], mask[:sv, :sv])
    np.testing.assert_array_equal(mask[sv:action, sv:action], mask[:sv, :sv])


def test_three_tower_rgb_action_outputs_equal_classic_path():
    video = build_video()
    hog = build_video()
    action = build_action()
    mot = MoT(video=video, hog=hog, action=action)
    latents, act, context, _, t_video, t_action = sample_inputs()
    hog_latents = jax.random.normal(jax.random.key(91), latents.shape)
    context_mask = jnp.ones((BATCH, CONTEXT_LEN), bool)

    vs = video.pre_dit(latents, t_video, context, context_mask)
    hs = hog.pre_dit(hog_latents, t_video + 17, context, context_mask)
    acs = action.pre_dit(act, t_action, context, context_mask)
    classic_mask = build_mot_attention_mask(vs.tokens.shape[1], acs.tokens.shape[1], vs.tokens_per_frame)
    classic = mot.forward_states(vs, acs, classic_mask)
    triple_mask = build_hog_mot_attention_mask(
        vs.tokens.shape[1], hs.tokens.shape[1], acs.tokens.shape[1], vs.tokens_per_frame
    )
    triple = mot.forward_hog_states(vs, hs, acs, triple_mask)

    np.testing.assert_allclose(np.asarray(triple[0]), np.asarray(classic[0]), rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(np.asarray(triple[2]), np.asarray(classic[1]), rtol=2e-5, atol=2e-5)


def test_hog_and_rgb_towers_can_start_equal_but_are_independent():
    rgb = build_video()
    hog = build_video()
    flat = {
        ".".join(str(p) for p in path): np.asarray(value[...])
        for path, value in nnx.to_flat_state(nnx.state(rgb, nnx.Param))
    }
    load_flat_params(hog, flat)
    rgb_leaf = jax.tree.leaves(nnx.state(rgb, nnx.Param))[0]
    hog_leaf = jax.tree.leaves(nnx.state(hog, nnx.Param))[0]
    np.testing.assert_array_equal(np.asarray(rgb_leaf), np.asarray(hog_leaf))
    assert rgb_leaf is not hog_leaf
