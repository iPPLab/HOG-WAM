"""Deployment transforms must preserve the native LIBERO training contract."""
import importlib.util
from pathlib import Path

import numpy as np

from fastwam_tpu.data.libero import compose_cameras, normalize, denormalize

spec = importlib.util.spec_from_file_location(
    'libero_plus_policy_server', Path(__file__).parents[1] / 'tools/libero_plus_policy_server.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_minmax_stats_adapter_exactly_matches_training():
    stats = {'min': [-2, 4, 1], 'max': [2, 4, 3]}
    adapter = module.MinMaxDimension(stats)
    raw = np.asarray([0.5, 4, 1.2], np.float32)
    assert adapter.dim == 3
    np.testing.assert_array_equal(adapter.forward(raw), normalize(raw, stats))
    np.testing.assert_array_equal(adapter.backward(adapter.forward(raw)), denormalize(normalize(raw, stats), stats))


def test_deployment_canvas_uses_same_float_resize_and_orientation():
    rng = np.random.default_rng(5)
    agent = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, agent.shape, dtype=np.uint8)
    policy = object.__new__(module.LiberoPolicy)
    image = np.asarray(policy.canvas_to_tensor((agent, wrist)))
    assert image.shape == (1, 3, 224, 448)
    np.testing.assert_array_equal(image[0], compose_cameras(agent[None], wrist[None])[:, 0])


def test_rgb_to_hog_deployment_replaces_pixels_before_vae():
    """Exercise the real infer_action path; HOG must not be merely a config label."""
    import jax.numpy as jnp
    from types import SimpleNamespace
    from fastwam_tpu.models.hog import hog_image
    rng = np.random.default_rng(6)
    agent = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, agent.shape, dtype=np.uint8)
    policy = object.__new__(module.LiberoPolicy)
    policy.visual_input_mode = 'rgb_to_hog'
    policy.context_len, policy.text_dim = 128, 4096
    policy.proprio_dim, policy.action_dim = 8, 7
    policy.stats = SimpleNamespace(proprio=module.MinMaxDimension({'min': [-1.] * 8, 'max': [1.] * 8}),
                                  action=module.MinMaxDimension({'min': [-1.] * 7, 'max': [1.] * 7}))
    policy.noise_mode, policy.replans = 'fixed', 0
    policy.vae = policy._model_state = policy.noise = None
    captured = []
    def encode(vae, pixels):
        captured.append(np.asarray(pixels))
        return jnp.zeros((1, 48, 1, 14, 28))
    policy._encode = encode
    policy._denoise = lambda *args: jnp.zeros((1, 32, 7))
    action = policy.infer_action((agent, wrist), np.zeros(8, np.float32),
        np.zeros((128, 4096), np.float32), np.ones(128, bool))
    canvas = compose_cameras(agent[None], wrist[None])
    expected = np.asarray(hog_image(jnp.asarray(canvas[None]))[:, :, 0])
    np.testing.assert_array_equal(captured[0], expected)
    assert not np.allclose(captured[0], canvas[:, 0][None])
    assert action.shape == (32, 7) and np.isfinite(action).all()
    assert policy.replans == 1
