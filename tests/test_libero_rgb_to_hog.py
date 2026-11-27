"""The LIBERO RGB-to-HOG ablation changes visual content, not the task contract."""
import dataclasses

import numpy as np

from fastwam_tpu.config import load_config

RECIPE = 'training/libero-rgb-to-hog.example.yaml'


def test_matched_libero_rgb_to_hog_recipe():
    base = load_config('training/libero-baseline.example.yaml')
    new = load_config(RECIPE)
    assert new.data == base.data
    assert new.model == dataclasses.replace(base.model, visual_input_mode='rgb_to_hog')
    for name in ['flow', 'optimizer', 'learning_rate', 'weight_decay', 'num_epochs', 'seed', 'max_grad_norm']:
        assert getattr(new, name) == getattr(base, name)
    assert new.tpu.global_batch_size == base.tpu.global_batch_size == 256
    assert new.tpu.micro_batch_size == 256 and new.tpu.grad_accum == 1
    assert new.tpu.steps == new.tpu.decay_steps == 13200
    assert new.tpu.compute_dtype == base.tpu.compute_dtype == 'bfloat16'
    assert new.tpu.param_dtype == base.tpu.param_dtype == 'float32'
    assert new.tpu.ckpt == base.tpu.ckpt


def test_libero_hog_geometry_and_current_future_consistency():
    import jax.numpy as jnp
    from fastwam_tpu.data.libero import compose_cameras
    from fastwam_tpu.models.hog import hog_image
    rng = np.random.default_rng(42)
    canvas = compose_cameras(rng.integers(0, 256, (9, 24, 24, 3), np.uint8),
                             rng.integers(0, 256, (9, 24, 24, 3), np.uint8))
    hog = hog_image(jnp.asarray(canvas[None]))
    current = hog_image(jnp.asarray(canvas[None, :, :1]))
    assert hog.shape == (1, 3, 9, 224, 448)
    assert np.isfinite(hog).all() and float(hog.min()) >= -1 and float(hog.max()) <= 1
    np.testing.assert_allclose(hog[:, :, :1], current, atol=1e-6, rtol=0)


def test_two_tower_libero_rgb_to_hog_optimizer_step():
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from test_train_step import small_model_config, make_micro
    from fastwam_tpu.models.fastwam import FastWAM
    from fastwam_tpu.training.train import make_fused_step
    cfg = small_model_config()
    cfg = dataclasses.replace(cfg, visual_input_mode='rgb_to_hog', training_towers='rgb_action',
        proprio_dim=8, video_dit_config=dataclasses.replace(cfg.video_dit_config, action_dim=7),
        action_dit_config=dataclasses.replace(cfg.action_dit_config, action_dim=7),
        loss=dataclasses.replace(cfg.loss, lambda_video=1.0, lambda_hog=0.0, lambda_action=1.0))
    model = FastWAM(cfg, rngs=nnx.Rngs(42), include_hog=False, scan_layers=True, remat_policy='full')
    graph, params = nnx.split(model)
    assert 'hog' not in nnx.to_pure_dict(params)['mot']
    batch = make_micro(17, 1)
    batch.pop('hog_latents')
    batch['action'] = jnp.ones((1, 8, 7)) * .1
    batch['proprio'] = jnp.ones((1, 8, 8)) * .1
    tx = optax.adamw(1e-4)
    step = make_fused_step(graph, tx, accum=1, zero_n=None, copy_lengths=(8,) * 8)
    _, _, metrics = step(params, tx.init(params), [jax.random.key(42)], [batch])
    assert np.isfinite(float(metrics['loss'])) and float(metrics['grad_norm']) > 0
    assert float(metrics['loss_video']) > 0 and float(metrics['loss_action']) > 0
    assert float(metrics['loss_hog']) == 0
