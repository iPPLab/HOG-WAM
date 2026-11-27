import dataclasses
import json
import hashlib

import numpy as np
import pytest

from fastwam_tpu.config import load_config
from fastwam_tpu.data.libero import retained_indices, normalize, denormalize, compose_cameras


def test_native_contract_and_matched_recipes():
    baseline = load_config('training/libero-baseline.example.yaml')
    hog = load_config('training/libero-hog.example.yaml')
    assert baseline.data.format == hog.data.format == 'libero_hdf5'
    assert baseline.data.train.video_size == (224, 448)
    assert baseline.model.proprio_dim == 8
    assert baseline.model.action_dit_config.action_dim == 7
    assert baseline.shapes.latent_shape == (48, 3, 14, 28)
    assert baseline.flow.tau_copies == hog.flow.tau_copies == 8
    assert baseline.model.training_towers == 'rgb_action'
    assert hog.model.training_towers == 'rgb_hog_action'
    assert baseline.data == hog.data
    assert baseline.learning_rate == hog.learning_rate
    assert baseline.tpu.global_batch_size == hog.tpu.global_batch_size == 256
    assert baseline.model.loss.lambda_hog == 0
    assert hog.model.loss.lambda_hog == 1
    # Existing RoboTwin recipes must keep their geometry and third tower.
    old = load_config('training/full.example.yaml')
    assert old.data.format == 'robotwin_twp'
    assert old.model.training_towers == 'rgb_hog_action'
    assert old.shapes.latent_shape == (48, 3, 24, 20)


def test_noops_retain_gripper_only_changes():
    actions = np.zeros((5, 7), np.float32)
    actions[:, 6] = [-1, -1, 1, 1, 1]
    actions[4, 0] = .1
    np.testing.assert_array_equal(retained_indices(actions), [0, 2, 4])


def test_native_normalization_round_trip_and_constant_dimensions():
    stats = {'min': [-1, 2, 0], 'max': [1, 6, 0]}
    raw = np.asarray([[-1, 2, 0], [0, 4, 0], [1, 6, 0]], np.float32)
    normed = normalize(raw, stats)
    np.testing.assert_allclose(normed, [[-1, -1, 0], [0, 0, 0], [1, 1, 0]])
    np.testing.assert_allclose(denormalize(normed, stats), raw)


def test_two_camera_canvas_preserves_order_and_channels():
    agent = np.zeros((2, 8, 8, 3), np.uint8)
    wrist = np.zeros_like(agent)
    agent[..., 0] = 255
    wrist[..., 2] = 255
    image = compose_cameras(agent, wrist)
    assert image.shape == (3, 2, 224, 448)
    np.testing.assert_allclose(image[0, :, :, :224], 1, atol=1e-6)
    np.testing.assert_allclose(image[2, :, :, :224], -1, atol=1e-6)
    np.testing.assert_allclose(image[2, :, :, 224:], 1, atol=1e-6)


def test_hdf5_loader_alignment_padding_and_determinism(tmp_path):
    h5py = pytest.importorskip('h5py')
    from fastwam_tpu.data.libero import LiberoHdf5Dataset
    from fastwam_tpu.inference.text import TextEmbeddingCache, build_prompt

    # Synthetic observations are a unit-test fixture, never production embeddings.
    task_file = tmp_path / 'fixture.hdf5'
    with h5py.File(task_file, 'w') as f:
        obs = f.create_group('data/demo_0/obs')
        agent = np.zeros((5, 8, 8, 3), np.uint8)
        wrist = np.zeros_like(agent)
        agent[..., 0] = np.arange(5, dtype=np.uint8)[:, None, None] * 40
        wrist[..., 2] = 255
        obs.create_dataset('agentview_rgb', data=agent)
        obs.create_dataset('eye_in_hand_rgb', data=wrist)
    manifest = {'source': {'unit_test': True}, 'episodes': [{
        'task': 'fixture', 'file': task_file.name, 'demo': 'demo_0',
        'instruction': 'unit test', 'length': 2, 'offset': 0,
    }]}
    blob = json.dumps(manifest).encode()
    (tmp_path / 'index.json').write_bytes(blob)
    actions = np.arange(14, dtype=np.float32).reshape(2, 7) / 14
    states = np.arange(16, dtype=np.float32).reshape(2, 8) / 16
    np.savez(tmp_path / 'numeric.npz', actions=actions, states=states,
             raw_indices=np.asarray([1, 4]))
    stats = {'index_sha256': hashlib.sha256(blob).hexdigest(),
             'action': {'min': [-1] * 7, 'max': [1] * 7},
             'state': {'min': [-1] * 8, 'max': [1] * 8}}
    text_dir = tmp_path / 'text'
    TextEmbeddingCache(str(text_dir)).put(
        build_prompt('unit test'), np.ones((1, 4096), np.float32), 1)
    dataset = LiberoHdf5Dataset(tmp_path, stats_blob=json.dumps(stats).encode(),
                               stats_source='unit-test', text_dir=text_dir)
    try:
        sample = dataset._sample((0, 0))
        np.testing.assert_allclose(sample['action'][:2], actions, atol=1e-7)
        np.testing.assert_allclose(sample['proprio'][:2], states, atol=1e-7)
        # Retained observation one is raw row 1, not raw row 0 or future row 4.
        np.testing.assert_allclose(sample['video'][0, 0, :, :224], 2 * 40 / 255 - 1,
                                   atol=1e-6)
        np.testing.assert_allclose(sample['video'][0, 1:, :, :224], 2 * 160 / 255 - 1,
                                   atol=1e-6)
        np.testing.assert_array_equal(sample['image_is_pad'], [False] + [True] * 8)
        np.testing.assert_array_equal(sample['action_is_pad'], [False] * 2 + [True] * 30)
        for k, v in dataset.sample_batch(np.random.default_rng(17), 3).items():
            np.testing.assert_array_equal(
                v, dataset.sample_batch(np.random.default_rng(17), 3)[k])
        # Read-ahead must preserve the per-step generator stream and output order.
        import concurrent.futures
        seeds = [19, 23, 29]
        expected = [dataset.sample_batch(np.random.default_rng(seed), 2) for seed in seeds]
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(dataset.sample_batch, np.random.default_rng(seeds[0]), 2)
            for index, seed in enumerate(seeds):
                actual = future.result()
                if index + 1 < len(seeds):
                    future = executor.submit(dataset.sample_batch, np.random.default_rng(seeds[index + 1]), 2)
                for key in actual:
                    np.testing.assert_array_equal(actual[key], expected[index][key])
    finally:
        dataset.close()
    stats['index_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='fingerprints'):
        LiberoHdf5Dataset(tmp_path, stats_blob=json.dumps(stats).encode(),
                          stats_source='unit-test', text_dir=text_dir)


def test_training_step_runs_for_both_tower_choices():
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from test_train_step import small_model_config, make_micro
    from fastwam_tpu.models.fastwam import FastWAM
    from fastwam_tpu.training.train import make_fused_step
    for include_hog in [False, True]:
        cfg = small_model_config()
        cfg = dataclasses.replace(cfg, proprio_dim=8,
            video_dit_config=dataclasses.replace(cfg.video_dit_config, action_dim=7),
            action_dit_config=dataclasses.replace(cfg.action_dit_config, action_dim=7),
            loss=dataclasses.replace(cfg.loss, lambda_hog=1.0 if include_hog else 0.0))
        model = FastWAM(cfg, rngs=nnx.Rngs(7), include_hog=include_hog, scan_layers=True, remat_policy='full')
        graph, params = nnx.split(model)
        tx = optax.adamw(1e-4)
        micro = make_micro(11, 1)
        micro['action'] = jnp.ones((1, 8, 7)) * .1
        micro['proprio'] = jnp.ones((1, 8, 8)) * .1
        if not include_hog:
            micro.pop('hog_latents')
        step = make_fused_step(graph, tx, accum=1, zero_n=None, copy_lengths=(8,) * 8)
        _, _, metrics = step(params, tx.init(params), [jax.random.key(17)], [micro])
        assert np.isfinite(float(metrics['loss']))
        assert float(metrics['grad_norm']) > 0
        assert (float(metrics['loss_hog']) > 0) == include_hog
