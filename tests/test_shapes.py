"""Forward-shape gate, at the REAL released geometry.

Every model here is built and run under `nnx.eval_shape`, so the 6.02 B FastWAM is
traced without allocating a single array and the file still runs on a laptop. The
numerical behaviour of the same code at small widths lives in test_dit_parity.py (M1)
and test_mot_parity.py (M2); this file only pins shapes.

Every expected number comes from spec §4.5 via the config's `DerivedShapes`, never from
a literal in a model module. The one remaining skip is M3's.
"""

import numpy as np
import pytest

from fastwam_tpu import load_config

TASK = "training/full.example.yaml"


def test_stub_modules_import():
    """Every stub must at least be importable — catches syntax/typo rot early."""
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import fastwam_tpu.data.loader  # noqa: F401
    import fastwam_tpu.inference.policy  # noqa: F401
    import fastwam_tpu.models.action_dit  # noqa: F401
    import fastwam_tpu.models.fastwam  # noqa: F401
    import fastwam_tpu.models.mot  # noqa: F401
    import fastwam_tpu.models.wan.attention  # noqa: F401
    import fastwam_tpu.models.wan.embeddings  # noqa: F401
    import fastwam_tpu.models.wan.rope  # noqa: F401
    import fastwam_tpu.models.wan.transformer  # noqa: F401
    import fastwam_tpu.models.wan.vae  # noqa: F401
    import fastwam_tpu.schedulers.flow_match  # noqa: F401
    import fastwam_tpu.training.checkpoint  # noqa: F401
    import fastwam_tpu.training.remat  # noqa: F401
    import fastwam_tpu.training.sharding  # noqa: F401


def test_shape_contract_is_stated():
    """The numbers the M1/M2 tests assert come from the config, not from literals."""
    s = load_config(TASK).shapes
    assert (s.latent_shape, s.video_tokens, s.action_tokens, s.joint_seq) == (
        (48, 3, 24, 20),
        360,
        32,
        392,
    )


def test_rope_tables():
    """precompute_freqs_3d(128, 3, 12, 10) -> cos/sin [360, 64] fp32.
    Axis split 44/42/42 -> 22/21/21 complex pairs. Interleaved (even, odd) pairing.

    The interleaved pairing and the elementwise match against torch's float64 tables
    live in test_dit_parity.py; this is the shape/convention contract.
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from fastwam_tpu.models.wan.rope import apply_rope, precompute_freqs_3d, rope_axis_dims

    shapes = load_config(TASK).shapes
    grid = (shapes.latent_frames, shapes.latent_height // 2, shapes.latent_width // 2)
    assert grid == (3, 12, 10)

    assert rope_axis_dims(128) == (44, 42, 42)
    cos, sin = precompute_freqs_3d(128, *grid)
    assert cos.shape == sin.shape == (shapes.video_tokens, 64) == (360, 64)
    assert cos.dtype == sin.dtype == jnp.float32
    # 22/21/21 complex pairs: position 0 of every axis has cos 1 / sin 0, so token 0
    # (f=h=w=0) must be the identity rotation.
    assert np.allclose(np.asarray(cos[0]), 1.0)
    assert np.allclose(np.asarray(sin[0]), 0.0)

    x = jnp.ones((2, 360, 24, 128))
    assert apply_rope(x, cos, sin).shape == x.shape
    assert np.allclose(np.asarray(apply_rope(x, cos, sin)[:, 0]), 1.0)


def test_video_dit_forward_shapes():
    """latents [2,48,3,24,20], timestep [2], context [2,129,4096], mask [2,129]
    -> tokens [2,360,3072], t [2,360,3072], t_mod [2,360,6,3072]
    -> post_dit -> [2,48,3,24,20].

    Run under nnx.eval_shape: the real 5 B geometry, no arrays allocated. The
    numerical behaviour of this same code at a small width is test_dit_parity.py.
    """
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax
    from flax import nnx

    from fastwam_tpu.models.wan.transformer import WanVideoDiT

    config = load_config(TASK)
    shapes = config.shapes
    batch = 2

    def forward():
        model = WanVideoDiT.from_config(config.model.video_dit_config, rngs=nnx.Rngs(0))
        latents = jax.numpy.zeros((batch, *shapes.latent_shape))
        timestep = jax.numpy.zeros((batch,))
        context = jax.numpy.zeros((batch, shapes.context_tokens, config.model.video_dit_config.text_dim))
        context_mask = jax.numpy.ones((batch, shapes.context_tokens), bool)
        state = model.pre_dit(latents, timestep, context, context_mask)
        return state.tokens, state.t, state.t_mod, state.context_mask, model(
            latents, timestep, context, context_mask
        )

    tokens, t, t_mod, context_mask, out = nnx.eval_shape(forward)
    hidden = config.model.video_dit_config.hidden_dim
    assert tokens.shape == (batch, shapes.video_tokens, hidden) == (2, 360, 3072)
    assert t.shape == (batch, shapes.video_tokens, hidden)
    assert t_mod.shape == (batch, shapes.video_tokens, 6, hidden) == (2, 360, 6, 3072)
    assert context_mask.shape == (batch, shapes.video_tokens, shapes.context_tokens) == (2, 360, 129)
    assert out.shape == (batch, *shapes.latent_shape) == (2, 48, 3, 24, 20)


def test_video_self_attention_mask():
    """[360,360] bool; rows 0:120 are False on columns 120:360, True elsewhere."""
    pytest.importorskip("jax")

    from fastwam_tpu.models.wan.transformer import build_video_self_attention_mask

    shapes = load_config(TASK).shapes
    mask = np.asarray(build_video_self_attention_mask(shapes.tokens_per_frame, shapes.latent_frames))
    assert mask.shape == (shapes.video_tokens, shapes.video_tokens) == (360, 360)
    assert mask.dtype == np.bool_
    first = shapes.tokens_per_frame
    assert mask[:first, :first].all()
    assert not mask[:first, first:].any()
    assert mask[first:, :].all()


def test_vae_shapes():
    """video [1,3,9,384,320] -> latents [1,48,3,24,20] -> decode -> [1,3,9,384,320].

    Run on a narrowed VAE (base_dim 8 instead of 160/256) so it fits in a CPU test:
    the channel widths do not enter the shape contract, only z_dim and the 16x/4x
    compression factors do, and those are the released values.
    """
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax
    from flax import nnx

    from fastwam_tpu.models.wan.vae import Wan22VaeConfig, WanVAE38

    shapes = load_config(TASK).shapes
    vae = WanVAE38(Wan22VaeConfig(base_dim=8, decoder_base_dim=8), nnx.Rngs(0))
    vae.check_geometry(shapes)

    video = jax.random.uniform(
        jax.random.key(0), (1, 3, shapes.video_frames, 384, 320), minval=-1.0, maxval=1.0
    )
    latents = vae.encode(video)
    assert latents.shape == (1, *shapes.latent_shape) == (1, 48, 3, 24, 20)
    # M3 prefills a single observation; that is one latent frame of the same clip.
    assert vae.encode_frame(video[:, :, :1]).shape == (1, 48, 1, 24, 20)
    assert vae.decode(latents).shape == video.shape


def test_action_dit_forward_shapes():
    """action [2,32,14], timestep [2] -> tokens [2,32,1024], t_mod [2,6,1024]
    -> q/k/v [2,32,24,128] (attention width 3072, not 1024) -> head -> [2,32,14].

    Under nnx.eval_shape at the real 1.02 B geometry. The numerical behaviour of the
    same code at a small width is test_mot_parity.py.
    """
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import jax
    from flax import nnx

    from fastwam_tpu.models.action_dit import ActionDiT

    config = load_config(TASK)
    action_config = config.model.action_dit_config
    shapes = config.shapes
    batch = 2

    def forward():
        model = ActionDiT.from_config(action_config, rngs=nnx.Rngs(0))
        action = jax.numpy.zeros((batch, shapes.action_tokens, action_config.action_dim))
        timestep = jax.numpy.zeros((batch,))
        context = jax.numpy.zeros((batch, shapes.context_tokens, action_config.text_dim))
        context_mask = jax.numpy.ones((batch, shapes.context_tokens), bool)
        state = model.pre_dit(action, timestep, context, context_mask)
        qkv = model.blocks[0].pre_attention(state.tokens, state.t_mod, state.cos, state.sin)[0]
        return state, qkv, model(action, timestep, context, context_mask)

    state, (query, key, value), out = nnx.eval_shape(forward)
    hidden = action_config.hidden_dim
    assert state.tokens.shape == (batch, shapes.action_tokens, hidden) == (2, 32, 1024)
    assert state.t.shape == (batch, hidden)
    # Rank 3, NOT the video tower's rank 4: one modulation shared by all action tokens.
    assert state.t_mod.shape == (batch, 6, hidden) == (2, 6, 1024)
    assert state.context.shape == (batch, shapes.context_tokens, hidden) == (2, 129, 1024)
    assert state.context_mask.shape == (batch, shapes.action_tokens, shapes.context_tokens)
    assert state.cos.shape == state.sin.shape == (shapes.action_tokens, 64)
    # The attention is 3072 wide even though the tower is 1024 wide (spec §2.3).
    for tensor in (query, key, value):
        assert tensor.shape == (batch, shapes.action_tokens, 24, 128) == (2, 32, 24, 128)
    assert out.shape == (batch, shapes.action_tokens, action_config.action_dim) == (2, 32, 14)


def test_mot_attention_mask():
    """[392,392] bool: action rows (360:392) True only on columns 0:120 and 360:392;
    video rows never see action columns; video block keeps first_frame_causal."""
    pytest.importorskip("jax")

    from fastwam_tpu.models.mot import build_mot_attention_mask

    shapes = load_config(TASK).shapes
    mask = np.asarray(
        build_mot_attention_mask(shapes.video_tokens, shapes.action_tokens, shapes.tokens_per_frame)
    )
    video, per_frame = shapes.video_tokens, shapes.tokens_per_frame
    assert mask.shape == (shapes.joint_seq, shapes.joint_seq) == (392, 392)
    assert mask.dtype == np.bool_
    assert mask[:per_frame, :per_frame].all()
    assert not mask[:per_frame, per_frame:video].any()
    assert mask[per_frame:video, :video].all()
    assert not mask[:video, video:].any()
    assert mask[video:, :per_frame].all()
    assert not mask[video:, per_frame:video].any()
    assert mask[video:, video:].all()


def test_mot_forward_shapes():
    """video [2,360,3072] + action [2,32,1024] through a mixed layer at FULL width,
    then the two post_dits -> [2,48,3,24,20] and [2,32,14].

    Full released widths (3072 / 1024 / 24 x 128), but `num_layers` reduced to 1. The
    layer is shape-idempotent — its output shape is its input shape — so one layer pins
    every geometry a real-width test can pin: the 1024 -> 3072 -> 1024 attention, the
    concat/split at Sv = 360, and both heads. Tracing all 30 costs ~15 minutes of wall
    clock under `nnx.eval_shape` and adds nothing.

    What the other two thirds are covered by:
      * all 30 layers, unrolled AND scanned, numerically — test_mot_parity.py;
      * the full 30-layer 6.02 B parameter inventory, key by key —
        test_mot_parity.py::test_official_inventory_has_the_expected_shape_and_counts.
    """
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import dataclasses

    import jax
    from flax import nnx

    from fastwam_tpu.models.mot import MoT, build_mot_attention_mask

    config = load_config(TASK)
    shapes = config.shapes
    batch = 2
    assert config.model.video_dit_config.num_layers == 30
    one_layer = dataclasses.replace(
        config.model,
        video_dit_config=dataclasses.replace(config.model.video_dit_config, num_layers=1),
        action_dit_config=dataclasses.replace(config.model.action_dit_config, num_layers=1),
    )

    def forward():
        mot = MoT.from_config(one_layer, rngs=nnx.Rngs(0))
        latents = jax.numpy.zeros((batch, *shapes.latent_shape))
        action = jax.numpy.zeros((batch, shapes.action_tokens, one_layer.action_dit_config.action_dim))
        context = jax.numpy.zeros((batch, shapes.context_tokens, one_layer.video_dit_config.text_dim))
        context_mask = jax.numpy.ones((batch, shapes.context_tokens), bool)
        timestep = jax.numpy.zeros((batch,))
        video_state = mot.video.pre_dit(latents, timestep, context, context_mask)
        action_state = mot.action.pre_dit(action, timestep, context, context_mask)
        mask = build_mot_attention_mask(
            shapes.video_tokens, shapes.action_tokens, shapes.tokens_per_frame
        )
        video_tokens, action_tokens = mot(video_state, action_state, mask)
        return (
            video_state.tokens,
            action_state.tokens,
            video_tokens,
            action_tokens,
            mot.video.post_dit(video_tokens, video_state.t, video_state.grid),
            mot.action.post_dit(action_tokens),
        )

    video_in, action_in, video_tokens, action_tokens, pred_video, pred_action = nnx.eval_shape(forward)
    assert video_in.shape == video_tokens.shape == (batch, shapes.video_tokens, 3072) == (2, 360, 3072)
    assert action_in.shape == action_tokens.shape == (batch, shapes.action_tokens, 1024) == (2, 32, 1024)
    assert pred_video.shape == (batch, *shapes.latent_shape) == (2, 48, 3, 24, 20)
    assert pred_action.shape == (batch, shapes.action_tokens, 14) == (2, 32, 14)


def test_training_loss_shapes():
    """FastWAM.training_loss(batch) -> scalar loss + {loss_video, loss_action};
    video loss drops latent frame 0, so it reduces over [B, 2]; action over [B, 32].

    The real-geometry part is the LOSS PLUMBING — `build_inputs` producing a 129-token
    context, the frame-0 drop leaving T_l - 1 = 2 steps, and image_is_pad folding 4:1
    onto those 2 steps. Those are what the released 9-frame / 3-latent-frame geometry
    can break, and they are checked here directly.

    The composed `training_loss` is NOT traced at 6 B here: it is one 30-layer MoT pass,
    which under `nnx.eval_shape` costs ~20 minutes of wall clock and would make this
    file unusable as a gate. The contract it would prove — scalar loss, scalar metrics,
    [B] timesteps, jittable and differentiable — is asserted at small width in
    test_mot_parity.py (`test_training_loss_parity`,
    `test_training_loss_runs_under_jit_and_grad`), and the 6 B forward it would trace is
    already covered by `test_mot_forward_shapes` above.
    """
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import dataclasses

    import jax.numpy as jnp
    from flax import nnx

    from fastwam_tpu.models.fastwam import FastWAM
    from fastwam_tpu.schedulers import flow_match

    config = load_config(TASK)
    model_config = config.model
    shapes = config.shapes
    batch = 2

    # `build_inputs` only ever touches the proprio encoder, so the two towers are
    # narrowed to one layer each: constructing 60 blocks at d=3072/1024 abstractly is
    # minutes of wall clock and proves nothing this test asserts.
    one_layer = dataclasses.replace(
        config,
        model=dataclasses.replace(
            model_config,
            video_dit_config=dataclasses.replace(model_config.video_dit_config, num_layers=1),
            action_dit_config=dataclasses.replace(model_config.action_dit_config, num_layers=1),
        ),
    )
    model = nnx.eval_shape(lambda: FastWAM.from_config(one_layer, rngs=nnx.Rngs(0)))
    # The proprio token lives in TEXT space (4096), not in either tower's width.
    assert model.proprio_encoder.kernel.shape == (model_config.proprio_dim, 4096) == (14, 4096)

    def assemble():
        built = FastWAM.from_config(one_layer, rngs=nnx.Rngs(0))
        data = {
            "latents": jnp.zeros((batch, *shapes.latent_shape)),
            "action": jnp.zeros(
                (batch, shapes.action_tokens, model_config.action_dit_config.action_dim)
            ),
            "proprio": jnp.zeros((batch, shapes.action_tokens, model_config.proprio_dim)),
            "context": jnp.zeros(
                (batch, shapes.context_tokens - 1, model_config.video_dit_config.text_dim)
            ),
            "context_mask": jnp.ones((batch, shapes.context_tokens - 1), bool),
            "image_is_pad": jnp.zeros((batch, shapes.video_frames), bool),
            "action_is_pad": jnp.zeros((batch, shapes.action_tokens), bool),
        }
        return built.build_inputs(data)

    inputs = nnx.eval_shape(assemble)
    # 128 text tokens + 1 proprio token, in TEXT space (4096), not tower space.
    assert inputs.context.shape == (batch, shapes.context_tokens, 4096) == (2, 129, 4096)
    assert inputs.context_mask.shape == (batch, shapes.context_tokens) == (2, 129)
    assert inputs.latents.shape == (batch, *shapes.latent_shape) == (2, 48, 3, 24, 20)

    # The loss reductions, at the released geometry, with real arrays (they are tiny).
    real = FastWAM.__new__(FastWAM)  # the two reducers do not touch any parameter
    tail_frames = shapes.latent_frames - 1
    pred = jnp.zeros((batch, shapes.latent_channels, tail_frames, shapes.latent_height,
                      shapes.latent_width))
    target = jnp.ones_like(pred)
    is_pad = jnp.zeros((batch, shapes.video_frames), bool).at[0, 5:].set(True)
    video_per_sample = FastWAM._video_loss_per_sample(real, pred, target, is_pad)
    assert video_per_sample.shape == (batch,)
    # 9 frames -> tail 8 -> folded 4:1 onto exactly T_l - 1 = 2 latent steps.
    assert (shapes.video_frames - 1) // 4 == tail_frames == 2

    action_per_sample = FastWAM._action_loss_per_sample(
        jnp.zeros((batch, shapes.action_tokens, 14)),
        jnp.ones((batch, shapes.action_tokens, 14)),
        jnp.zeros((batch, shapes.action_tokens), bool),
    )
    assert action_per_sample.shape == (batch,)

    timestep = jnp.array([137.0, 913.0])
    weight = flow_match.training_weight(timestep, model_config.video_scheduler.num_train_timesteps)
    assert weight.shape == (batch,)
    assert jnp.mean(video_per_sample * weight).shape == ()


@pytest.mark.skip(reason="M3: KV cache prefill equals the joint path within 1e-2")
def test_prefill_equivalence():
    """prefill: 30 layers x {k, v} [1,120,3072]; infer_action -> [32,14]."""
