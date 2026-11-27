"""M1 numerical gate: the video tower against a float64 transcription of FastWAM.

The golden is `tests/_golden_ref/fastwam_torch_ref.py` — see its docstring for why it
is a numpy transcription of FastWAM's torch source rather than a vendored copy of
maxdiffusion. Everything here is a small config (2 layers, d=128, 4 heads) so the
whole file runs on CPU in seconds.

Reading the numbers: the golden runs in float64, the model in float32, so the
residual is an fp32 rounding budget. `MAX_REL` is the bar the gate is set at;
`report()` prints the measured value under `pytest -s` so drift is visible.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from _golden_ref import fastwam_torch_ref as ref
from flax import nnx

from fastwam_tpu.models.wan import rope
from fastwam_tpu.models.wan.attention import dot_product_attention
from fastwam_tpu.models.wan.embeddings import sinusoidal_embedding_1d
from fastwam_tpu.models.wan.transformer import (
    WanVideoDiT,
    build_token_timesteps,
    build_video_self_attention_mask,
    unpatchify,
)
from tools.convert_weights import (
    ConversionError,
    compare_param_shapes,
    stack_blocks_for_scan,
    video_dit_param_shapes,
)

# Small parity config. head_dim 32 -> RoPE axis split (12, 10, 10) -> 6+5+5 = 16 pairs.
DIM = 128
HEADS = 4
HEAD_DIM = 32
LAYERS = 2
FREQ_DIM = 32
TEXT_DIM = 40
IN_CH = 8
PATCH = (1, 2, 2)
LATENT_SHAPE = (2, IN_CH, 3, 8, 6)  # -> grid (3, 4, 3), 12 tokens/frame, Sv = 36
CONTEXT_LEN = 9

MAX_REL = 2e-5
FULL_CONFIG = "training/full.example.yaml"


def report(name: str, value: float) -> float:
    print(f"[parity] {name}: {value:.3e}")
    return value


def rel_error(got, want) -> float:
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    return float(np.max(np.abs(got - want)) / max(np.max(np.abs(want)), 1e-30))


def flat_params(module: nnx.Module) -> dict[str, np.ndarray]:
    """{dotted path: np.ndarray} — the same keys the Orbax checkpoint uses."""
    state = nnx.state(module, nnx.Param)
    return {
        ".".join(str(p) for p in path): np.asarray(value[...])
        for path, value in nnx.to_flat_state(state)
    }


def build_model(**overrides) -> WanVideoDiT:
    kwargs = dict(
        in_channels=IN_CH,
        out_channels=IN_CH,
        hidden_dim=DIM,
        ffn_dim=2 * DIM,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        text_dim=TEXT_DIM,
        patch_size=PATCH,
        eps=1e-6,
    )
    kwargs.update(overrides)
    return WanVideoDiT(rngs=nnx.Rngs(0), **kwargs)


def ref_config(mask_mode: str) -> ref.RefConfig:
    return ref.RefConfig(
        hidden_dim=DIM,
        num_heads=HEADS,
        attn_head_dim=HEAD_DIM,
        num_layers=LAYERS,
        freq_dim=FREQ_DIM,
        out_dim=IN_CH,
        patch_size=PATCH,
        eps=1e-6,
        video_attention_mask_mode=mask_mode,
    )


def sample_inputs(seed: int = 1):
    keys = jax.random.split(jax.random.key(seed), 3)
    latents = jax.random.normal(keys[0], LATENT_SHAPE)
    context = jax.random.normal(keys[1], (LATENT_SHAPE[0], CONTEXT_LEN, TEXT_DIM))
    timestep = jnp.array([137.0, 913.0])
    return latents, timestep, context


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def test_sinusoidal_embedding_matches_torch_formula():
    """cos first, then sin — and the fp64 frequency table, not maxdiffusion's exp form."""
    positions = jnp.array([0.0, 1.0, 137.0, 499.5, 999.0])
    got = np.asarray(sinusoidal_embedding_1d(256, positions))
    want = ref.sinusoidal_embedding_1d(256, np.asarray(positions))
    assert got.shape == want.shape == (5, 256)
    # If cos/sin were swapped the first half at position 0 would be 0, not 1.
    assert np.allclose(got[0, :128], 1.0)
    assert np.allclose(got[0, 128:], 0.0)
    assert report("sinusoidal_embedding_1d", rel_error(got, want)) < 1e-6


def test_sinusoidal_embedding_beats_naive_fp32():
    """Justify the Dekker two-product in embeddings.py, and guard it against XLA.

    The naive fp32 form is off by ~3e-5 at t≈1000; if the correction ever stops
    working (an XLA reassociation, a refactor) this test goes red instead of the
    error quietly reappearing inside every adaLN modulation.
    """
    positions = jnp.array([999.0, 913.0, 499.5])
    want = ref.sinusoidal_embedding_1d(256, np.asarray(positions))

    inv = np.power(10000.0, -np.arange(128, dtype=np.float64) / 128).astype(np.float32)
    angles = np.asarray(positions, np.float32)[:, None] * inv
    naive = np.concatenate([np.cos(angles), np.sin(angles)], axis=-1)
    naive_err = report("sinusoidal naive fp32", rel_error(naive, want))
    ours_err = rel_error(sinusoidal_embedding_1d(256, positions), want)
    assert naive_err > 1e-5, "the naive path is no longer inaccurate; the correction may be dead code"
    assert ours_err < naive_err / 100


def test_rope_axis_split_matches_fastwam_at_128():
    """maxdiffusion's 2*(d//6) form must agree with FastWAM's d//3 form at d=128."""
    assert rope.rope_axis_dims(128) == (128 - 2 * (128 // 3), 128 // 3, 128 // 3) == (44, 42, 42)
    for head_dim in (32, 64, 96, 128):
        dims = rope.rope_axis_dims(head_dim)
        assert sum(dims) == head_dim
        assert all(d % 2 == 0 for d in dims)


def test_rope_tables_match_torch_formula():
    """cos/sin elementwise against torch.polar's real/imag, in float64."""
    grid = (3, 12, 10)
    cos, sin = rope.precompute_freqs_3d(128, *grid)
    assert cos.shape == sin.shape == (grid[0] * grid[1] * grid[2], 64)

    freqs_f, freqs_h, freqs_w = ref.precompute_freqs_cis_3d(128)
    f, h, w = grid
    want = np.concatenate(
        [
            np.broadcast_to(freqs_f[:f].reshape(f, 1, 1, -1), (f, h, w, freqs_f.shape[-1])),
            np.broadcast_to(freqs_h[:h].reshape(1, h, 1, -1), (f, h, w, freqs_h.shape[-1])),
            np.broadcast_to(freqs_w[:w].reshape(1, 1, w, -1), (f, h, w, freqs_w.shape[-1])),
        ],
        axis=-1,
    ).reshape(f * h * w, -1)
    cos_err = float(np.max(np.abs(np.asarray(cos, np.float64) - want.real)))
    sin_err = float(np.max(np.abs(np.asarray(sin, np.float64) - want.imag)))
    assert report("rope table cos", cos_err) < 1e-7
    assert report("rope table sin", sin_err) < 1e-7


def test_rope_table_fp32_build_gap_is_bounded():
    """Plan risk #1, measured: how much does building the table in fp32 cost?

    torch precomputes in float64; TPU has none, so we build in float64 on the host
    and store fp32. maxdiffusion instead builds the whole table in fp32. This is the
    gap between those two choices — small, but it is the reason `freqs_dtype` exists
    and the reason the parity tests below pin the fp64 path.
    """
    cos64, sin64 = rope.precompute_freqs_3d(128, 3, 12, 10)
    cos32, sin32 = rope.precompute_freqs_3d(128, 3, 12, 10, freqs_dtype=np.float32)
    gap = max(
        float(np.max(np.abs(np.asarray(cos64) - np.asarray(cos32)))),
        float(np.max(np.abs(np.asarray(sin64) - np.asarray(sin32)))),
    )
    assert report("rope table fp64-vs-fp32 build", gap) < 1e-4


def test_apply_rope_is_interleaved_not_rotate_half():
    """The single highest-risk convention in the port (spec §9.1 item 1)."""
    seq, heads, head_dim = 36, 4, 32
    x = jax.random.normal(jax.random.key(3), (2, seq, heads, head_dim))
    cos, sin = rope.precompute_freqs_3d(head_dim, 3, 4, 3)
    got = np.asarray(rope.apply_rope(x, cos, sin), dtype=np.float64)

    freqs_f, freqs_h, freqs_w = ref.precompute_freqs_cis_3d(head_dim)
    freqs = np.concatenate(
        [
            np.broadcast_to(freqs_f[:3].reshape(3, 1, 1, -1), (3, 4, 3, freqs_f.shape[-1])),
            np.broadcast_to(freqs_h[:4].reshape(1, 4, 1, -1), (3, 4, 3, freqs_h.shape[-1])),
            np.broadcast_to(freqs_w[:3].reshape(1, 1, 3, -1), (3, 4, 3, freqs_w.shape[-1])),
        ],
        axis=-1,
    ).reshape(seq, 1, -1)
    flat = np.asarray(x, np.float64).reshape(2, seq, heads * head_dim)
    want = ref.rope_apply(flat, freqs, heads).reshape(2, seq, heads, head_dim)
    assert report("apply_rope", rel_error(got, want)) < 1e-6

    # A rotate-half implementation would be a different function entirely; assert the
    # two are far apart so this test cannot pass by accident.
    half = head_dim // 2
    c = np.asarray(cos, np.float64)[None, :, None, :]
    s = np.asarray(sin, np.float64)[None, :, None, :]
    x64 = np.asarray(x, np.float64)
    rotate_half = np.concatenate(
        [x64[..., :half] * c - x64[..., half:] * s, x64[..., :half] * s + x64[..., half:] * c], axis=-1
    )
    assert np.max(np.abs(rotate_half - want)) > 0.1


def test_dot_product_attention_masks_keys():
    query = jax.random.normal(jax.random.key(4), (2, 5, 3, 8))
    key = jax.random.normal(jax.random.key(5), (2, 7, 3, 8))
    value = jax.random.normal(jax.random.key(6), (2, 7, 3, 8))

    unmasked = dot_product_attention(query, key, value)
    all_true = dot_product_attention(query, key, value, jnp.ones((5, 7), bool))
    assert np.array_equal(np.asarray(unmasked), np.asarray(all_true))

    mask = jnp.ones((5, 7), bool).at[:, 3].set(False)
    masked = dot_product_attention(query, key, value, mask)
    # Dropping a key must be identical to deleting it from k/v entirely.
    keep = jnp.array([0, 1, 2, 4, 5, 6])
    dropped = dot_product_attention(query, key[:, keep], value[:, keep])
    assert report("attention mask vs key deletion", rel_error(masked, dropped)) < 1e-6


def test_patch_embedding_equals_conv3d():
    """Spec §9.5 item 22: reshape+dense must be the Conv3d it replaces."""
    model = build_model()
    latents = jax.random.normal(jax.random.key(7), LATENT_SHAPE)
    got = model.patch_embedding(latents)

    kernel = model.patch_embedding.kernel[...]
    conv = jax.lax.conv_general_dilated(
        jnp.transpose(latents, (0, 2, 3, 4, 1)),
        kernel,
        window_strides=PATCH,
        padding="VALID",
        dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
    )
    want = conv.reshape(conv.shape[0], -1, conv.shape[-1]) + model.patch_embedding.bias[...]
    assert report("patch_embedding vs conv3d", rel_error(got, want)) < 1e-6


def test_unpatchify_channel_index_is_last():
    """Spec §9.5 item 23, checked by explicit indexing rather than a second rearrange."""
    grid = (3, 4, 3)
    batch = 2
    seq = grid[0] * grid[1] * grid[2]
    p_t, p_h, p_w = PATCH
    packed = jax.random.normal(jax.random.key(8), (batch, seq, IN_CH * p_t * p_h * p_w))
    got = np.asarray(unpatchify(packed, grid, PATCH))
    assert got.shape == (batch, IN_CH, grid[0] * p_t, grid[1] * p_h, grid[2] * p_w)

    packed_np = np.asarray(packed)
    want = np.empty_like(got)
    for f in range(grid[0]):
        for h in range(grid[1]):
            for w in range(grid[2]):
                token = packed_np[:, (f * grid[1] + h) * grid[2] + w]
                for pt in range(p_t):
                    for ph in range(p_h):
                        for pw in range(p_w):
                            base = ((pt * p_h + ph) * p_w + pw) * IN_CH
                            cell = token[:, base : base + IN_CH]
                            want[:, :, f * p_t + pt, h * p_h + ph, w * p_w + pw] = cell
    assert np.array_equal(got, want)


# --------------------------------------------------------------------------- #
# whole tower
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mask_mode,with_context_mask",
    [("bidirectional", False), ("first_frame_causal", False), ("first_frame_causal", True)],
)
def test_video_dit_parity_per_token_timestep(mask_mode, with_context_mask):
    """The TI2V path: latent frame 0 is clean, every other token carries t."""
    model = build_model(video_attention_mask_mode=mask_mode)
    latents, timestep, context = sample_inputs()
    context_mask = None
    if with_context_mask:
        context_mask = jnp.ones((LATENT_SHAPE[0], CONTEXT_LEN), bool).at[:, 5:].set(False)

    got = model(latents, timestep, context, context_mask)
    want = ref.video_dit_forward(
        flat_params(model), ref_config(mask_mode), latents, timestep, context, context_mask
    )
    assert got.shape == want.shape == LATENT_SHAPE
    name = f"video_dit per-token {mask_mode}{' +ctx-mask' if with_context_mask else ''}"
    assert report(name, rel_error(got, want)) < MAX_REL


def test_video_dit_parity_global_timestep():
    """maxdiffusion's rank-3 t_mod path — the shape the action expert uses in M2."""
    model = build_model(per_token_timestep=False, video_attention_mask_mode="bidirectional")
    latents, timestep, context = sample_inputs()
    got = model(latents, timestep, context)
    want = ref.video_dit_forward(
        flat_params(model), ref_config("bidirectional"), latents, timestep, context, per_token=False
    )
    assert report("video_dit global timestep", rel_error(got, want)) < MAX_REL


def test_explicit_per_token_timestep_matches_derived():
    """Passing [B, Sv] directly must equal letting pre_dit derive it from [B]."""
    model = build_model(video_attention_mask_mode="first_frame_causal")
    latents, timestep, context = sample_inputs()
    derived = build_token_timesteps(timestep, 3, 12)
    assert np.array_equal(np.asarray(derived[:, :12]), np.zeros((2, 12), np.float32))
    assert np.allclose(np.asarray(derived[:, 12:]), np.asarray(timestep)[:, None])
    assert report(
        "explicit vs derived per-token t",
        rel_error(model(latents, derived, context), model(latents, timestep, context)),
    ) == 0.0


def test_first_frame_causal_isolates_the_first_latent_frame():
    """The premise the whole KV-cache inference path rests on.

    Frame-0 queries see only frame-0 keys, so the frame-0 output must not move when
    the future video changes. The `bidirectional` half of the test proves the check
    has teeth.
    """
    latents, timestep, context = sample_inputs()
    perturbed = latents.at[:, :, 1:].add(jax.random.normal(jax.random.key(9), (2, IN_CH, 2, 8, 6)))

    causal = build_model(video_attention_mask_mode="first_frame_causal")
    first_a = causal(latents, timestep, context)[:, :, 0]
    first_b = causal(perturbed, timestep, context)[:, :, 0]
    assert report("first_frame_causal leakage", rel_error(first_b, first_a)) < 1e-6

    bidirectional = build_model(video_attention_mask_mode="bidirectional")
    moved = rel_error(
        bidirectional(perturbed, timestep, context)[:, :, 0],
        bidirectional(latents, timestep, context)[:, :, 0],
    )
    assert moved > 1e-2, f"bidirectional first frame barely moved ({moved:.3e}); test has no teeth"


def test_self_attention_mask_shape_and_pattern():
    mask = np.asarray(build_video_self_attention_mask(120, 3))
    assert mask.shape == (360, 360)
    assert mask[:120, :120].all()
    assert not mask[:120, 120:].any()
    assert mask[120:, :].all()


def load_flat_params(module: nnx.Module, flat: dict[str, np.ndarray]) -> None:
    state = nnx.state(module, nnx.Param)
    updated = [
        (path, variable.replace(jnp.asarray(flat[".".join(str(p) for p in path)], dtype=variable[...].dtype)))
        for path, variable in nnx.to_flat_state(state)
    ]
    nnx.update(module, nnx.from_flat_state(updated))


def test_scan_layers_matches_unrolled():
    """Same weights, two layer layouts. This is exactly what the TPU loader has to do."""
    unrolled = build_model(scan_layers=False, video_attention_mask_mode="first_frame_causal")
    scanned = build_model(scan_layers=True, video_attention_mask_mode="first_frame_causal")
    load_flat_params(scanned, stack_blocks_for_scan(flat_params(unrolled), LAYERS))

    latents, timestep, context = sample_inputs()
    assert report(
        "scan vs unrolled",
        rel_error(scanned(latents, timestep, context), unrolled(latents, timestep, context)),
    ) < 1e-6


def test_unstack_round_trips():
    from tools.convert_weights import unstack_scanned_blocks

    original = flat_params(build_model(scan_layers=False))
    round_tripped = unstack_scanned_blocks(stack_blocks_for_scan(original, LAYERS), LAYERS)
    assert set(round_tripped) == set(original)
    for key, value in original.items():
        assert np.array_equal(round_tripped[key], value)


def test_remat_does_not_change_numerics():
    latents, timestep, context = sample_inputs()
    plain = build_model(remat_policy="none")
    rematted = build_model(remat_policy="full")
    assert rel_error(rematted(latents, timestep, context), plain(latents, timestep, context)) < 1e-7


# --------------------------------------------------------------------------- #
# 5B weight mapping
# --------------------------------------------------------------------------- #
def test_5b_param_inventory_matches_the_converted_checkpoint():
    """The key/shape table of the published Wan2.2-5B Orbax transformer, verbatim.

    Audited against the live tree's Orbax `_METADATA` on 2026-08-03 (825 keys, all
    matching); this test freezes the inventory so a later refactor that renames a
    module cannot silently break the load. `tools/convert_weights.py check-transformer`
    re-runs the comparison against the real tree.
    """
    from fastwam_tpu import load_config

    config = load_config(FULL_CONFIG).model.video_dit_config
    actual = video_dit_param_shapes(config, scan_layers=False)

    per_block = {
        "adaln_scale_shift_table": (1, 6, 3072),
        "norm2.scale": (3072,),
        "norm2.bias": (3072,),
        "ffn.act_fn.proj.kernel": (3072, 14336),
        "ffn.act_fn.proj.bias": (14336,),
        "ffn.proj_out.kernel": (14336, 3072),
        "ffn.proj_out.bias": (3072,),
    }
    for attn in ("attn1", "attn2"):
        for proj in ("query", "key", "value", "proj_attn"):
            per_block[f"{attn}.{proj}.kernel"] = (3072, 3072)
            per_block[f"{attn}.{proj}.bias"] = (3072,)
        for norm in ("norm_q", "norm_k"):
            per_block[f"{attn}.{norm}.scale"] = (3072,)

    expected = {f"blocks.{i}.{k}": v for i in range(30) for k, v in per_block.items()}
    expected.update(
        {
            "patch_embedding.kernel": (1, 2, 2, 48, 3072),
            "patch_embedding.bias": (3072,),
            "condition_embedder.time_embedder.linear_1.kernel": (256, 3072),
            "condition_embedder.time_embedder.linear_1.bias": (3072,),
            "condition_embedder.time_embedder.linear_2.kernel": (3072, 3072),
            "condition_embedder.time_embedder.linear_2.bias": (3072,),
            "condition_embedder.time_proj.kernel": (3072, 18432),
            "condition_embedder.time_proj.bias": (18432,),
            "condition_embedder.text_embedder.linear_1.kernel": (4096, 3072),
            "condition_embedder.text_embedder.linear_1.bias": (3072,),
            "condition_embedder.text_embedder.linear_2.kernel": (3072, 3072),
            "condition_embedder.text_embedder.linear_2.bias": (3072,),
            "proj_out.kernel": (3072, 192),
            "proj_out.bias": (192,),
            "scale_shift_table": (1, 2, 3072),
        }
    )
    assert len(expected) == 30 * 27 + 15 == 825
    assert compare_param_shapes(expected, actual, what="model") == 825


def test_5b_scanned_layout_stacks_the_block_axis():
    from fastwam_tpu import load_config

    config = load_config(FULL_CONFIG).model.video_dit_config
    scanned = video_dit_param_shapes(config, scan_layers=True)
    assert len(scanned) == 27 + 15
    assert scanned["blocks.attn1.query.kernel"] == (30, 3072, 3072)
    assert scanned["blocks.adaln_scale_shift_table"] == (30, 1, 6, 3072)
    assert scanned["patch_embedding.kernel"] == (1, 2, 2, 48, 3072)


def test_mapping_audit_reports_every_kind_of_mismatch():
    """The house rule: missing / extra / wrong shape all raise, none of them log."""
    base = {"a": (2, 3), "b": (4,)}
    with pytest.raises(ConversionError, match="missing"):
        compare_param_shapes(base, {"a": (2, 3)})
    with pytest.raises(ConversionError, match="extra"):
        compare_param_shapes(base, {**base, "c": (1,)})
    with pytest.raises(ConversionError, match="shape a"):
        compare_param_shapes(base, {"a": (3, 2), "b": (4,)})


# --------------------------------------------------------------------------- #
# the hooks M2 and M3 are going to pull on
# --------------------------------------------------------------------------- #
def test_block_halves_compose_to_the_whole_block():
    """MoT drives pre_attention / post_attention around its own mixed attention.

    Running them back to back with a single-expert attention in between must be
    bit-identical to calling the block, or M2 starts from a broken split.
    """
    model = build_model()
    block = model.blocks[0]
    latents, timestep, context = sample_inputs()
    state = model.pre_dit(latents, timestep, context)
    mask = model.self_attention_mask(state.grid)

    whole = block(state.tokens, state.context, state.t_mod, state.cos, state.sin, mask, state.context_mask)
    (query, key, value), gates = block.pre_attention(state.tokens, state.t_mod, state.cos, state.sin)
    assert query.shape == key.shape == value.shape == (2, 36, HEADS, HEAD_DIM)
    attn_out = dot_product_attention(query, key, value, mask)
    halves = block.post_attention(state.tokens, attn_out, gates, state.context, state.context_mask)
    assert rel_error(halves, whole) == 0.0


def test_cross_attention_kv_cache_hook_is_exact():
    """The context is constant across denoising steps, so its K/V can be cached."""
    model = build_model()
    block = model.blocks[0]
    latents, timestep, context = sample_inputs()
    state = model.pre_dit(latents, timestep, context)
    mask = model.self_attention_mask(state.grid)

    plain = block(state.tokens, state.context, state.t_mod, state.cos, state.sin, mask, state.context_mask)
    cached = block(
        state.tokens,
        state.context,
        state.t_mod,
        state.cos,
        state.sin,
        mask,
        state.context_mask,
        cached_kv=block.compute_context_kv(state.context),
    )
    assert rel_error(cached, plain) == 0.0


# --------------------------------------------------------------------------- #
# checkpoint restore
# --------------------------------------------------------------------------- #
def _small_config():
    """The 5 B VideoDiTConfig, narrowed so a real save/restore fits in a CPU test.

    Only the widths change; patch size, the per-token timestep path and the mask mode
    are the released values, so the tree SHAPE under test is the real one.
    """
    import dataclasses

    from fastwam_tpu import load_config

    return dataclasses.replace(
        load_config(FULL_CONFIG).model.video_dit_config,
        hidden_dim=64,
        ffn_dim=128,
        num_heads=2,
        attn_head_dim=32,
        num_layers=3,
        freq_dim=16,
        text_dim=24,
        in_dim=4,
        out_dim=4,
    )


@pytest.fixture(scope="module")
def saved_checkpoints(tmp_path_factory):
    """One model written out under both list-index key spellings.

    The converted Wan2.2-5B tree was written by flax 0.10, whose
    `to_pure_dict` spells list indices as the strings "0", "1", ... (its orbax
    `_METADATA` records `key_type: 2` for every `blocks` index). flax 0.12 spells them
    as ints. Restoring one against the other is a structure mismatch on a tree that is
    in fact identical, so the loader has to handle both — and this fixture is what
    proves it, without needing GCS credentials or 17 GiB of disk.
    """
    import orbax.checkpoint as ocp

    from fastwam_tpu.models.wan.transformer import stringify_tree_keys

    config = _small_config()
    model = WanVideoDiT.from_config(config, rngs=nnx.Rngs(7))
    pure = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    root = tmp_path_factory.mktemp("dit_ckpt")

    # The context manager matters: orbax saves asynchronously, and letting the
    # interpreter tear down mid-commit raises "cannot schedule new futures after
    # shutdown" instead of writing the tree.
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(str(root / "flax010"), stringify_tree_keys(pure))
        checkpointer.save(str(root / "flax012"), pure)
        truncated = {key: value for key, value in pure.items() if key != "proj_out"}
        checkpointer.save(str(root / "truncated"), stringify_tree_keys(truncated))
    return config, model, root


def _small_inputs(config):
    """Inputs sized for `_small_config`, not for the module-level parity config."""
    keys = jax.random.split(jax.random.key(11), 2)
    latents = jax.random.normal(keys[0], (2, config.in_dim, 3, 8, 6))
    context = jax.random.normal(keys[1], (2, 5, config.text_dim))
    return latents, jnp.array([300.0, 700.0]), context


@pytest.mark.parametrize("layout", ["flax010", "flax012"])
@pytest.mark.parametrize("scan_layers", [False, True])
def test_load_video_dit_round_trips_both_key_spellings(saved_checkpoints, layout, scan_layers):
    from fastwam_tpu.models.wan.transformer import load_video_dit

    config, model, root = saved_checkpoints
    latents, timestep, context = _small_inputs(config)
    want = model(latents, timestep, context)

    restored = load_video_dit(str(root / layout), config, scan_layers=scan_layers)
    got = restored(latents, timestep, context)
    # Unrolled must be bit-identical; scanned differs only by fp32 reassociation.
    bar = 1e-5 if scan_layers else 0.0
    assert report(f"restore {layout} scan={scan_layers}", rel_error(got, want)) <= bar


def test_load_video_dit_raises_on_a_truncated_tree(saved_checkpoints):
    """No partial restores. The reference's strict=False habit is the bug being avoided."""
    from fastwam_tpu.models.wan.transformer import WanVideoDiTConfigError, load_video_dit

    config, _, root = saved_checkpoints
    with pytest.raises(WanVideoDiTConfigError, match="every key style and restore path"):
        load_video_dit(str(root / "truncated"), config)
