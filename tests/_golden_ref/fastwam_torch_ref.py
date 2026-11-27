"""float64 numpy transcription of FastWAM's torch model — the M1/M2 parity golden.

Source, line by line:
    reference/FastWAM/src/fastwam/models/wan22/wan_video_dit.py (git HEAD 45d8e14)
        sinusoidal_embedding_1d       L189-193
        precompute_freqs_cis / _3d    L196-211
        rope_apply                    L214-220
        RMSNorm                       L147-158
        flash_attention (= plain SDPA) L173-181
        SelfAttention / CrossAttention L171-220
        DiTBlock.forward              L249-268
        Head.forward                  L300-308
        WanVideoDiT.pre_dit           L509-620, post_dit L622-626, unpatchify L410-415
        build_video_to_video_mask     L473-507
    reference/FastWAM/src/fastwam/models/wan22/action_dit.py                       [M2]
        ActionDiT.pre_dit             L226-299, post_dit L301-302
    reference/FastWAM/src/fastwam/models/wan22/mot.py                              [M2]
        MoT._split_modulation         L58-75, _build_expert_attention_io L124-184,
        _apply_expert_post_block      L99-122, forward L447-556
    reference/FastWAM/src/fastwam/models/wan22/fastwam.py                          [M2]
        _build_mot_attention_mask     L385-407, _compute_video_loss_per_sample L409-446,
        training_loss                 L448-568, _predict_joint_noise L570-632,
        infer_joint                   L725-903, _append_proprio_to_context L219-240
    reference/FastWAM/src/fastwam/models/wan22/schedulers/scheduler_continuous.py  [M2]
        the whole 88-line file
    reference/FastWAM/scripts/preprocess_action_dit_backbone.py                    [M2]
        _interpolate_last_dim L53-58, _resize_tensor_to_shape L61-96, main L196-210

Why this and not a vendored copy of maxdiffusion (the plan's suggested golden):
    1. maxdiffusion is not importable here — `maxdiffusion.models.attention_flax`
       drags in huggingface_hub, transformers, einops, msgpack, aqt quantization and
       max_utils, and the host disk is at 99%.
    2. More importantly, our port was *transcribed from* maxdiffusion, so comparing
       against a vendored copy of it mostly proves the copy was faithful, not that
       the model is right. FastWAM's torch source is the actual authority — it is
       what the released weights were trained with — so transcribing *that*, in a
       different language and numeric precision, gives an independent check.
    3. Working in float64 additionally bounds plan risk #1 (torch builds the RoPE and
       timestep tables in fp64; TPU only has fp32). The residual error the tests
       report is therefore an honest fp32-rounding budget, not a definitional gap.

Parameters come in as a flat dict keyed by our NNX paths (`blocks.0.attn1.query.kernel`,
...). The mapping to torch names is spelled out in `TORCH_KEY_EQUIVALENTS` below, which
doubles as the reference for M2's official-checkpoint converter.

Everything is float64. Nothing here may import jax.
"""

import numpy as np

__all__ = [
    "TORCH_KEY_EQUIVALENTS",
    "TORCH_KEY_EQUIVALENTS_ACTION",
    "TORCH_TOP_LEVEL_PREFIXES",
    "RefConfig",
    "ActionRefConfig",
    "video_dit_forward",
    "video_dit_pre",
    "video_dit_post",
    "action_dit_pre",
    "action_dit_post",
    "action_dit_forward",
    "dit_block",
    "block_pre_attention",
    "block_post_attention",
    "mot_forward",
    "build_mot_attention_mask",
    "append_proprio_to_context",
    "fastwam_predict",
    "fastwam_training_loss",
    "denoise_joint",
    "denoise_action",
    "sinusoidal_embedding_1d",
    "phi",
    "add_noise",
    "training_target",
    "training_weight",
    "build_inference_schedule",
    "resize_tensor_to_shape",
    "action_backbone_from_video",
]


# Ours (= maxdiffusion / the converted Wan2.2 Orbax tree)  ->  FastWAM torch name.
# The norm2/norm3 swap is the trap: FastWAM's norm3 is the affine pre-cross-attention
# norm, ours is called norm2.
#
# Layout: every `.kernel` here is a torch `.weight` TRANSPOSED — torch stores Linear as
# [out, in], flax as [in, out]. The exceptions are the parameters that are not Linear
# weights (`adaln_scale_shift_table`, `scale_shift_table`, the RMSNorm/LayerNorm scales
# and every bias), which are identical in both. `patch_embedding.kernel` is the one
# genuinely reshaped tensor: torch Conv3d [out, in, p_t, p_h, p_w] -> [p_t,p_h,p_w,in,out].
TORCH_KEY_EQUIVALENTS = {
    "patch_embedding.{kernel,bias}": "patch_embedding.{weight,bias}",
    "condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.time_proj": "time_projection.1",
    "condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "blocks.N.adaln_scale_shift_table": "blocks.N.modulation",
    "blocks.N.attn1.{query,key,value,proj_attn}": "blocks.N.self_attn.{q,k,v,o}",
    "blocks.N.attn1.{norm_q,norm_k}.scale": "blocks.N.self_attn.{norm_q,norm_k}.weight",
    "blocks.N.attn2.{query,key,value,proj_attn}": "blocks.N.cross_attn.{q,k,v,o}",
    "blocks.N.attn2.{norm_q,norm_k}.scale": "blocks.N.cross_attn.{norm_q,norm_k}.weight",
    "blocks.N.norm2.{scale,bias}": "blocks.N.norm3.{weight,bias}",  # affine, pre-cross-attn
    "blocks.N.norm1": "blocks.N.norm1",  # no params (elementwise_affine=False)
    "blocks.N.norm3": "blocks.N.norm2",  # no params, pre-FFN
    "blocks.N.ffn.act_fn.proj": "blocks.N.ffn.0",
    "blocks.N.ffn.proj_out": "blocks.N.ffn.2",
    "norm_out": "head.norm",  # no params
    "proj_out": "head.head",
    "scale_shift_table": "head.modulation",
}

# The action expert (M2). Its blocks are the SAME DiTBlock class, so every
# `blocks.N.*` row above applies verbatim; only the four non-block modules differ, and
# `proj_out` means something else here (a plain Linear head, not the patchified one).
TORCH_KEY_EQUIVALENTS_ACTION = {
    "action_encoder": "action_encoder",
    "condition_embedder.time_embedder.linear_1": "time_embedding.0",
    "condition_embedder.time_embedder.linear_2": "time_embedding.2",
    "condition_embedder.time_proj": "time_projection.1",
    "condition_embedder.text_embedder.linear_1": "text_embedding.0",
    "condition_embedder.text_embedder.linear_2": "text_embedding.2",
    "blocks.N.*": "blocks.N.*",  # identical to the video table above
    "proj_out": "head",  # Linear(1024, action_dim); NOT the video tower's proj_out
}

# Where the two towers sit inside the released checkpoint. `FastWAM.save_checkpoint`
# (fastwam.py:1088-1098) writes {"mot": mot.state_dict(), "proprio_encoder": ..., ...},
# and MoT holds the experts in an `nn.ModuleDict` named `mixtures`.
TORCH_TOP_LEVEL_PREFIXES = {
    "video": "mot.mixtures.video.",
    "action": "mot.mixtures.action.",
    "proprio_encoder": "proprio_encoder.",
}


class RefConfig:
    """The handful of numbers the reference needs, mirroring VideoDiTConfig."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        freq_dim: int,
        out_dim: int,
        patch_size: tuple[int, int, int],
        eps: float,
        video_attention_mask_mode: str = "first_frame_causal",
    ):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.freq_dim = freq_dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        self.video_attention_mask_mode = video_attention_mask_mode


class ActionRefConfig:
    """The action expert's numbers, mirroring ActionDiTConfig.

    `hidden_dim != num_heads * attn_head_dim` here — that inequality IS the action
    expert (1024 hidden, 3072 attention width).
    """

    def __init__(
        self,
        *,
        action_dim: int,
        hidden_dim: int,
        num_heads: int,
        attn_head_dim: int,
        num_layers: int,
        freq_dim: int,
        eps: float,
        rope_cache_len: int = 1024,
    ):
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_head_dim = attn_head_dim
        self.num_layers = num_layers
        self.freq_dim = freq_dim
        self.eps = eps
        self.rope_cache_len = rope_cache_len


# --------------------------------------------------------------------------- #
# primitives (wan_video_dit.py L147-220)
# --------------------------------------------------------------------------- #
def silu(x):
    return x / (1.0 + np.exp(-x))


def gelu_tanh(x):
    """nn.GELU(approximate='tanh')."""
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return 0.5 * x * (1.0 + np.tanh(inner))


def rms_norm(x, weight, eps):
    """RMSNorm.forward: norm in fp32-or-better, times weight."""
    return x * (1.0 / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)) * weight


def layer_norm(x, eps, weight=None, bias=None):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)
    if weight is not None:
        out = out * weight
    if bias is not None:
        out = out + bias
    return out


def sinusoidal_embedding_1d(dim, position):
    """cat([cos, sin]) — COS FIRST (L189-193)."""
    position = np.asarray(position, dtype=np.float64)
    freqs = np.power(10000.0, -np.arange(dim // 2, dtype=np.float64) / (dim // 2))
    sinusoid = np.outer(position, freqs)
    return np.concatenate([np.cos(sinusoid), np.sin(sinusoid)], axis=1)


def precompute_freqs_cis(dim, end=1024, theta=10000.0):
    """torch.polar(1, outer(arange(end), 1/theta**(arange(0,dim,2)/dim))) (L202-211)."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64)[: dim // 2] / dim))
    angles = np.outer(np.arange(end, dtype=np.float64), freqs)
    return np.exp(1j * angles)


def precompute_freqs_cis_3d(dim, end=1024, theta=10000.0):
    """(frame, height, width) = (dim - 2*(dim//3), dim//3, dim//3) at dim=128 (L196-201).

    The `hw` form below is maxdiffusion's `2*(dim//6)`; it is identical at dim=128
    (42/42/44) and stays even for dims where FastWAM's `dim//3` would not, which is
    what lets the tests run at a small head_dim.
    """
    hw = 2 * (dim // 6)
    return (
        precompute_freqs_cis(dim - 2 * hw, end, theta),
        precompute_freqs_cis(hw, end, theta),
        precompute_freqs_cis(hw, end, theta),
    )


def rope_apply(x, freqs, num_heads):
    """x [B,S,n*d] -> view_as_complex * freqs -> view_as_real (L214-220).

    `freqs` is [S, 1, d/2] complex, exactly the layout FastWAM builds in pre_dit.
    """
    b, s, _ = x.shape
    x = x.reshape(b, s, num_heads, -1)
    pairs = x.reshape(b, s, num_heads, -1, 2)
    complex_x = pairs[..., 0] + 1j * pairs[..., 1]
    rotated = complex_x * freqs[None]
    out = np.stack([rotated.real, rotated.imag], axis=-1)
    return out.reshape(b, s, -1)


def sdpa(query, key, value, mask=None):
    """F.scaled_dot_product_attention on [B, n, S, d], boolean attn_mask."""
    scale = 1.0 / np.sqrt(query.shape[-1])
    logits = np.einsum("bnqd,bnkd->bnqk", query, key) * scale
    if mask is not None:
        logits = np.where(mask, logits, -np.inf)
    logits = logits - logits.max(axis=-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.einsum("bnqk,bnkd->bnqd", weights, value)


def _split_heads(x, num_heads):
    """'b s (n d) -> b n s d'."""
    b, s, _ = x.shape
    return x.reshape(b, s, num_heads, -1).transpose(0, 2, 1, 3)


def _merge_heads(x):
    """'b n s d -> b s (n d)'."""
    b, n, s, d = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, s, n * d)


def _linear(params, prefix, x):
    return x @ params[f"{prefix}.kernel"] + params[f"{prefix}.bias"]


# --------------------------------------------------------------------------- #
# attention (L171-220)
# --------------------------------------------------------------------------- #
def self_attention(params, prefix, x, freqs, num_heads, eps, mask=None):
    query = rms_norm(_linear(params, f"{prefix}.query", x), params[f"{prefix}.norm_q.scale"], eps)
    key = rms_norm(_linear(params, f"{prefix}.key", x), params[f"{prefix}.norm_k.scale"], eps)
    value = _linear(params, f"{prefix}.value", x)
    query = _split_heads(rope_apply(query, freqs, num_heads), num_heads)
    key = _split_heads(rope_apply(key, freqs, num_heads), num_heads)
    out = sdpa(query, key, _split_heads(value, num_heads), mask)
    return _linear(params, f"{prefix}.proj_attn", _merge_heads(out))


def cross_attention(params, prefix, x, context, num_heads, eps, mask=None):
    query = rms_norm(_linear(params, f"{prefix}.query", x), params[f"{prefix}.norm_q.scale"], eps)
    key = rms_norm(_linear(params, f"{prefix}.key", context), params[f"{prefix}.norm_k.scale"], eps)
    value = _linear(params, f"{prefix}.value", context)
    out = sdpa(
        _split_heads(query, num_heads), _split_heads(key, num_heads), _split_heads(value, num_heads), mask
    )
    return _linear(params, f"{prefix}.proj_attn", _merge_heads(out))


# --------------------------------------------------------------------------- #
# block (L249-268), split at the mixed-attention seam the MoT drives (mot.py L124-184)
# --------------------------------------------------------------------------- #
def split_modulation(params, prefix, t_mod):
    """MoT._split_modulation (mot.py:58-75). Rank 4 t_mod = per-token, rank 3 = shared."""
    has_seq = t_mod.ndim == 4
    chunk_dim = 2 if has_seq else 1
    modulation = params[f"{prefix}.adaln_scale_shift_table"]
    if has_seq:
        modulation = modulation[None]
    parts = np.split(modulation + t_mod, 6, axis=chunk_dim)
    if has_seq:
        parts = [p.squeeze(2) for p in parts]
    return parts


def block_pre_attention(params, prefix, x, t_mod, freqs, num_heads, eps):
    """-> ((q, k, v) each [B,S,n*d], post-RoPE), (gate_msa, shift_mlp, scale_mlp, gate_mlp)."""
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = split_modulation(params, prefix, t_mod)
    # norm1: non-affine LayerNorm before self-attention.
    attn_in = layer_norm(x, eps) * (1 + scale_msa) + shift_msa
    attn = f"{prefix}.attn1"
    query = rms_norm(_linear(params, f"{attn}.query", attn_in), params[f"{attn}.norm_q.scale"], eps)
    key = rms_norm(_linear(params, f"{attn}.key", attn_in), params[f"{attn}.norm_k.scale"], eps)
    value = _linear(params, f"{attn}.value", attn_in)
    query = rope_apply(query, freqs, num_heads)
    key = rope_apply(key, freqs, num_heads)
    return (query, key, value), (gate_msa, shift_mlp, scale_mlp, gate_mlp)


def block_post_attention(params, prefix, x, mixed, gates, context, num_heads, eps, context_mask=None):
    """MoT._apply_expert_post_block (mot.py:99-122). `mixed` is [B, S, n*d], merged."""
    gate_msa, shift_mlp, scale_mlp, gate_mlp = gates
    if context_mask is not None and context_mask.ndim == 3:
        context_mask = context_mask[:, None]  # (B, 1, S, L), broadcast over heads

    x = x + gate_msa * _linear(params, f"{prefix}.attn1.proj_attn", mixed)

    # Ours-norm2 == FastWAM-norm3: affine, before cross-attention. NO gate here.
    cross_in = layer_norm(x, eps, params[f"{prefix}.norm2.scale"], params[f"{prefix}.norm2.bias"])
    x = x + cross_attention(params, f"{prefix}.attn2", cross_in, context, num_heads, eps, mask=context_mask)

    # Ours-norm3 == FastWAM-norm2: non-affine, before the FFN.
    ffn_in = layer_norm(x, eps) * (1 + scale_mlp) + shift_mlp
    hidden = gelu_tanh(_linear(params, f"{prefix}.ffn.act_fn.proj", ffn_in))
    ffn_out = _linear(params, f"{prefix}.ffn.proj_out", hidden)
    return x + gate_mlp * ffn_out


def dit_block(
    params, index, x, context, t_mod, freqs, num_heads, eps, context_mask=None, self_attn_mask=None
):
    """The whole block = the two halves around a single-expert SDPA."""
    prefix = f"blocks.{index}"
    (query, key, value), gates = block_pre_attention(params, prefix, x, t_mod, freqs, num_heads, eps)
    mixed = _merge_heads(
        sdpa(
            _split_heads(query, num_heads),
            _split_heads(key, num_heads),
            _split_heads(value, num_heads),
            self_attn_mask,
        )
    )
    return block_post_attention(
        params, prefix, x, mixed, gates, context, num_heads, eps, context_mask=context_mask
    )


# --------------------------------------------------------------------------- #
# whole tower (L410-626)
# --------------------------------------------------------------------------- #
def build_video_to_video_mask(video_seq_len, tokens_per_frame, mode):
    if mode == "bidirectional":
        return None
    if mode != "first_frame_causal":
        raise ValueError(f"unsupported mask mode {mode!r}")
    mask = np.ones((video_seq_len, video_seq_len), dtype=bool)
    first = min(tokens_per_frame, video_seq_len)
    mask[:first, first:] = False
    return mask


def patchify(params, latents, patch_size):
    """Conv3d with kernel == stride, written as the gather + matmul it is."""
    b, c, t, h, w = latents.shape
    p_t, p_h, p_w = patch_size
    grid = (t // p_t, h // p_h, w // p_w)
    x = latents.transpose(0, 2, 3, 4, 1)  # [B,T,H,W,C]
    x = x.reshape(b, grid[0], p_t, grid[1], p_h, grid[2], p_w, c)
    x = x.transpose(0, 1, 3, 5, 2, 4, 6, 7)  # [B,f,h,w,p_t,p_h,p_w,C]
    x = x.reshape(b, grid[0] * grid[1] * grid[2], p_t * p_h * p_w * c)
    kernel = params["patch_embedding.kernel"].reshape(-1, params["patch_embedding.kernel"].shape[-1])
    return x @ kernel + params["patch_embedding.bias"], grid


def unpatchify(x, grid, patch_size):
    """'b (f h w) (x y z c) -> b c (f x) (h y) (w z)' (L410-415)."""
    b = x.shape[0]
    p_t, p_h, p_w = patch_size
    f, h, w = grid
    x = x.reshape(b, f, h, w, p_t, p_h, p_w, -1)
    x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
    return x.reshape(b, -1, f * p_t, h * p_h, w * p_w)


def build_token_timesteps(timestep, num_frames, tokens_per_frame):
    """L541-547: every token gets t, then latent frame 0 is overwritten with 0."""
    batch = timestep.shape[0]
    tokens = np.ones((batch, num_frames, tokens_per_frame), dtype=np.float64) * timestep[:, None, None]
    tokens[:, 0, :] = 0
    return tokens.reshape(batch, -1)


def _as_f64_params(params):
    return {k: np.asarray(v, dtype=np.float64) for k, v in params.items()}


def video_dit_pre(params, cfg: RefConfig, latents, timestep, context, context_mask=None, per_token=True):
    """WanVideoDiT.pre_dit (L509-620) -> the dict the blocks and the head consume."""
    latents = np.asarray(latents, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    timestep = np.asarray(timestep, dtype=np.float64)
    batch = latents.shape[0]

    tokens, grid = patchify(params, latents, cfg.patch_size)
    tokens_per_frame = grid[1] * grid[2]

    if timestep.ndim == 2:
        token_timesteps = timestep
    elif per_token:
        token_timesteps = build_token_timesteps(timestep, grid[0], tokens_per_frame)
    else:
        token_timesteps = None

    if token_timesteps is not None:
        t_sin = sinusoidal_embedding_1d(cfg.freq_dim, token_timesteps.reshape(-1))
    else:
        t_sin = sinusoidal_embedding_1d(cfg.freq_dim, timestep)
    t = _linear(params, "condition_embedder.time_embedder.linear_2",
                silu(_linear(params, "condition_embedder.time_embedder.linear_1", t_sin)))
    t_mod = _linear(params, "condition_embedder.time_proj", silu(t))
    if token_timesteps is not None:
        t = t.reshape(batch, -1, cfg.hidden_dim)
        t_mod = t_mod.reshape(batch, -1, 6, cfg.hidden_dim)
    else:
        t_mod = t_mod.reshape(batch, 6, cfg.hidden_dim)

    context_emb = _linear(params, "condition_embedder.text_embedder.linear_2",
                          gelu_tanh(_linear(params, "condition_embedder.text_embedder.linear_1", context)))

    seq_len = tokens.shape[1]
    attn_context_mask = None
    if context_mask is not None:
        context_mask = np.asarray(context_mask)
        attn_context_mask = np.broadcast_to(
            context_mask[:, None, :], (batch, seq_len, context_mask.shape[1])
        )

    # FastWAM builds the tables once at end=1024 and slices; entry i does not depend
    # on `end`, so this is the same numbers either way.
    freqs_f, freqs_h, freqs_w = precompute_freqs_cis_3d(cfg.attn_head_dim, end=1024)
    f, h, w = grid
    freqs = np.concatenate(
        [
            np.broadcast_to(freqs_f[:f].reshape(f, 1, 1, -1), (f, h, w, freqs_f.shape[-1])),
            np.broadcast_to(freqs_h[:h].reshape(1, h, 1, -1), (f, h, w, freqs_h.shape[-1])),
            np.broadcast_to(freqs_w[:w].reshape(1, 1, w, -1), (f, h, w, freqs_w.shape[-1])),
        ],
        axis=-1,
    ).reshape(f * h * w, 1, -1)

    return {
        "tokens": tokens,
        "t": t,
        "t_mod": t_mod,
        "context": context_emb,
        "context_mask": attn_context_mask,
        "freqs": freqs,
        "grid": grid,
        "tokens_per_frame": tokens_per_frame,
    }


def video_dit_post(params, cfg: RefConfig, x, t, grid):
    """Head (L300-308) + unpatchify. Note it consumes `t`, NOT `t_mod`."""
    table = params["scale_shift_table"]
    if t.ndim == 3:
        shift, scale = np.split(table[None] + t[:, :, None, :], 2, axis=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
    else:
        # FastWAM's own global branch is `modulation + t_mod` with shapes [1,2,D] and
        # [B,D], which only broadcasts when B == 2 — it is dead code there (every
        # shipped config is per-token). This follows maxdiffusion's correct form.
        shift, scale = np.split(table + t[:, None, :], 2, axis=1)
    head_in = layer_norm(x, cfg.eps) * (1 + scale) + shift
    return unpatchify(_linear(params, "proj_out", head_in), grid, cfg.patch_size)


def video_dit_forward(params, cfg: RefConfig, latents, timestep, context, context_mask=None, per_token=True):
    """Full pre_dit -> blocks -> post_dit, in float64."""
    params = _as_f64_params(params)
    pre = video_dit_pre(params, cfg, latents, timestep, context, context_mask, per_token)
    self_attn_mask = build_video_to_video_mask(
        pre["tokens"].shape[1], pre["tokens_per_frame"], cfg.video_attention_mask_mode
    )

    x = pre["tokens"]
    for index in range(cfg.num_layers):
        x = dit_block(
            params, index, x, pre["context"], pre["t_mod"], pre["freqs"], cfg.num_heads, cfg.eps,
            context_mask=pre["context_mask"], self_attn_mask=self_attn_mask,
        )
    return video_dit_post(params, cfg, x, pre["t"], pre["grid"])


# --------------------------------------------------------------------------- #
# action expert (action_dit.py L226-302)
# --------------------------------------------------------------------------- #
def action_dit_pre(params, cfg: ActionRefConfig, action, timestep, context, context_mask=None):
    """ActionDiT.pre_dit (L226-299).

    t_mod comes out rank 3 ([B, 6, hidden]): one modulation shared by all action tokens,
    unlike the video tower's per-token rank 4.
    """
    action = np.asarray(action, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    timestep = np.asarray(timestep, dtype=np.float64)
    batch, seq_len, _ = action.shape

    t_sin = sinusoidal_embedding_1d(cfg.freq_dim, timestep)
    t = _linear(params, "condition_embedder.time_embedder.linear_2",
                silu(_linear(params, "condition_embedder.time_embedder.linear_1", t_sin)))
    t_mod = _linear(params, "condition_embedder.time_proj", silu(t)).reshape(batch, 6, cfg.hidden_dim)

    tokens = _linear(params, "action_encoder", action)
    context_emb = _linear(params, "condition_embedder.text_embedder.linear_2",
                          gelu_tanh(_linear(params, "condition_embedder.text_embedder.linear_1", context)))

    attn_context_mask = None
    if context_mask is not None:
        context_mask = np.asarray(context_mask)
        attn_context_mask = np.broadcast_to(
            context_mask[:, None, :], (batch, seq_len, context_mask.shape[1])
        )

    # precompute_freqs_cis(attn_head_dim, end=1024)[:seq_len] -> [S, 1, d/2] complex.
    freqs = precompute_freqs_cis(cfg.attn_head_dim, end=cfg.rope_cache_len)[:seq_len]
    freqs = freqs.reshape(seq_len, 1, -1)

    return {
        "tokens": tokens,
        "t": t,
        "t_mod": t_mod,
        "context": context_emb,
        "context_mask": attn_context_mask,
        "freqs": freqs,
    }


def action_dit_post(params, tokens):
    """ActionDiT.post_dit (L301-302): a plain Linear, no adaLN and no norm."""
    return _linear(params, "proj_out", tokens)


def action_dit_forward(params, cfg: ActionRefConfig, action, timestep, context, context_mask=None):
    """ActionDiT.forward (L304-337) — the tower run ALONE, all-to-all self attention."""
    params = _as_f64_params(params)
    pre = action_dit_pre(params, cfg, action, timestep, context, context_mask)
    x = pre["tokens"]
    for index in range(cfg.num_layers):
        x = dit_block(
            params, index, x, pre["context"], pre["t_mod"], pre["freqs"], cfg.num_heads, cfg.eps,
            context_mask=pre["context_mask"], self_attn_mask=None,
        )
    return action_dit_post(params, x)


# --------------------------------------------------------------------------- #
# MoT (mot.py L447-556) and the FastWAM mask (fastwam.py L385-407)
# --------------------------------------------------------------------------- #
def build_mot_attention_mask(
    video_seq_len, action_seq_len, video_tokens_per_frame, mode="first_frame_causal"
):
    total = video_seq_len + action_seq_len
    mask = np.zeros((total, total), dtype=bool)
    video_block = build_video_to_video_mask(video_seq_len, video_tokens_per_frame, mode)
    mask[:video_seq_len, :video_seq_len] = (
        True if video_block is None else video_block
    )
    mask[video_seq_len:, video_seq_len:] = True
    first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
    mask[video_seq_len:, :first_frame_tokens] = True
    return mask


def mot_forward(video_params, action_params, video_cfg, action_cfg, video_pre, action_pre, mask):
    """MoT.forward: per layer, both experts' q/k/v concatenated into ONE attention.

    Returns the two towers' block outputs (before either `post_dit`).
    """
    num_heads = video_cfg.num_heads
    x_video = video_pre["tokens"]
    x_action = action_pre["tokens"]
    video_seq_len = x_video.shape[1]

    for index in range(video_cfg.num_layers):
        prefix = f"blocks.{index}"
        (q_v, k_v, v_v), gates_v = block_pre_attention(
            video_params, prefix, x_video, video_pre["t_mod"], video_pre["freqs"], num_heads, video_cfg.eps
        )
        (q_a, k_a, v_a), gates_a = block_pre_attention(
            action_params, prefix, x_action, action_pre["t_mod"], action_pre["freqs"], num_heads,
            action_cfg.eps,
        )
        query = np.concatenate([q_v, q_a], axis=1)
        key = np.concatenate([k_v, k_a], axis=1)
        value = np.concatenate([v_v, v_a], axis=1)
        mixed = _merge_heads(
            sdpa(
                _split_heads(query, num_heads),
                _split_heads(key, num_heads),
                _split_heads(value, num_heads),
                mask,
            )
        )
        x_video = block_post_attention(
            video_params, prefix, x_video, mixed[:, :video_seq_len], gates_v, video_pre["context"],
            num_heads, video_cfg.eps, context_mask=video_pre["context_mask"],
        )
        x_action = block_post_attention(
            action_params, prefix, x_action, mixed[:, video_seq_len:], gates_a, action_pre["context"],
            num_heads, action_cfg.eps, context_mask=action_pre["context_mask"],
        )
    return x_video, x_action


# --------------------------------------------------------------------------- #
# FastWAM assembly (fastwam.py)
# --------------------------------------------------------------------------- #
def append_proprio_to_context(proprio_params, context, context_mask, proprio):
    """_append_proprio_to_context (fastwam.py:219-240): one extra TEXT-space token."""
    context = np.asarray(context, dtype=np.float64)
    token = _linear(proprio_params, "proprio_encoder", np.asarray(proprio, np.float64)[:, None, :])
    ones = np.ones((context_mask.shape[0], 1), dtype=bool)
    mask = np.concatenate([np.asarray(context_mask, bool), ones], axis=1)
    return np.concatenate([context, token], axis=1), mask


def fastwam_predict(
    params, video_cfg: RefConfig, action_cfg: ActionRefConfig,
    noisy_latents, noisy_action, t_video, t_action, context, context_mask,
):
    """_predict_joint_noise (fastwam.py:570-632) -> (pred_video, pred_action).

    `params` is {"video": {...}, "action": {...}} in this repo's key names.
    """
    video_params = _as_f64_params(params["video"])
    action_params = _as_f64_params(params["action"])

    video_pre = video_dit_pre(video_params, video_cfg, noisy_latents, t_video, context, context_mask)
    action_pre = action_dit_pre(action_params, action_cfg, noisy_action, t_action, context, context_mask)
    mask = build_mot_attention_mask(
        video_pre["tokens"].shape[1], action_pre["tokens"].shape[1], video_pre["tokens_per_frame"],
        video_cfg.video_attention_mask_mode,
    )
    tokens_video, tokens_action = mot_forward(
        video_params, action_params, video_cfg, action_cfg, video_pre, action_pre, mask
    )
    return (
        video_dit_post(video_params, video_cfg, tokens_video, video_pre["t"], video_pre["grid"]),
        action_dit_post(action_params, tokens_action),
    )


# --------------------------------------------------------------------------- #
# scheduler (schedulers/scheduler_continuous.py), float64
# --------------------------------------------------------------------------- #
Y_MIN = 0.6065306597
WEIGHT_NORM_CONST = 0.1894955536
WEIGHT_EPS = 1e-10


def phi(u, shift):
    u = np.asarray(u, dtype=np.float64)
    return shift * u / (1.0 + (shift - 1.0) * u)


def add_noise(x0, noise, timestep, num_train_timesteps=1000):
    x0 = np.asarray(x0, np.float64)
    noise = np.asarray(noise, np.float64)
    sigma = (np.asarray(timestep, np.float64) / num_train_timesteps).reshape(-1, *([1] * (x0.ndim - 1)))
    return (1 - sigma) * x0 + sigma * noise


def training_target(x0, noise):
    return np.asarray(noise, np.float64) - np.asarray(x0, np.float64)


def training_weight(timestep, num_train_timesteps=1000):
    t = np.asarray(timestep, np.float64)
    steps = float(num_train_timesteps)
    y = np.exp(-2.0 * ((t - steps / 2.0) / steps) ** 2)
    return (y - Y_MIN) / (WEIGHT_NORM_CONST + WEIGHT_EPS)


def build_inference_schedule(num_inference_steps, shift, num_train_timesteps=1000):
    u = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float64)
    sigma = phi(u, shift)
    return sigma[:-1] * num_train_timesteps, sigma[1:] - sigma[:-1]


def _precompute_weight_stats(num_train_timesteps=1000, shift=5.0):
    """The reference derives Y_MIN / WEIGHT_NORM_CONST from a 1000-point grid at import
    time; this reproduces that derivation so the hardcoded constants can be checked."""
    steps = num_train_timesteps
    u_grid = np.linspace(1.0, 0.0, steps + 1, dtype=np.float64)[:-1]
    t_grid = phi(u_grid, shift) * steps
    y_grid = np.exp(-2.0 * ((t_grid - steps / 2.0) / steps) ** 2)
    y_min = float(y_grid.min())
    return y_min, float((y_grid - y_min).mean())


# --------------------------------------------------------------------------- #
# training loss (fastwam.py L409-568) and the denoising loops (L725-1048)
# --------------------------------------------------------------------------- #
def fastwam_training_loss(
    params, video_cfg: RefConfig, action_cfg: ActionRefConfig,
    latents, action, context, context_mask, proprio,
    t_video, t_action, noise_video, noise_action,
    image_is_pad=None, action_is_pad=None,
    lambda_video=1.0, lambda_action=1.0, num_train_timesteps=1000, vae_temporal_factor=4,
):
    """training_loss (fastwam.py:448-568). Timesteps and noise are supplied, not drawn."""
    latents = np.asarray(latents, np.float64)
    action = np.asarray(action, np.float64)

    context, context_mask = append_proprio_to_context(
        _as_f64_params(params["proprio"]), context, context_mask, proprio[:, 0, :]
    )

    noisy_latents = add_noise(latents, noise_video, t_video, num_train_timesteps)
    target_video = training_target(latents, noise_video)
    noisy_latents[:, :, 0:1] = latents[:, :, 0:1]  # fuse_vae_embedding_in_latents

    noisy_action = add_noise(action, noise_action, t_action, num_train_timesteps)
    target_action = training_target(action, noise_action)

    pred_video, pred_action = fastwam_predict(
        params, video_cfg, action_cfg, noisy_latents, noisy_action,
        t_video, t_action, context, context_mask,
    )
    pred_video = pred_video[:, :, 1:]
    target_video = target_video[:, :, 1:]

    per_step_video = np.mean((pred_video - target_video) ** 2, axis=(1, 3, 4))
    if image_is_pad is None:
        video_per_sample = per_step_video.mean(axis=1)
    else:
        image_is_pad = np.asarray(image_is_pad, bool)
        tail = image_is_pad[:, 1:].reshape(image_is_pad.shape[0], -1, vae_temporal_factor)
        valid = (~tail.all(axis=2)).astype(np.float64)
        video_per_sample = (per_step_video * valid).sum(axis=1) / np.clip(valid.sum(axis=1), 1.0, None)
    loss_video = float((video_per_sample * training_weight(t_video, num_train_timesteps)).mean())

    per_step_action = np.mean((pred_action - target_action) ** 2, axis=2)
    if action_is_pad is None:
        action_per_sample = per_step_action.mean(axis=1)
    else:
        valid = (~np.asarray(action_is_pad, bool)).astype(np.float64)
        action_per_sample = (per_step_action * valid).sum(axis=1) / np.clip(valid.sum(axis=1), 1.0, None)
    loss_action = float((action_per_sample * training_weight(t_action, num_train_timesteps)).mean())

    return {
        "loss": lambda_video * loss_video + lambda_action * loss_action,
        "loss_video": lambda_video * loss_video,
        "loss_action": lambda_action * loss_action,
    }


def denoise_joint(
    params, video_cfg: RefConfig, action_cfg: ActionRefConfig,
    first_frame_latents, noise_video, noise_action, context, context_mask, proprio=None,
    num_inference_steps=10, shift=5.0, num_train_timesteps=1000,
):
    """infer_joint's loop (fastwam.py:854-891), minus the VAE decode."""
    if proprio is not None:
        context, context_mask = append_proprio_to_context(
            _as_f64_params(params["proprio"]), context, context_mask, proprio
        )
    first_frame_latents = np.asarray(first_frame_latents, np.float64)
    latents_video = np.asarray(noise_video, np.float64).copy()
    latents_video[:, :, 0:1] = first_frame_latents
    latents_action = np.asarray(noise_action, np.float64).copy()

    t_video_all, delta_video = build_inference_schedule(num_inference_steps, shift, num_train_timesteps)
    t_action_all, delta_action = build_inference_schedule(num_inference_steps, shift, num_train_timesteps)
    batch = latents_video.shape[0]

    for index in range(num_inference_steps):
        pred_video, pred_action = fastwam_predict(
            params, video_cfg, action_cfg, latents_video, latents_action,
            np.full((batch,), t_video_all[index]), np.full((batch,), t_action_all[index]),
            context, context_mask,
        )
        latents_video = latents_video + pred_video * delta_video[index]
        latents_action = latents_action + pred_action * delta_action[index]
        latents_video[:, :, 0:1] = first_frame_latents  # spec §9.5 item 25

    return {"action": latents_action, "latents_video": latents_video}


def denoise_action(
    params, video_cfg: RefConfig, action_cfg: ActionRefConfig,
    first_frame_latents, noise_action, context, context_mask, proprio=None,
    num_inference_steps=10, shift=5.0, num_train_timesteps=1000,
):
    """infer_action's loop (fastwam.py:1030-1044) with the video tower re-run each step
    instead of prefilled — numerically identical to the cache, see mot.py's docstring."""
    if proprio is not None:
        context, context_mask = append_proprio_to_context(
            _as_f64_params(params["proprio"]), context, context_mask, proprio
        )
    first_frame_latents = np.asarray(first_frame_latents, np.float64)
    latents_action = np.asarray(noise_action, np.float64).copy()
    timesteps, deltas = build_inference_schedule(num_inference_steps, shift, num_train_timesteps)
    batch = latents_action.shape[0]

    for index in range(num_inference_steps):
        _, pred_action = fastwam_predict(
            params, video_cfg, action_cfg, first_frame_latents, latents_action,
            np.zeros((batch,)), np.full((batch,), timesteps[index]), context, context_mask,
        )
        latents_action = latents_action + pred_action * deltas[index]
    return latents_action


# --------------------------------------------------------------------------- #
# ActionDiT initialization (scripts/preprocess_action_dit_backbone.py)
# --------------------------------------------------------------------------- #
def _interpolate_axis(tensor, axis, new_size):
    """F.interpolate(mode="linear", align_corners=True) applied along one axis.

    align_corners=True maps output index i to source position i*(L-1)/(N-1), so the two
    endpoints are preserved exactly. N == 1 is torch's degenerate case: the scale is 0
    and every output takes source index 0.
    """
    length = tensor.shape[axis]
    if length == new_size:
        return tensor
    moved = np.moveaxis(tensor, axis, -1)
    if new_size == 1:
        out = moved[..., :1]
    else:
        position = np.arange(new_size, dtype=np.float64) * ((length - 1) / (new_size - 1))
        low = np.floor(position).astype(np.int64)
        high = np.minimum(low + 1, length - 1)
        weight = position - low
        out = moved[..., low] * (1.0 - weight) + moved[..., high] * weight
    return np.moveaxis(out, -1, axis)


def resize_tensor_to_shape(src, target_shape):
    """_resize_tensor_to_shape (L61-96): sequential per-dimension 1-D linear interp.

    Ranks must already agree — the reference's unsqueeze/squeeze rank-fixing never
    fires for any ActionDiT key, so reproducing it would be untested code.
    """
    src = np.asarray(src, dtype=np.float64)
    if src.ndim != len(target_shape):
        raise ValueError(f"rank mismatch: src {src.shape} vs target {target_shape}")
    out = src
    for axis, new_size in enumerate(target_shape):
        out = _interpolate_axis(out, axis, new_size)
    if tuple(out.shape) != tuple(target_shape):
        raise ValueError(f"resize produced {out.shape}, expected {target_shape}")
    return out


def action_backbone_from_video(video_state, action_shapes, apply_alpha_scaling=True):
    """main()'s loop (L196-210), in TORCH layout and torch key names.

    For every action key: copy the same-named video tensor if the shapes already agree,
    otherwise resize it; and if the tensor is at least 2-D and its LAST dim changed,
    scale by alpha = sqrt(video_last / action_last) = sqrt(3) for the 3072->1024 pairs.

    `action_shapes` must already exclude action_encoder.* / head.* — the caller owns the
    skip list (`ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES`).
    """
    out = {}
    for key, target_shape in action_shapes.items():
        if key not in video_state:
            raise KeyError(f"`{key}` not found in the video expert state dict")
        src = np.asarray(video_state[key], dtype=np.float64)
        target_shape = tuple(target_shape)
        if src.shape == target_shape:
            out[key] = src
            continue
        value = resize_tensor_to_shape(src, target_shape)
        if apply_alpha_scaling and src.ndim >= 2 and src.shape[-1] != target_shape[-1]:
            value = value * (float(src.shape[-1]) / float(target_shape[-1])) ** 0.5
        out[key] = value
    return out
