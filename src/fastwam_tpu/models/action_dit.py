"""The action expert: a narrow DiT (d=1024) with video-width attention (24x128=3072).

M2 implements this.

Written fresh, reusing
    models/wan/attention.py   (the query_dim != inner_dim path exists for this module)
    models/wan/embeddings.py  (same TimestepEmbedding / TextEmbedding, hidden=1024)
    models/wan/rope.py        (precompute_freqs_1d over the action-step axis)
    models/wan/transformer.py (the SAME WanDiTBlock class, just narrower)
    Semantics from reference/FastWAM/src/fastwam/models/wan22/action_dit.py

Structure (spec §2.3)
    action_encoder  = Linear(action_dim=14 -> 1024)
    text_embedding  = Linear(4096, 1024) -> GELU(tanh) -> Linear(1024, 1024)
    time_embedding  = Linear(256, 1024) -> SiLU -> Linear(1024, 1024)
    time_projection = SiLU -> Linear(1024, 6144)
    blocks          = 30 x the SAME DiTBlock class as the video tower, so q/k/v are
                      Linear(1024 -> 3072), o is Linear(3072 -> 1024), and norm_q /
                      norm_k are RMSNorm over 3072.
    head            = Linear(1024, action_dim) -- a plain linear, NO adaLN.
                      (`ActionHead` in the reference file is dead code.)
    freqs           = precompute_freqs_cis(128, end=1024), plain 1D RoPE.

t_mod here is [B, 6, 1024] (ndim 3) -> modulation is shared across tokens, unlike
the video tower's per-token [B, S, 6, 3072]. `WanDiTBlock.modulation` already branches
on that rank; M1 built and parity-tested the branch for exactly this.

NAMING follows the video tower's (maxdiffusion's), not FastWAM's, for the reason
spelled out in transformer.py: `condition_embedder.{time_embedder,time_proj,
text_embedder}` instead of `time_embedding.0/2` + `time_projection.1` +
`text_embedding.0/2`, and `proj_out` instead of `head`. Two consequences worth stating:
  * the ActionDiT tree is then key-for-key the video tree minus patch_embedding /
    scale_shift_table / norm_out, plus action_encoder — which is exactly the relation
    the interpolation init (spec §5.1) walks, so it needs no rename table either;
  * `proj_out` collides by name with the video tower's head (different shapes). It is
    in ACTION_ONLY_KEYS and therefore skipped by the init, and
    `tools/convert_weights.py::build_action_dit_init` raises unless the skip set is
    exactly the four keys below, so the collision cannot silently interpolate garbage.

Initialization is NOT random: the backbone is produced by interpolating the Wan2.2
video DiT down from 3072 to 1024 (tools/convert_weights.py, spec §5.1). Only
action_encoder and proj_out are randomly initialized.

Shapes (RoboTwin): action [B,32,14] -> tokens [B,32,1024] -> output [B,32,14].
"""

import os
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..config import ActionDiTConfig
from ..copies import normalize_copy_lengths
from .wan.embeddings import ConditionEmbedder
from .wan.rope import precompute_freqs_1d
from .wan.transformer import (
    REMAT_POLICIES,
    WanDiTBlock,
    WanVideoDiTConfigError,
    _restore_unfinalized,
)
from .wan.tree_keys import restore_tree_key_types, stringify_tree_keys

__all__ = ["ACTION_ONLY_KEYS", "ActionPreState", "ActionDiT", "load_action_dit"]

# The two modules ActionDiT does NOT inherit from the video expert
# (`ActionDiT.ACTION_BACKBONE_SKIP_PREFIXES` = ("action_encoder.", "head.") upstream,
# translated into this repo's names). Frozen as an explicit key set rather than a
# prefix test so a renamed module fails loudly instead of quietly widening the skip.
ACTION_ONLY_KEYS = frozenset(
    {"action_encoder.kernel", "action_encoder.bias", "proj_out.kernel", "proj_out.bias"}
)

# ActionDiT's RoPE cache length (`precompute_freqs_cis(attn_head_dim, end=1024)`).
ROPE_CACHE_LEN = 1024


class ActionPreState(NamedTuple):
    """Everything the blocks need, mirroring transformer.py's VideoPreState.

    `t` is carried for symmetry only: ActionDiT.post_dit is a plain Linear and ignores
    it (`post_dit(self, tokens, pre_state): return self.head(tokens)`).
    """

    tokens: jax.Array  # [B, Sa, hidden]
    t: jax.Array  # [B, hidden]
    t_mod: jax.Array  # [B, 6, hidden]   -- rank 3: shared across tokens
    context: jax.Array  # [B, L, hidden]
    context_mask: jax.Array | None  # [B, Sa, L]
    cos: jax.Array  # [Sa, head_dim//2]
    sin: jax.Array  # [Sa, head_dim//2]


class ActionDiT(nnx.Module):
    """The action expert.

    pre_dit(action, timestep, context, context_mask)
        action:       [B, 32, 14]  (noised, normalized)
        timestep:     [B] float
        context:      [B, 129, 4096]
        context_mask: [B, 129] bool
        ->
        ActionPreState(tokens [B,32,1024], t [B,1024], t_mod [B,6,1024],
                       context [B,129,1024], context_mask [B,32,129], cos/sin [32,64])

    post_dit(tokens) -> [B, 32, 14]   (plain Linear head)
    """

    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        action_dim: int,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        freq_dim: int,
        text_dim: int,
        eps: float = 1e-6,
        scan_layers: bool = False,
        remat_policy: str = "none",
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | None = None,
        freqs_dtype=np.float64,
    ):
        if head_dim % 2 != 0:
            raise WanVideoDiTConfigError(f"attn_head_dim must be even for RoPE, got {head_dim}")
        if num_heads <= 0 or head_dim <= 0:
            raise WanVideoDiTConfigError(
                f"num_heads/head_dim must be positive, got {num_heads}, {head_dim}"
            )
        if remat_policy not in REMAT_POLICIES:
            raise WanVideoDiTConfigError(
                f"remat_policy must be one of {REMAT_POLICIES}, got {remat_policy!r}"
            )

        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.eps = eps
        self.scan_layers = scan_layers
        self.remat_policy = remat_policy
        self.dtype = dtype
        self.freqs_dtype = freqs_dtype

        self.action_encoder = nnx.Linear(
            rngs=rngs,
            in_features=action_dim,
            out_features=hidden_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )
        self.condition_embedder = ConditionEmbedder(
            rngs=rngs,
            hidden_dim=hidden_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            param_dtype=param_dtype,
            precision=precision,
            freqs_dtype=freqs_dtype,
        )

        # query_dim=hidden_dim (1024) but the attention runs at num_heads*head_dim
        # (3072): q/k/v are 1024->3072, o is 3072->1024, norm_q/norm_k are over 3072.
        block_kwargs = dict(
            dim=hidden_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            eps=eps,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
        )
        if scan_layers:

            @nnx.split_rngs(splits=num_layers)
            @nnx.vmap(in_axes=0, out_axes=0, transform_metadata={nnx.PARTITION_NAME: "layers_per_stage"})
            def init_block(block_rngs: nnx.Rngs) -> WanDiTBlock:
                return WanDiTBlock(rngs=block_rngs, **block_kwargs)

            self.blocks = init_block(rngs)
        else:
            self.blocks = nnx.data([WanDiTBlock(rngs=rngs, **block_kwargs) for _ in range(num_layers)])

        # FastWAM calls this `head`; renamed for symmetry with the video tower, and
        # excluded from the interpolation init (see the module docstring).
        self.proj_out = nnx.Linear(
            rngs=rngs,
            in_features=hidden_dim,
            out_features=action_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            kernel_init=nnx.initializers.xavier_uniform(),
        )

    @classmethod
    def from_config(cls, config: ActionDiTConfig, *, rngs: nnx.Rngs, **overrides) -> "ActionDiT":
        kwargs = dict(
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            ffn_dim=config.ffn_dim,
            num_heads=config.num_heads,
            head_dim=config.attn_head_dim,
            num_layers=config.num_layers,
            freq_dim=config.freq_dim,
            text_dim=config.text_dim,
            eps=config.eps,
        )
        kwargs.update(overrides)
        return cls(rngs=rngs, **kwargs)

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def rope_tables(self, seq_len: int, copy_lengths=None) -> tuple[jax.Array, jax.Array]:
        """1D RoPE over the action-step axis, sliced from the end=1024 cache.

        With `flow.tau_copies` the rows are several copies of the same chunk
        (`copies.py`), and positions RESTART at 0 for each: `[0..L0-1 | 0..L1-1 | ...]`.
        Letting them run on would place copy 1 a whole chunk into the future, and would
        also change its distance to the video keys -- which does NOT cancel, because
        those keys keep their own positions. Legal shapes, no error, a different model.
        None (or one copy) returns the plain `0..seq_len-1` table unchanged.
        """
        lengths = normalize_copy_lengths(copy_lengths, seq_len, "the action token count")
        longest = max(lengths)
        if longest > ROPE_CACHE_LEN:
            raise WanVideoDiTConfigError(
                f"action copy length {longest} exceeds the RoPE cache {ROPE_CACHE_LEN}"
            )
        cos, sin = precompute_freqs_1d(self.head_dim, ROPE_CACHE_LEN, freqs_dtype=self.freqs_dtype)
        if len(lengths) == 1:
            return cos[:seq_len], sin[:seq_len]
        return (
            jnp.concatenate([cos[:length] for length in lengths], axis=0),
            jnp.concatenate([sin[:length] for length in lengths], axis=0),
        )

    def pre_dit(
        self,
        action: jax.Array,
        timestep: jax.Array,
        context: jax.Array,
        context_mask: jax.Array | None = None,
        copy_lengths=None,
    ) -> ActionPreState:
        """`timestep` is [B] -- one flow time for the whole chunk, the released recipe --
        or [B, T] per row, which is how `flow.tau_copies` gives each copy its own tau.

        The per-row path is not new machinery: `embed_time` already accepts [B, seq] (the
        video expert's `seperated_timestep` path runs on it) and `WanDiTBlock.modulation`
        already dispatches on a rank-4 `t_mod`. The action head is a plain Linear with no
        adaLN, so nothing downstream of the blocks has to learn about it either.
        """
        if action.ndim != 3:
            raise WanVideoDiTConfigError(f"action must be [B, T, action_dim], got {action.shape}")
        if action.shape[2] != self.action_dim:
            raise WanVideoDiTConfigError(
                f"action last dim must be {self.action_dim}, got {action.shape[2]}"
            )
        if timestep.ndim not in (1, 2) or timestep.shape[0] != action.shape[0]:
            raise WanVideoDiTConfigError(
                f"timestep must be [B] or [B, T] matching the action batch {action.shape[0]}, "
                f"got {timestep.shape}"
            )
        if timestep.ndim == 2 and timestep.shape[1] != action.shape[1]:
            raise WanVideoDiTConfigError(
                f"per-row timestep has {timestep.shape[1]} rows but the action stream has "
                f"{action.shape[1]}"
            )
        if context.ndim != 3 or context.shape[0] != action.shape[0]:
            raise WanVideoDiTConfigError(
                f"context must be [B, L, D] matching the action batch, got {context.shape}"
            )

        seq_len = action.shape[1]
        tokens = self.action_encoder(action)
        t, t_mod = self.condition_embedder.embed_time(timestep)
        context_emb = self.condition_embedder.embed_text(context)

        expanded_mask = None
        if context_mask is not None:
            if context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
                raise WanVideoDiTConfigError(
                    f"context_mask must be [B, L] matching context {context.shape[:2]}, "
                    f"got {context_mask.shape}"
                )
            # [B, L] -> [B, Sa, L] (action_dit.py:285).
            expanded_mask = jnp.broadcast_to(
                context_mask[:, None, :], (context_mask.shape[0], seq_len, context_mask.shape[1])
            )

        cos, sin = self.rope_tables(seq_len, copy_lengths)
        return ActionPreState(
            tokens=tokens,
            t=t,
            t_mod=t_mod,
            context=context_emb.astype(tokens.dtype),
            context_mask=expanded_mask,
            cos=cos,
            sin=sin,
        )

    def _block_body(self):
        def body(block: WanDiTBlock, x, context, t_mod, cos, sin, self_attn_mask, context_mask):
            return block(x, context, t_mod, cos, sin, self_attn_mask, context_mask)

        if self.remat_policy == "full":
            return nnx.remat(body, prevent_cse=not self.scan_layers, policy=None)
        return body

    def run_blocks(
        self,
        tokens: jax.Array,
        context: jax.Array,
        t_mod: jax.Array,
        cos: jax.Array,
        sin: jax.Array,
        self_attn_mask: jax.Array | None = None,
        context_mask: jax.Array | None = None,
    ) -> jax.Array:
        """The action tower run ALONE (all-to-all self attention among action tokens).

        FastWAM never runs it this way — the MoT always drives the two towers together —
        but `ActionDiT.forward` upstream does exactly this, and having it here is what
        lets the parity tests pin the tower on its own before the MoT is in the picture.
        """
        body = self._block_body()
        args = (context, t_mod, cos, sin, self_attn_mask, context_mask)
        if not self.scan_layers:
            for block in self.blocks:
                tokens = body(block, tokens, *args)
            return tokens

        def scan_fn(carry, block):
            return body(block, carry, *args), None

        carry, _ = nnx.scan(
            scan_fn, length=self.num_layers, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )(tokens, self.blocks)
        return carry

    def post_dit(self, tokens: jax.Array) -> jax.Array:
        """Plain Linear(1024 -> action_dim). No adaLN, no norm (action_dit.py:301-302)."""
        return self.proj_out(tokens)

    def __call__(
        self,
        action: jax.Array,
        timestep: jax.Array,
        context: jax.Array,
        context_mask: jax.Array | None = None,
        self_attn_mask: jax.Array | None = None,
    ) -> jax.Array:
        state = self.pre_dit(action, timestep, context, context_mask)
        tokens = self.run_blocks(
            state.tokens,
            state.context,
            state.t_mod,
            state.cos,
            state.sin,
            self_attn_mask,
            state.context_mask,
        )
        return self.post_dit(tokens)


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #
def load_action_dit(
    ckpt_dir: str,
    config: ActionDiTConfig,
    *,
    rngs: nnx.Rngs | None = None,
    **overrides,
) -> ActionDiT:
    """Restore the action expert from an Orbax dir (local or ``gs://``). Any drift raises.

    Same contract as ``transformer.py::load_video_dit``: the on-disk tree is UNSCANNED,
    and ``scan_layers=True`` stacks the block parameters after the restore. This is the
    loader for the interpolation-init tree written by
    ``tools/convert_weights.py build-action-init``.
    """
    import orbax.checkpoint as ocp

    rngs = rngs if rngs is not None else nnx.Rngs(0)
    scan_layers = bool(overrides.get("scan_layers", False))
    path = ckpt_dir if ckpt_dir.startswith("gs://") else os.path.abspath(os.path.expanduser(ckpt_dir))

    flat_overrides = {**overrides, "scan_layers": False}
    unscanned = nnx.eval_shape(lambda: ActionDiT.from_config(config, rngs=nnx.Rngs(0), **flat_overrides))
    template = nnx.to_pure_dict(nnx.state(unscanned, nnx.Param))

    checkpointer = ocp.StandardCheckpointer()
    errors = {}
    restored = None
    for style, target in (
        ("flax 0.10 string keys", stringify_tree_keys(template)),
        ("flax 0.12 int keys", template),
    ):
        # `_restore_unfinalized` is the fallback for trees written before orbax started
        # writing `commit_success.txt`; see transformer.py for why the shared checkpoints
        # are read around that marker rather than rewritten.
        for how, restore in (
            ("standard", lambda t=target: checkpointer.restore(path, target=t)),
            ("unfinalized", lambda t=target: _restore_unfinalized(path, t)),
        ):
            try:
                restored = restore_tree_key_types(restore(), template)
            except Exception as exc:  # noqa: BLE001 - every cause is reported below
                errors[f"{style} / {how}"] = exc
                continue
            break
        if restored is not None:
            break
    if restored is None:
        raise WanVideoDiTConfigError(
            f"restoring the action DiT from {path} failed under every key style and restore path.\n"
            + "\n".join(f"  {style}: {type(exc).__name__}: {exc}" for style, exc in errors.items())
            + "\nThe checkpoint tree must match this module exactly - no partial restores."
        ) from next(iter(errors.values()))

    if scan_layers:
        per_layer = [restored["blocks"][index] for index in range(config.num_layers)]
        restored = {
            **{key: value for key, value in restored.items() if key != "blocks"},
            "blocks": jax.tree.map(lambda *layers: jnp.stack(layers, axis=0), *per_layer),
        }

    model = ActionDiT.from_config(config, rngs=rngs, **overrides)
    graphdef, state, rest = nnx.split(model, nnx.Param, ...)
    nnx.replace_by_pure_dict(state, restored)
    return nnx.merge(graphdef, state, rest)
