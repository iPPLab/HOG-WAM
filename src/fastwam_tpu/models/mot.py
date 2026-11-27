"""Mixture-of-Transformers: 30 layers, two experts, one mixed self-attention each.

M2 implements this. It was the largest technical unknown in the project (plan risk #2):
no precedent existed for driving two nnx-scanned towers in lockstep. It works — see
"Scan result" below.

Written fresh; semantics from
    reference/FastWAM/src/fastwam/models/wan22/mot.py  (556 lines) — the authority.
Uses the block halves exposed by
    models/wan/transformer.py::WanDiTBlock.pre_attention / post_attention
    models/action_dit.py::ActionDiT (same block class, narrower)

Per layer (spec §2.4), expert_order = ["video", "action"]:
    1. Each expert independently: adaLN from its own t_mod, then q = norm_q(Wq x),
       k = norm_k(Wk x), v = Wv x, then RoPE on q and k. Weights are per-expert; the
       head geometry (24 x 128) is shared, which is what makes step 2 legal.
    2. q_cat = concat([q_video, q_action], axis=1) (same for k, v) -> ONE attention
       over S = Sv + Sa with the boolean [S, S] FastWAM mask.
    3. Split the output back and finish each expert separately: gated o-projection,
       then cross-attention (ungated), then gated FFN.

Shapes (RoboTwin, spec §4.5)
    Sv = 360, Sa = 32, S = 392. Prefill/inference: Sv = 120, S = 152.
    q/k/v after concat: [B, 392, 24, 128].

Scan result (the M2 opening spike, now `test_mot_parity.py::test_scan_matches_unrolled`)
    `nnx.scan` drives both towers with a two-element carry `(video_hidden, action_hidden)`
    and a TUPLE of the two stacked block modules as the scanned argument:

        nnx.scan(body, length=L, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))(
            (video_tokens, action_tokens), (self.video.blocks, self.action.blocks))

    `in_axes=0` applies elementwise across that tuple, so the two `nnx.vmap`-stacked
    block pytrees are sliced together, one layer of each per iteration. It needed no
    workaround: the first spike ran, and the only gap against the 30-unrolled path is
    fp32 reassociation (3.1e-7 relative on the video tower, 1.5e-7 on the action tower,
    at the parity widths) — the same order of magnitude M1 already measured for the
    video tower alone. The unrolled path is kept and tested as the reference, not as a
    fallback; `remat_policy` and `scan_layers` are independent knobs on both.

Cross-attention K/V are recomputed per layer, not cached, because both towers'
`context` changes every training step; the constant-context cache hook exists on the
block (`compute_context_kv`) and is M3's business.

Inference entry points (spec §2.4, used by inference/policy.py) — M3 implements these:
    prefill_video_cache(...)             -> VideoKVCache, k/v [30, B, Sv, 3072], post-RoPE
    forward_action_with_video_cache(...) -> per layer k_cat = concat([k_cached, k_action]),
                                            mask rows = attention_mask[Sv:, :Sv+Sa]
M2 proved the property they rest on before either existed: with this mask, running the
joint MoT on the first latent frame alone gives bit-comparable action tokens to running
it on the full video (test_mot_parity.py::test_action_output_equals_single_frame_prefill).
That equivalence *is* the prefill semantics; the cache is only an implementation of it,
and tests/test_inference_cache.py holds the two against each other numerically.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from ..config import ModelConfig
from ..copies import normalize_copy_lengths
from ..training import sharding
from .action_dit import ActionDiT, ActionPreState
from .wan.attention import dot_product_attention
from .wan.transformer import (
    REMAT_POLICIES,
    VideoPreState,
    WanDiTBlock,
    WanVideoDiT,
    video_self_attention_mask_numpy,
)

__all__ = [
    "MoTError",
    "VideoKVCache",
    "build_mot_attention_mask",
    "build_hog_mot_attention_mask",
    "MoT",
]


class MoTError(ValueError):
    """A MoT shape/config combination that cannot be a correct FastWAM call."""


class VideoKVCache(NamedTuple):
    """The prefill product: every layer's video self-attention K/V, RoPE already applied.

    k / v: [num_layers, B, Sv, num_heads*head_dim] — RoboTwin inference: [30, 1, 120, 3072],
    44 MB the pair in bf16.

    Upstream returns a *python list* of 30 `{"k": ..., "v": ...}` dicts (mot.py:257-337).
    Stacking on a leading layer axis instead is the one structural deviation in M3, and it
    is not cosmetic: `nnx.scan` needs the per-layer tensors as a single array to slice
    alongside the stacked block parameters, so under `scan_layers=True` a list could not be
    consumed at all. The contents are identical, layer i is `cache.k[i]`, and the unrolled
    path stacks its list at the end so both layouts produce the same object.

    Heads are MERGED here ([..., 3072]) rather than kept split ([..., 24, 128]) so the
    shape matches what the reference caches — the split is a free reshape on the way in.
    """

    k: jax.Array
    v: jax.Array


def build_mot_attention_mask(
    video_tokens: int, action_tokens: int, tokens_per_frame: int, copy_lengths=None
) -> jax.Array:
    """The one mask that is the whole paper (fastwam.py:386-407).

    Action queries attend ONLY to the first video latent frame plus themselves;
    video queries keep the first_frame_causal pattern among themselves and never
    see action tokens. That is what makes the first frame's K/V independent of the
    future video, hence the cache-and-skip-imagination inference path.

        mask = zeros(S, S)                                   # nothing attends anywhere
        mask[:Sv, :Sv] = build_video_to_video_mask(...)      # first_frame_causal
        mask[Sv:, Sv:] = True                                # action <-> action
        mask[Sv:, :tokens_per_frame] = True                  # action -> frame 0 only

    Note the video->action block stays False: the video tower is unaffected by the
    action tokens, which is the other half of the prefill argument.

    Args:
      copy_lengths: with `flow.tau_copies`, the action rows are several independent
        copies of the chunk (`copies.py`) and the single `action <-> action` block above
        becomes one block per copy on the diagonal, nothing off it. A copy that could
        attend a sibling would be reading the same chunk at another noise level, which
        inference never provides. None -- and equally a one-element tuple -- is the
        single full-length copy, byte-identical to the pre-copies mask.

    Returns: [Sv+Sa, Sv+Sa] bool, True = attend. RoboTwin: [392, 392].
    """
    if video_tokens <= 0 or action_tokens <= 0 or tokens_per_frame <= 0:
        raise MoTError(
            f"token counts must be positive, got Sv={video_tokens}, Sa={action_tokens}, "
            f"tokens_per_frame={tokens_per_frame}"
        )
    if video_tokens % tokens_per_frame != 0:
        raise MoTError(
            f"Sv={video_tokens} is not a whole number of frames of {tokens_per_frame} tokens"
        )
    lengths = normalize_copy_lengths(copy_lengths, action_tokens, "the action token count")
    total = video_tokens + action_tokens
    mask = np.zeros((total, total), dtype=bool)
    mask[:video_tokens, :video_tokens] = video_self_attention_mask_numpy(
        tokens_per_frame, video_tokens // tokens_per_frame
    )
    # One diagonal block per copy. A single copy makes this the whole [Sa, Sa] block, i.e.
    # exactly `mask[Sv:, Sv:] = True`.
    offset = video_tokens
    for length in lengths:
        mask[offset : offset + length, offset : offset + length] = True
        offset += length
    first_frame = min(tokens_per_frame, video_tokens)
    mask[video_tokens:, :first_frame] = True
    return jnp.asarray(mask)


def build_hog_mot_attention_mask(
    video_tokens: int,
    hog_tokens: int,
    action_tokens: int,
    tokens_per_frame: int,
    copy_lengths=None,
) -> jax.Array:
    """Removal-safe RGB/HOG/Action mask, in ``[video, hog, action]`` order.

    RGB and Action queries deliberately cannot see HOG keys.  Their forward graph is
    therefore the classic FastWAM graph and the HOG expert can be omitted at inference.
    HOG queries see both their own first-frame-causal stream and the equivalently causal
    RGB stream.  Consequently the auxiliary HOG loss backpropagates into the RGB tower,
    while never becoming a privileged input that deployment would later remove.
    """
    if video_tokens <= 0 or hog_tokens <= 0 or action_tokens <= 0 or tokens_per_frame <= 0:
        raise MoTError(
            f"token counts must be positive, got Sv={video_tokens}, Sh={hog_tokens}, "
            f"Sa={action_tokens}, tokens_per_frame={tokens_per_frame}"
        )
    if video_tokens != hog_tokens:
        raise MoTError(
            f"RGB and HOG towers must have identical token geometry, got {video_tokens} and "
            f"{hog_tokens}"
        )
    if video_tokens % tokens_per_frame:
        raise MoTError(
            f"Sv={video_tokens} is not a whole number of frames of {tokens_per_frame} tokens"
        )
    lengths = normalize_copy_lengths(copy_lengths, action_tokens, "the action token count")
    causal = video_self_attention_mask_numpy(tokens_per_frame, video_tokens // tokens_per_frame)
    action_start = video_tokens + hog_tokens
    total = action_start + action_tokens
    mask = np.zeros((total, total), dtype=bool)

    # Classic RGB path.  It sees neither HOG nor Action.
    mask[:video_tokens, :video_tokens] = causal
    # Auxiliary HOG path: same temporal visibility into RGB and into itself.
    mask[video_tokens:action_start, :video_tokens] = causal
    mask[video_tokens:action_start, video_tokens:action_start] = causal
    # Classic Action path: observed RGB frame plus one isolated diagonal block per tau copy.
    offset = action_start
    for length in lengths:
        mask[offset : offset + length, offset : offset + length] = True
        offset += length
    mask[action_start:, : min(tokens_per_frame, video_tokens)] = True
    return jnp.asarray(mask)


class MoT(nnx.Module):
    """The trainable model: video expert + action expert = 6.0207 B params.

    __call__(video_state, action_state, attention_mask)
        -> (video_tokens [B, Sv, 3072], action_tokens [B, Sa, 1024])

    where the two states are the `pre_dit` outputs of the respective towers. The MoT
    owns the towers (upstream's `nn.ModuleDict({"video": ..., "action": ...})` and its
    `expert_order`), so `model.dit` == the MoT == exactly the trainable parameters.
    """

    def __init__(
        self,
        *,
        video: WanVideoDiT,
        action: ActionDiT,
        hog: WanVideoDiT | None = None,
        remat_policy: str = "none",
    ):
        # The three build-time asserts from fastwam.py:140-146, kept here rather than
        # only in config.py because a MoT can also be assembled directly in a test.
        if action.num_heads != video.num_heads:
            raise MoTError(
                f"the experts must share num_heads for mixed attention: "
                f"{video.num_heads} vs {action.num_heads}"
            )
        if action.head_dim != video.head_dim:
            raise MoTError(
                f"the experts must share attn_head_dim: {video.head_dim} vs {action.head_dim}"
            )
        if action.num_layers != video.num_layers:
            raise MoTError(f"the experts must share num_layers: {video.num_layers} vs {action.num_layers}")
        if bool(video.scan_layers) != bool(action.scan_layers):
            raise MoTError(
                f"both experts must use the same layer layout, got video scan_layers="
                f"{video.scan_layers} and action scan_layers={action.scan_layers}"
            )
        if hog is not None:
            for attr in ("num_heads", "head_dim", "num_layers", "scan_layers"):
                if getattr(hog, attr) != getattr(video, attr):
                    raise MoTError(
                        f"the HOG and RGB experts must agree on {attr}: "
                        f"{getattr(hog, attr)} vs {getattr(video, attr)}"
                    )
        if remat_policy not in REMAT_POLICIES:
            raise MoTError(f"remat_policy must be one of {REMAT_POLICIES}, got {remat_policy!r}")

        self.video = video
        self.hog = hog
        self.action = action
        self.num_layers = video.num_layers
        self.num_heads = video.num_heads
        self.head_dim = video.head_dim
        self.scan_layers = bool(video.scan_layers)
        self.remat_policy = remat_policy

    @classmethod
    def from_config(
        cls,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        include_hog: bool = False,
        **overrides,
    ) -> "MoT":
        """Build both experts from `Config.model`. `overrides` go to BOTH towers.

        `remat_policy` is consumed by the MoT itself (it wraps the joint layer, which is
        the only thing that runs during training) and is not forwarded to the towers,
        whose own `run_blocks` are single-expert paths the MoT never calls.

        `config.mot_checkpoint_mixed_attn` is deliberately NOT read here. Upstream it
        wraps the mixed SDPA in `torch.utils.checkpoint`, i.e. it is a remat knob — and
        the shipped FastWAM model config sets it FALSE, so `remat_policy="none"` is
        the faithful default. Wiring the flag to a JAX remat policy is M5's job, along
        with the rest of the six-level dial in training/remat.py; doing it here would
        bake in a mapping that M5 then has to unpick.
        """
        remat_policy = overrides.pop("remat_policy", "none")
        return cls(
            video=WanVideoDiT.from_config(config.video_dit_config, rngs=rngs, **overrides),
            action=ActionDiT.from_config(config.action_dit_config, rngs=rngs, **overrides),
            hog=(
                WanVideoDiT.from_config(config.video_dit_config, rngs=rngs, **overrides)
                if include_hog
                else None
            ),
            remat_policy=remat_policy,
        )

    # ------------------------------------------------------------------ #
    # one mixed layer
    # ------------------------------------------------------------------ #
    def _layer(
        self,
        video_block: WanDiTBlock,
        action_block: WanDiTBlock,
        video_tokens: jax.Array,
        action_tokens: jax.Array,
        video_state: VideoPreState,
        action_state: ActionPreState,
        attention_mask: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """mot.py:472-554 for one layer, both experts.

        The `sharding.shard_activation` calls are no-ops except under the trainer's mesh
        with `tpu.shard_activations` on, and they change no arithmetic — they only forbid
        GSPMD the lowering that resharded these tensors onto the global batch (see
        `training/sharding.py`'s module docstring). They sit at the layer's boundaries —
        in, either side of the mixed attention, out — rather than around each individual
        matmul. Everything between them is elementwise or attention, so the constraint
        propagates through both block halves in both directions; adding pins inside
        `WanDiTBlock.post_attention` as well was tried and changed the optimized HLO not
        at all.
        """
        video_tokens = sharding.shard_activation(video_tokens)
        action_tokens = sharding.shard_activation(action_tokens)
        (video_q, video_k, video_v), video_gates = video_block.pre_attention(
            video_tokens, video_state.t_mod, video_state.cos, video_state.sin
        )
        (action_q, action_k, action_v), action_gates = action_block.pre_attention(
            action_tokens, action_state.t_mod, action_state.cos, action_state.sin
        )

        # The concat is along the SEQUENCE axis; both towers produce [B, S, 24, 128]
        # because they share the head geometry even though their hidden dims differ.
        query, key, value = (
            sharding.shard_activation(jnp.concatenate([v, a], axis=1))
            for v, a in ((video_q, action_q), (video_k, action_k), (video_v, action_v))
        )
        mixed = sharding.shard_activation(
            dot_product_attention(
                query, key, value, attention_mask, precision=video_block.attn1.precision
            )
        )

        split = video_tokens.shape[1]
        video_out = video_block.post_attention(
            video_tokens, mixed[:, :split], video_gates, video_state.context, video_state.context_mask
        )
        action_out = action_block.post_attention(
            action_tokens, mixed[:, split:], action_gates, action_state.context, action_state.context_mask
        )
        return sharding.shard_activation(video_out), sharding.shard_activation(action_out)

    def _layer_hog(
        self,
        video_block: WanDiTBlock,
        hog_block: WanDiTBlock,
        action_block: WanDiTBlock,
        video_tokens: jax.Array,
        hog_tokens: jax.Array,
        action_tokens: jax.Array,
        video_state: VideoPreState,
        hog_state: VideoPreState,
        action_state: ActionPreState,
        attention_mask: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """One removal-safe three-expert mixed-attention layer."""
        video_tokens = sharding.shard_activation(video_tokens)
        hog_tokens = sharding.shard_activation(hog_tokens)
        action_tokens = sharding.shard_activation(action_tokens)
        (video_q, video_k, video_v), video_gates = video_block.pre_attention(
            video_tokens, video_state.t_mod, video_state.cos, video_state.sin
        )
        (hog_q, hog_k, hog_v), hog_gates = hog_block.pre_attention(
            hog_tokens, hog_state.t_mod, hog_state.cos, hog_state.sin
        )
        (action_q, action_k, action_v), action_gates = action_block.pre_attention(
            action_tokens, action_state.t_mod, action_state.cos, action_state.sin
        )
        query, key, value = (
            sharding.shard_activation(jnp.concatenate([v, h, a], axis=1))
            for v, h, a in (
                (video_q, hog_q, action_q),
                (video_k, hog_k, action_k),
                (video_v, hog_v, action_v),
            )
        )
        mixed = sharding.shard_activation(
            dot_product_attention(
                query, key, value, attention_mask, precision=video_block.attn1.precision
            )
        )
        video_end = video_tokens.shape[1]
        hog_end = video_end + hog_tokens.shape[1]
        video_out = video_block.post_attention(
            video_tokens,
            mixed[:, :video_end],
            video_gates,
            video_state.context,
            video_state.context_mask,
        )
        hog_out = hog_block.post_attention(
            hog_tokens,
            mixed[:, video_end:hog_end],
            hog_gates,
            hog_state.context,
            hog_state.context_mask,
        )
        action_out = action_block.post_attention(
            action_tokens,
            mixed[:, hog_end:],
            action_gates,
            action_state.context,
            action_state.context_mask,
        )
        return tuple(sharding.shard_activation(x) for x in (video_out, hog_out, action_out))

    @staticmethod
    def _pin_one_state(state):
        """Batch-pin the per-sample tensors the layers read but never write.

        The two pre-states ride into the scan as closure constants, so an unpinned one is
        resharded on EVERY iteration. `t_mod` is the expensive one by a wide margin: it is
        [B, Sv, 6, hidden], i.e. six times the hidden state and, because `modulation()`
        upcasts, fp32 while everything around it is bf16 — 13.6 GB at the v4-32 microbatch
        against 1.1 GB for the hidden state itself. Left unpinned it cost 102 GB/microstep
        of `collective-permute` under pure FSDP, and under HSDP it became a 1.2 TB
        all-gather over the `data` axis, which is single-handedly what made HSDP look
        unusable before this.

        `cos`/`sin` are deliberately NOT pinned: they are [S, head_dim//2] RoPE tables with
        no batch axis at all, and `batch_spec` would put the sequence on the batch axes.
        `t` is not pinned either — post_dit consumes it, outside the layers.
        """
        return state._replace(
            t_mod=sharding.shard_activation(state.t_mod),
            context=sharding.shard_activation(state.context),
            context_mask=(
                None
                if state.context_mask is None
                else sharding.shard_activation(state.context_mask)
            ),
        )

    @classmethod
    def _pin_states(cls, video_state, action_state):
        return cls._pin_one_state(video_state), cls._pin_one_state(action_state)

    @classmethod
    def _pin_hog_states(cls, video_state, hog_state, action_state):
        return (
            cls._pin_one_state(video_state),
            cls._pin_one_state(hog_state),
            cls._pin_one_state(action_state),
        )

    def _layer_body(self):
        def body(video_block, action_block, video_tokens, action_tokens, video_state, action_state, mask):
            return self._layer(
                video_block, action_block, video_tokens, action_tokens, video_state, action_state, mask
            )

        if self.remat_policy == "full":
            return nnx.remat(body, prevent_cse=not self.scan_layers, policy=None)
        return body

    def _layer_hog_body(self):
        def body(
            video_block,
            hog_block,
            action_block,
            video_tokens,
            hog_tokens,
            action_tokens,
            video_state,
            hog_state,
            action_state,
            mask,
        ):
            return self._layer_hog(
                video_block,
                hog_block,
                action_block,
                video_tokens,
                hog_tokens,
                action_tokens,
                video_state,
                hog_state,
                action_state,
                mask,
            )

        if self.remat_policy == "full":
            return nnx.remat(body, prevent_cse=not self.scan_layers, policy=None)
        return body

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        video_state: VideoPreState,
        action_state: ActionPreState,
        attention_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        video_tokens = video_state.tokens
        action_tokens = action_state.tokens
        if video_tokens.shape[0] != action_tokens.shape[0]:
            raise MoTError(
                f"the two experts must share a batch: {video_tokens.shape[0]} vs {action_tokens.shape[0]}"
            )
        total = video_tokens.shape[1] + action_tokens.shape[1]
        if attention_mask is not None:
            if attention_mask.ndim != 2 or attention_mask.shape != (total, total):
                raise MoTError(
                    f"attention_mask must be [{total}, {total}] (Sv+Sa square), "
                    f"got {attention_mask.shape}"
                )

        video_state, action_state = self._pin_states(video_state, action_state)
        body = self._layer_body()
        if not self.scan_layers:
            for video_block, action_block in zip(self.video.blocks, self.action.blocks):
                video_tokens, action_tokens = body(
                    video_block, action_block, video_tokens, action_tokens,
                    video_state, action_state, attention_mask,
                )
            return video_tokens, action_tokens

        def scan_fn(carry, blocks):
            video_block, action_block = blocks
            tokens = body(
                video_block, action_block, carry[0], carry[1], video_state, action_state, attention_mask
            )
            return tokens, None

        carry, _ = nnx.scan(
            scan_fn, length=self.num_layers, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )((video_tokens, action_tokens), (self.video.blocks, self.action.blocks))
        return carry

    def forward_states(
        self,
        video_state: VideoPreState,
        action_state: ActionPreState,
        attention_mask: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """pre_dit states in, DECODED outputs out: (pred_video, pred_action).

        The `post_dit` of each tower applied to the MoT output — i.e. everything
        `FastWAM._predict_joint_noise` does after `self.mot(...)`.
        """
        video_tokens, action_tokens = self(video_state, action_state, attention_mask)
        return (
            self.video.post_dit(video_tokens, video_state.t, video_state.grid),
            self.action.post_dit(action_tokens),
        )

    def forward_hog_states(
        self,
        video_state: VideoPreState,
        hog_state: VideoPreState,
        action_state: ActionPreState,
        attention_mask: jax.Array | None = None,
        *,
        predict_video: bool = True,
    ) -> tuple[jax.Array | None, jax.Array, jax.Array]:
        """Training-only three-tower forward; inference continues through ``forward_states``."""
        if self.hog is None:
            raise MoTError("forward_hog_states requires a model built with include_hog=True")
        video_tokens, hog_tokens, action_tokens = (
            video_state.tokens,
            hog_state.tokens,
            action_state.tokens,
        )
        batches = {video_tokens.shape[0], hog_tokens.shape[0], action_tokens.shape[0]}
        if len(batches) != 1:
            raise MoTError(
                f"all three experts must share a batch, got "
                f"{video_tokens.shape[0]}, {hog_tokens.shape[0]}, {action_tokens.shape[0]}"
            )
        total = video_tokens.shape[1] + hog_tokens.shape[1] + action_tokens.shape[1]
        if attention_mask is not None and (
            attention_mask.ndim != 2 or attention_mask.shape != (total, total)
        ):
            raise MoTError(
                f"HOG attention_mask must be [{total}, {total}] (Sv+Sh+Sa square), "
                f"got {attention_mask.shape}"
            )

        video_state, hog_state, action_state = self._pin_hog_states(
            video_state, hog_state, action_state
        )
        body = self._layer_hog_body()
        if not self.scan_layers:
            for video_block, hog_block, action_block in zip(
                self.video.blocks, self.hog.blocks, self.action.blocks
            ):
                video_tokens, hog_tokens, action_tokens = body(
                    video_block,
                    hog_block,
                    action_block,
                    video_tokens,
                    hog_tokens,
                    action_tokens,
                    video_state,
                    hog_state,
                    action_state,
                    attention_mask,
                )
        else:
            def scan_fn(carry, blocks):
                video_block, hog_block, action_block = blocks
                tokens = body(
                    video_block,
                    hog_block,
                    action_block,
                    carry[0],
                    carry[1],
                    carry[2],
                    video_state,
                    hog_state,
                    action_state,
                    attention_mask,
                )
                return tokens, None

            (video_tokens, hog_tokens, action_tokens), _ = nnx.scan(
                scan_fn,
                length=self.num_layers,
                in_axes=(nnx.Carry, 0),
                out_axes=(nnx.Carry, 0),
            )(
                (video_tokens, hog_tokens, action_tokens),
                (self.video.blocks, self.hog.blocks, self.action.blocks),
            )

        return (
            self.video.post_dit(video_tokens, video_state.t, video_state.grid)
            if predict_video else None,
            self.hog.post_dit(hog_tokens, hog_state.t, hog_state.grid),
            self.action.post_dit(action_tokens),
        )

    # ------------------------------------------------------------------ #
    # M3: the KV-cache inference path
    #
    # Neither method reads `remat_policy`. Both run under `jax.jit` with no gradient
    # anywhere, so rematerialisation would only buy back memory nobody is spending —
    # the cache is 44 MB and the activations are one 120-token frame.
    # ------------------------------------------------------------------ #
    def _merge_heads(self, x: jax.Array) -> jax.Array:
        """[B, S, num_heads, head_dim] -> [B, S, num_heads*head_dim]."""
        return x.reshape(*x.shape[:-2], self.num_heads * self.head_dim)

    def _split_heads(self, x: jax.Array) -> jax.Array:
        """[B, S, num_heads*head_dim] -> [B, S, num_heads, head_dim]."""
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

    def prefill_video_cache(
        self,
        video_state: VideoPreState,
        video_attention_mask: jax.Array | None = None,
    ) -> VideoKVCache:
        """Run the video tower ALONE, once, keeping each layer's K/V (mot.py:257-337).

        Called with the single observed latent frame at t=0, so Sv = tokens_per_frame
        (120 for RoboTwin) and the returned cache is everything the action tokens will
        ever be allowed to see. Legal because of the joint mask: the video branch never
        attends to action tokens, so its K/V do not depend on the denoising step, and
        the action rows only reach latent frame 0, whose K/V do not depend on the future
        video either. Both halves are pinned as tests in test_mot_parity.py
        (`test_video_output_is_invariant_to_the_action_tokens`,
        `test_action_output_equals_single_frame_prefill`).

        Args:
            video_state: `WanVideoDiT.pre_dit(first_frame_latents, timestep=0, ...)`.
            video_attention_mask: the `[:Sv, :Sv]` block of the joint mask, i.e.
                `build_mot_attention_mask(...)[:Sv, :Sv]`. At Sv = tokens_per_frame that
                block is all-True; passing None means the same thing. It is an argument
                rather than something derived here because the caller already built the
                joint mask and slicing it is the only way to guarantee the two agree.

        Returns: :class:`VideoKVCache`, k/v each [num_layers, B, Sv, 3072].
        """
        tokens = video_state.tokens
        video_tokens = tokens.shape[1]
        if video_attention_mask is not None:
            if video_attention_mask.ndim != 2 or video_attention_mask.shape != (
                video_tokens,
                video_tokens,
            ):
                raise MoTError(
                    f"video_attention_mask must be the square [{video_tokens}, {video_tokens}] "
                    f"video block of the joint mask, got {video_attention_mask.shape}"
                )

        def body(block: WanDiTBlock, x: jax.Array):
            (query, key, value), gates = block.pre_attention(
                x, video_state.t_mod, video_state.cos, video_state.sin
            )
            mixed = dot_product_attention(
                query, key, value, video_attention_mask, precision=block.attn1.precision
            )
            updated = block.post_attention(
                x, mixed, gates, video_state.context, video_state.context_mask
            )
            # The cached K/V are the ones this layer just attended over: post-RMSNorm,
            # post-RoPE, pre head-merge. Caching anything earlier would make the action
            # pass redo work; anything later does not exist.
            return updated, (self._merge_heads(key), self._merge_heads(value))

        if not self.scan_layers:
            keys, values = [], []
            for block in self.video.blocks:
                tokens, (key, value) = body(block, tokens)
                keys.append(key)
                values.append(value)
            return VideoKVCache(k=jnp.stack(keys, axis=0), v=jnp.stack(values, axis=0))

        def scan_fn(carry, block):
            return body(block, carry)

        _, (keys, values) = nnx.scan(
            scan_fn, length=self.num_layers, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )(tokens, self.video.blocks)
        return VideoKVCache(k=keys, v=values)

    def forward_action_with_video_cache(
        self,
        action_state: ActionPreState,
        cache: VideoKVCache,
        attention_mask: jax.Array,
        video_seq_len: int,
    ) -> jax.Array:
        """The action tower against a prefilled video cache (mot.py:339-445).

        Per layer the action expert still recomputes its own q/k/v — they move with the
        denoising step — and then attends over `concat([k_cached, k_action])`, i.e. the
        SAME key set the joint forward would have built, with the video half read out of
        the cache instead of recomputed.

        Args:
            action_state: `ActionDiT.pre_dit(noisy_action, t, ...)` for this step.
            cache: the prefill product.
            attention_mask: the FULL joint `[Sv+Sa, Sv+Sa]` mask. The action query rows
                `attention_mask[Sv:, :Sv+Sa]` are sliced out here rather than passed in,
                which is what `test_cached_action_mask_rows_are_the_joint_mask_rows` pins.
            video_seq_len: Sv, the prefix length the cache covers.

        Returns: action tokens after all layers, [B, Sa, action hidden] — feed to
        `ActionDiT.post_dit` to get the predicted velocity.
        """
        tokens = action_state.tokens
        action_seq_len = tokens.shape[1]
        total = int(video_seq_len) + action_seq_len
        if cache.k.shape[0] != self.num_layers or cache.v.shape[0] != self.num_layers:
            raise MoTError(
                f"the cache must carry {self.num_layers} layers, got k={cache.k.shape[0]} "
                f"v={cache.v.shape[0]}"
            )
        if cache.k.shape[2] != video_seq_len or cache.v.shape[2] != video_seq_len:
            raise MoTError(
                f"cached K/V cover {cache.k.shape[2]}/{cache.v.shape[2]} video tokens but "
                f"video_seq_len={video_seq_len}"
            )
        if cache.k.shape[1] != tokens.shape[0]:
            raise MoTError(
                f"the cache was prefilled for batch {cache.k.shape[1]} but the action tokens "
                f"are batch {tokens.shape[0]}"
            )
        if attention_mask.ndim != 2 or attention_mask.shape != (total, total):
            raise MoTError(
                f"attention_mask must be the full joint [{total}, {total}] mask, got "
                f"{attention_mask.shape}"
            )
        action_rows = attention_mask[video_seq_len:total, :total]

        def body(block: WanDiTBlock, x: jax.Array, cached_k: jax.Array, cached_v: jax.Array):
            (query, key, value), gates = block.pre_attention(
                x, action_state.t_mod, action_state.cos, action_state.sin
            )
            key = jnp.concatenate([self._split_heads(cached_k), key], axis=1)
            value = jnp.concatenate([self._split_heads(cached_v), value], axis=1)
            mixed = dot_product_attention(
                query, key, value, action_rows, precision=block.attn1.precision
            )
            return block.post_attention(
                x, mixed, gates, action_state.context, action_state.context_mask
            )

        if not self.scan_layers:
            for index, block in enumerate(self.action.blocks):
                tokens = body(block, tokens, cache.k[index], cache.v[index])
            return tokens

        def scan_fn(carry, layer):
            block, cached_k, cached_v = layer
            return body(block, carry, cached_k, cached_v), None

        carry, _ = nnx.scan(
            scan_fn, length=self.num_layers, in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0)
        )(tokens, (self.action.blocks, cache.k, cache.v))
        return carry
