"""FastWAM (uncond): assembles VAE latents + text + proprio into the MoT, and the loss.

M2 implements training_loss and the joint denoising loop; M3 adds `infer_action`, the
KV-cached deployment path, and wraps it in inference/policy.py.

Written fresh; semantics from
    reference/FastWAM/src/fastwam/models/wan22/fastwam.py  (class FastWAM)
        build_inputs L277-383, _build_mot_attention_mask L385-407,
        _compute_video_loss_per_sample L409-446, training_loss L448-568,
        _predict_joint_noise L570-632, infer_joint L725-903
The joint (FastWAMJoint) and IDM (FastWAMIDM) variants are deliberately NOT ported:
they are the "does it need imagination?" ablations, and dropping them halves the
eval and validation surface. The model code they would share is already here.

Components
    mot             : models/mot.py           (trainable, 6.0207 B)
    proprio_encoder : Linear(14 -> 4096)      (trainable; 61,440 params for RoboTwin)
    vae             : models/wan/vae.py       (frozen; offline in our pipeline)
    text encoder    : not present — UMT5 embeddings are precomputed (spec §5.2)

Training step (spec §1.2, fastwam.py:448-568)
    1. latents [B,48,3,24,20]  (we read them from the offline cache instead of
       running the VAE inside the step; the only deliberate deviation, spec §9.1.4)
    2. sample two INDEPENDENT flow-matching timesteps t_video, t_action per sample
    3. noise both; then overwrite latent frame 0 with the clean first-frame latent
    4. pre_dit on both experts; append the proprio token to the text context (-> 129)
    5. one MoT pass with build_mot_attention_mask
    6. loss = 1.0 * loss_video + 1.0 * loss_action

Loss (spec §3.4) — exactly, including the parts that look redundant
    video:  drop latent frame 0 from BOTH pred and target, per-token MSE in fp32
            averaged over (C,H,W) -> [B, T_l-1], masked by image_is_pad folded 4:1,
            per-sample mean over valid, then * training_weight(t_video), then mean.
    action: per-step MSE in fp32 averaged over the action dim -> [B, 32], masked by
            action_is_pad, same weighting.
    No CFG, no EMA, no conditioning dropout. Do not add "standard" tricks.

Context convention (spec §2.8) — reproduce verbatim
    prompt_emb[i, seq_len_i:] = 0 and then the mask is declared ALL-ONES, so cross
    attention attends over all 128 positions with the padded ones being exact zeros.
    That zeroing happens where the embeddings are produced (offline, M4); this module
    only appends the proprio token, whose mask entry is likewise True.

Two things the reference does that are deliberately NOT reproduced
    * `loss_dict` calls `.item()` on every entry, i.e. a host sync per optimizer step
      (spec §9.5 item 26). The metrics here stay as device scalars.
    * `build_inputs` runs the VAE inside the step, one sample at a time. Latents arrive
      precomputed; `build_inputs` here only validates and assembles the context.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx

from ..config import Config, ModelConfig
from ..copies import resolve_copy_lengths
from ..schedulers import flow_match
from .action_dit import ActionPreState
from .mot import MoT, build_hog_mot_attention_mask, build_mot_attention_mask
from .wan.transformer import VideoPreState

__all__ = ["FastWAMError", "TrainingInputs", "FastWAM"]

# The keys `training_loss` reads out of a batch, with the RoboTwin shapes from spec §4.5.
REQUIRED_BATCH_KEYS = ("latents", "action", "proprio", "context", "context_mask")
OPTIONAL_BATCH_KEYS = ("image_is_pad", "action_is_pad")


class FastWAMError(ValueError):
    """A batch or argument that cannot be a correct FastWAM call."""


class TrainingInputs(NamedTuple):
    """`build_inputs`' return value: everything downstream of the VAE and the cache."""

    latents: jax.Array  # [B, 48, T_l, H_l, W_l]  clean, normalized
    hog_latents: jax.Array | None  # same geometry; training-only auxiliary target
    action: jax.Array  # [B, Sa, action_dim]     clean, normalized
    context: jax.Array  # [B, 129, 4096]          text + proprio token
    context_mask: jax.Array  # [B, 129] bool
    image_is_pad: jax.Array | None  # [B, T_video]
    action_is_pad: jax.Array | None  # [B, Sa]


class FastWAM(nnx.Module):
    """The uncond model.

    build_inputs(batch) -> TrainingInputs (validated, proprio token appended).
    training_loss(batch, rngs) -> (loss, metrics) with
        metrics = {"loss_video", "loss_action", "t_video", "t_action"}.

    Batch (latents produced in-loop by the VAE earlier in the train step, spec §4.5):
        latents      [B, 48, 3, 24, 20] float32   (VAE mu, normalized)
        action       [B, 32, 14]        float32   (z-scored, clamp(+-5))
        proprio      [B, 32, 14]        float32   (only index 0 is used)
        context      [B, 128, 4096]     bfloat16
        context_mask [B, 128]           bool      (all True by convention)
        image_is_pad [B, 9]             bool
        action_is_pad[B, 32]            bool
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        rngs: nnx.Rngs,
        dtype: jnp.dtype = jnp.float32,
        param_dtype: jnp.dtype = jnp.float32,
        include_hog: bool = False,
        **overrides,
    ):
        self.config = config
        self.mot = MoT.from_config(
            config,
            rngs=rngs,
            dtype=dtype,
            param_dtype=param_dtype,
            include_hog=include_hog,
            **overrides,
        )
        # Linear(proprio_dim -> text_dim): the proprio token lives in TEXT space and is
        # appended to the context BEFORE either tower's text_embedding runs, which is
        # why it is 14 -> 4096 and not 14 -> 3072 (fastwam.py:59, 219-240).
        self.proprio_encoder = nnx.Linear(
            rngs=rngs,
            in_features=config.proprio_dim,
            out_features=config.video_dit_config.text_dim,
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=nnx.initializers.xavier_uniform(),
        )
        self.tokens_per_frame: int | None = None  # filled by from_config, see below

    @classmethod
    def from_config(cls, config: Config, *, rngs: nnx.Rngs, **overrides) -> "FastWAM":
        """Build from the whole `Config` so the derived geometry can be pinned."""
        model = cls(config.model, rngs=rngs, **overrides)
        model.tokens_per_frame = config.shapes.tokens_per_frame
        return model

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    def append_proprio(
        self, context: jax.Array, context_mask: jax.Array, proprio: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """[B,128,4096] + [B,proprio_dim] -> [B,129,4096], mask likewise (fastwam.py:219-240).

        The appended mask entry is True: the proprio token is always attended to.
        """
        if proprio.ndim != 2:
            raise FastWAMError(f"proprio must be [B, D] at this point, got {proprio.shape}")
        if proprio.shape[1] != self.config.proprio_dim:
            raise FastWAMError(
                f"proprio last dim must be {self.config.proprio_dim}, got {proprio.shape[1]}"
            )
        token = self.proprio_encoder(proprio.astype(context.dtype))[:, None, :].astype(context.dtype)
        ones = jnp.ones((context_mask.shape[0], 1), dtype=context_mask.dtype)
        return jnp.concatenate([context, token], axis=1), jnp.concatenate([context_mask, ones], axis=1)

    def build_inputs(self, batch: dict[str, jax.Array]) -> TrainingInputs:
        required = REQUIRED_BATCH_KEYS + (("hog_latents",) if self.mot.hog is not None else ())
        optional = OPTIONAL_BATCH_KEYS + (() if self.mot.hog is not None else ("hog_latents",))
        missing = [key for key in required if key not in batch]
        if missing:
            raise FastWAMError(f"batch is missing {missing}; required: {list(required)}")
        unknown = sorted(set(batch) - set(required) - set(optional))
        if unknown:
            raise FastWAMError(
                f"batch has unknown key(s) {unknown}; known: "
                f"{list(required + optional)}"
            )

        latents, hog_latents, action = batch["latents"], batch.get("hog_latents"), batch["action"]
        context, context_mask, proprio = batch["context"], batch["context_mask"], batch["proprio"]
        if latents.ndim != 5:
            raise FastWAMError(f"latents must be [B, C, T_l, H_l, W_l], got {latents.shape}")
        if hog_latents is not None and hog_latents.shape != latents.shape:
            raise FastWAMError(
                f"hog_latents must exactly match RGB latents {latents.shape}, got {hog_latents.shape}"
            )
        if action.ndim != 3:
            raise FastWAMError(f"action must be [B, Sa, action_dim], got {action.shape}")
        if proprio.ndim != 3:
            raise FastWAMError(f"proprio must be [B, T, proprio_dim], got {proprio.shape}")
        batch_size = latents.shape[0]
        for name, array in (("action", action), ("proprio", proprio), ("context", context)):
            if array.shape[0] != batch_size:
                raise FastWAMError(f"{name} batch {array.shape[0]} != latents batch {batch_size}")
        if context.ndim != 3 or context_mask.ndim != 2 or context_mask.shape != context.shape[:2]:
            raise FastWAMError(
                f"context/context_mask must be [B,L,D]/[B,L], got {context.shape} / {context_mask.shape}"
            )

        # Only proprio step 0 is ever used (fastwam.py:361); the rest of the [B,32,14]
        # array exists because the dataset emits state[0..31].
        context, context_mask = self.append_proprio(context, context_mask, proprio[:, 0, :])
        return TrainingInputs(
            latents=latents,
            hog_latents=hog_latents,
            action=action,
            context=context,
            context_mask=context_mask,
            image_is_pad=batch.get("image_is_pad"),
            action_is_pad=batch.get("action_is_pad"),
        )

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def attention_mask(
        self, video_state: VideoPreState, action_state: ActionPreState, copy_lengths=None
    ) -> jax.Array:
        return build_mot_attention_mask(
            video_state.tokens.shape[1],
            action_state.tokens.shape[1],
            video_state.tokens_per_frame,
            copy_lengths,
        )

    def hog_attention_mask(
        self,
        video_state: VideoPreState,
        hog_state: VideoPreState,
        action_state: ActionPreState,
        copy_lengths=None,
    ) -> jax.Array:
        return build_hog_mot_attention_mask(
            video_state.tokens.shape[1],
            hog_state.tokens.shape[1],
            action_state.tokens.shape[1],
            video_state.tokens_per_frame,
            copy_lengths,
        )

    def predict(
        self,
        noisy_latents: jax.Array,
        noisy_action: jax.Array,
        t_video: jax.Array,
        t_action: jax.Array,
        context: jax.Array,
        context_mask: jax.Array,
        copy_lengths=None,
    ) -> tuple[jax.Array, jax.Array]:
        """One joint forward: `_predict_joint_noise` (fastwam.py:570-632).

        Returns (pred_video [B,48,T_l,H,W], pred_action [B,Sa,a_dim]) — flow-matching
        velocities, i.e. estimates of `noise - x0`.

        `copy_lengths` describes how `noisy_action`'s rows split into independent tau
        copies (`copies.py`); it reaches the mask and the action RoPE, which are the only
        two things that care. None is the single-chunk path. The model itself holds no
        copies state — this is a per-call argument, so nothing about inference or the
        checkpoint changes when training uses copies.
        """
        video_state = self.mot.video.pre_dit(noisy_latents, t_video, context, context_mask)
        action_state = self.mot.action.pre_dit(
            noisy_action, t_action, context, context_mask, copy_lengths
        )
        if self.tokens_per_frame is not None and video_state.tokens_per_frame != self.tokens_per_frame:
            raise FastWAMError(
                f"latents imply {video_state.tokens_per_frame} tokens/frame but the config says "
                f"{self.tokens_per_frame}"
            )
        return self.mot.forward_states(
            video_state,
            action_state,
            self.attention_mask(video_state, action_state, copy_lengths),
        )

    def predict_hog(
        self,
        noisy_latents: jax.Array,
        noisy_hog_latents: jax.Array,
        noisy_action: jax.Array,
        t_video: jax.Array,
        t_hog: jax.Array,
        t_action: jax.Array,
        context: jax.Array,
        context_mask: jax.Array,
        copy_lengths=None,
        *,
        predict_video: bool = True,
    ) -> tuple[jax.Array | None, jax.Array, jax.Array]:
        """Training-only joint RGB/HOG/Action forward."""
        if self.mot.hog is None:
            raise FastWAMError("predict_hog requires include_hog=True")
        video_state = self.mot.video.pre_dit(noisy_latents, t_video, context, context_mask)
        hog_state = self.mot.hog.pre_dit(noisy_hog_latents, t_hog, context, context_mask)
        action_state = self.mot.action.pre_dit(
            noisy_action, t_action, context, context_mask, copy_lengths
        )
        if video_state.tokens_per_frame != hog_state.tokens_per_frame:
            raise FastWAMError(
                f"RGB/HOG tokens per frame differ: {video_state.tokens_per_frame} vs "
                f"{hog_state.tokens_per_frame}"
            )
        return self.mot.forward_hog_states(
            video_state,
            hog_state,
            action_state,
            self.hog_attention_mask(video_state, hog_state, action_state, copy_lengths),
            predict_video=predict_video,
        )

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def _video_loss_per_sample(
        self,
        pred_video: jax.Array,
        target_video: jax.Array,
        image_is_pad: jax.Array | None,
    ) -> jax.Array:
        """fastwam.py:409-446, with `include_initial_video_step` fixed to False.

        That flag is `first_frame_latents is None` upstream, and
        `fuse_vae_embedding_in_latents` is True in every shipped config (config.py
        enforces it), so latent frame 0 is always dropped before this is called.
        """
        per_step = jnp.mean(
            (pred_video.astype(jnp.float32) - target_video.astype(jnp.float32)) ** 2, axis=(1, 3, 4)
        )
        if image_is_pad is None:
            return jnp.mean(per_step, axis=1)

        factor = 4  # vae.temporal_downsample_factor
        num_frames = image_is_pad.shape[1]
        if (num_frames - 1) % factor != 0:
            raise FastWAMError(
                f"cannot align image_is_pad with latent steps: num_frames={num_frames}, factor={factor}"
            )
        # image_is_pad[:, 1:] folded 4:1 — a latent step is padding only if all four of
        # the frames it covers are (fastwam.py:431-436).
        tail = image_is_pad[:, 1:].reshape(image_is_pad.shape[0], -1, factor)
        video_is_pad = jnp.all(tail, axis=2)
        if video_is_pad.shape[1] != per_step.shape[1]:
            raise FastWAMError(
                f"video-loss mask has {video_is_pad.shape[1]} steps but the loss has {per_step.shape[1]}"
            )
        valid = (~video_is_pad).astype(per_step.dtype)
        return jnp.sum(per_step * valid, axis=1) / jnp.clip(jnp.sum(valid, axis=1), 1.0)

    @staticmethod
    def _action_loss_per_sample(
        pred_action: jax.Array, target_action: jax.Array, action_is_pad: jax.Array | None
    ) -> jax.Array:
        """fastwam.py:550-556."""
        per_step = jnp.mean(
            (pred_action.astype(jnp.float32) - target_action.astype(jnp.float32)) ** 2, axis=2
        )
        if action_is_pad is None:
            return jnp.mean(per_step, axis=1)
        valid = (~action_is_pad).astype(per_step.dtype)
        return jnp.sum(per_step * valid, axis=1) / jnp.clip(jnp.sum(valid, axis=1), 1.0)

    @staticmethod
    def _pack_copies(chunk: jax.Array, lengths: tuple[int, ...]) -> jax.Array:
        """[B, n_act, ...] -> [B, sum(lengths), ...]: copy i is the first `lengths[i]` rows.

        A prefix, never an extension — every copy predicts the same future from the same
        observation, just over a shorter horizon. One full-length copy returns the array
        itself, so the single-copy path adds no op at all.
        """
        if len(lengths) == 1 and lengths[0] == chunk.shape[1]:
            return chunk
        return jnp.concatenate([chunk[:, :length] for length in lengths], axis=1)

    @staticmethod
    def _copy_t_per_row(t_action: jax.Array, lengths: tuple[int, ...]) -> jax.Array:
        """[B, M] per-copy flow times -> [B, sum(lengths)] per row. [B] passes through."""
        if len(lengths) == 1:
            return t_action
        return jnp.concatenate(
            [jnp.repeat(t_action[:, i : i + 1], length, axis=1) for i, length in enumerate(lengths)],
            axis=1,
        )

    def training_loss(
        self,
        batch: dict[str, jax.Array],
        rngs: "nnx.Rngs | jax.Array",
        *,
        timesteps: tuple[jax.Array, jax.Array] | None = None,
        noise: tuple[jax.Array, jax.Array] | None = None,
        copy_lengths=None,
    ):
        """(loss, metrics). `timesteps` / `noise` override the draws, for parity tests.

        The two timesteps are drawn INDEPENDENTLY (two schedulers, two `torch.rand`
        calls upstream); folding them into one draw would silently correlate the video
        and action noise levels.

        `copy_lengths` (from `flow.tau_copies`, resolved in `config.shapes.copy_lengths`)
        packs M independent copies of the action chunk into this one forward, each with
        its own flow time and its own noise — M samples of the action objective's
        tau-integral for one video forward. See `copies.py`. None, and equally a single
        full-length copy, is the released recipe with no extra op anywhere.

        `timesteps`, when given, supplies the PER-COPY times ([B] or [B, M]); the
        expansion to one time per row happens here.
        """
        inputs = self.build_inputs(batch)
        batch_size = inputs.latents.shape[0]
        video_cfg = self.config.video_scheduler
        action_cfg = self.config.action_scheduler
        # The same resolver the config uses, so "what the yaml asked for" and "what this
        # forward packs" can never be two different translations of the same knob.
        lengths = resolve_copy_lengths(1, copy_lengths, inputs.action.shape[1])
        n_copies = len(lengths)

        use_hog = self.mot.hog is not None
        if use_hog and inputs.hog_latents is None:
            raise FastWAMError("include_hog=True requires hog_latents")
        key = rngs.noise() if isinstance(rngs, nnx.Rngs) else rngs
        if use_hog:
            key_tv, key_th, key_ta, key_nv, key_nh, key_na = jax.random.split(key, 6)
        else:
            key_tv, key_ta, key_nv, key_na = jax.random.split(key, 4)
        if timesteps is None:
            t_video = flow_match.sample_training_t(
                key_tv, batch_size, video_cfg.train_shift, video_cfg.num_train_timesteps
            )
            t_hog = (
                flow_match.sample_training_t(
                    key_th, batch_size, video_cfg.train_shift, video_cfg.num_train_timesteps
                )
                if use_hog
                else None
            )
            t_action = flow_match.sample_training_t(
                key_ta, batch_size, action_cfg.train_shift, action_cfg.num_train_timesteps,
                copies=n_copies,
            )
        else:
            if use_hog:
                if len(timesteps) != 3:
                    raise FastWAMError("HOG training timesteps must be (t_video, t_hog, t_action)")
                t_video, t_hog, t_action = timesteps
            else:
                if len(timesteps) != 2:
                    raise FastWAMError("classic training timesteps must be (t_video, t_action)")
                t_video, t_action = timesteps
                t_hog = None
            # An override is PER-COPY, like the draw it replaces. Getting this wrong would
            # otherwise surface as an IndexError from the row expansion below, which says
            # nothing about what the caller actually got wrong.
            want = (batch_size,) if n_copies == 1 else (batch_size, n_copies)
            if t_action.shape != want:
                raise FastWAMError(
                    f"the t_action override must be {want} — one flow time per copy for "
                    f"{n_copies} copies {lengths} — got {t_action.shape}"
                )

        # Everything action-side now lives at the PACKED length. The clean chunk and its
        # pad mask are tiled first so the noise draw, the target and the loss all follow
        # the same geometry without any of them knowing about copies.
        clean_action = self._pack_copies(inputs.action, lengths)
        action_is_pad = (
            None if inputs.action_is_pad is None else self._pack_copies(inputs.action_is_pad, lengths)
        )
        t_action_rows = self._copy_t_per_row(t_action, lengths)

        if noise is None:
            noise_video = jax.random.normal(key_nv, inputs.latents.shape, inputs.latents.dtype)
            noise_hog = (
                jax.random.normal(key_nh, inputs.hog_latents.shape, inputs.hog_latents.dtype)
                if use_hog
                else None
            )
            noise_action = jax.random.normal(key_na, clean_action.shape, clean_action.dtype)
        else:
            if use_hog:
                if len(noise) != 3:
                    raise FastWAMError("HOG training noise must be (video, hog, action)")
                noise_video, noise_hog, noise_action = noise
            else:
                if len(noise) != 2:
                    raise FastWAMError("classic training noise must be (video, action)")
                noise_video, noise_action = noise
                noise_hog = None

        noisy_latents = flow_match.add_noise(
            inputs.latents, noise_video, t_video, video_cfg.num_train_timesteps
        )
        predict_video = self.config.loss.lambda_video != 0.0
        target_video = (
            flow_match.training_target(inputs.latents, noise_video) if predict_video else None
        )
        # The clean first latent frame is written back AFTER noising (fastwam.py:467-468).
        # The target keeps its noised-frame-0 entry, which is then dropped below.
        noisy_latents = noisy_latents.at[:, :, 0:1].set(inputs.latents[:, :, 0:1])

        if use_hog:
            noisy_hog_latents = flow_match.add_noise(
                inputs.hog_latents, noise_hog, t_hog, video_cfg.num_train_timesteps
            )
            target_hog = flow_match.training_target(inputs.hog_latents, noise_hog)
            # The three-tower ablation can remove the observed/current HOG
            # condition while retaining the future HOG prediction objective.
            # Future frames still carry their flow-matched noisy HOG latents;
            # only the frame-0 condition is replaced, and the frame-0 target
            # is discarded below exactly as in the historical recipe.
            hog_condition = inputs.hog_latents[:, :, 0:1]
            if self.config.hog_condition_mode == "none":
                hog_condition = jnp.zeros_like(hog_condition)
            elif self.config.hog_condition_mode == "noise":
                # Pure Gaussian placeholder: never mix in current-frame HOG,
                # even when the HOG flow timestep is close to clean. Frame 0
                # remains excluded from the objective; future targets are real.
                hog_condition = noise_hog[:, :, 0:1]
            noisy_hog_latents = noisy_hog_latents.at[:, :, 0:1].set(
                hog_condition
            )

        noisy_action = flow_match.add_noise(
            clean_action, noise_action, t_action_rows, action_cfg.num_train_timesteps
        )
        target_action = flow_match.training_target(clean_action, noise_action)

        if use_hog:
            pred_video, pred_hog, pred_action = self.predict_hog(
                noisy_latents,
                noisy_hog_latents,
                noisy_action,
                t_video,
                t_hog,
                t_action_rows,
                inputs.context,
                inputs.context_mask,
                lengths,
                predict_video=predict_video,
            )
            pred_hog = pred_hog[:, :, 1:]
            target_hog = target_hog[:, :, 1:]
        else:
            pred_video, pred_action = self.predict(
                noisy_latents,
                noisy_action,
                t_video,
                t_action_rows,
                inputs.context,
                inputs.context_mask,
                lengths,
            )
        if predict_video:
            loss_video = jnp.mean(
                self._video_loss_per_sample(
                    pred_video[:, :, 1:], target_video[:, :, 1:], inputs.image_is_pad
                ) * flow_match.training_weight(t_video, video_cfg.num_train_timesteps)
            )
        else:
            loss_video = jnp.asarray(0.0, jnp.float32)
        loss_hog = (
            jnp.mean(
                self._video_loss_per_sample(pred_hog, target_hog, inputs.image_is_pad)
                * flow_match.training_weight(t_hog, video_cfg.num_train_timesteps)
            )
            if use_hog
            else jnp.asarray(0.0, jnp.float32)
        )
        action_weight = flow_match.training_weight(t_action, action_cfg.num_train_timesteps)
        if n_copies == 1:
            loss_action = jnp.mean(
                self._action_loss_per_sample(pred_action, target_action, action_is_pad)
                * action_weight
            )
        else:
            # Average each copy's OWN mean, not all rows together. At equal lengths the two
            # agree; at unequal ones a row-wise mean lets a 32-row copy outweigh a 4-row copy
            # 8:1 and drowns out exactly the short horizons the mix exists to train. One copy
            # is one sample of the objective, whatever its length.
            per_copy = []
            offset = 0
            for index, length in enumerate(lengths):
                rows = slice(offset, offset + length)
                per_copy.append(
                    jnp.mean(
                        self._action_loss_per_sample(
                            pred_action[:, rows],
                            target_action[:, rows],
                            None if action_is_pad is None else action_is_pad[:, rows],
                        )
                        * action_weight[:, index]
                    )
                )
                offset += length
            loss_action = sum(per_copy) / n_copies
        loss = (
            self.config.loss.lambda_video * loss_video
            + self.config.loss.lambda_hog * loss_hog
            + self.config.loss.lambda_action * loss_action
        )
        metrics = {
            "loss_video": self.config.loss.lambda_video * loss_video,
            "loss_action": self.config.loss.lambda_action * loss_action,
            "t_video": t_video,
            "t_action": t_action,
        }
        if use_hog:
            metrics["loss_hog"] = self.config.loss.lambda_hog * loss_hog
            metrics["t_hog"] = t_hog
        else:
            # Keep the training-step metric schema stable for the two-tower
            # RGB→HOG recipe.  The zero is a device scalar and therefore does
            # not introduce a host sync or a fake auxiliary objective.
            metrics["loss_hog"] = jnp.asarray(0.0, jnp.float32)
            metrics["t_hog"] = jnp.zeros_like(t_video)
        return loss, metrics

    # ------------------------------------------------------------------ #
    # inference (joint path; the KV-cache fast path is M3)
    # ------------------------------------------------------------------ #
    def denoise_joint(
        self,
        *,
        first_frame_latents: jax.Array,
        noise_video: jax.Array,
        noise_action: jax.Array,
        context: jax.Array,
        context_mask: jax.Array,
        proprio: jax.Array | None = None,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
    ) -> dict[str, jax.Array]:
        """`infer_joint` (fastwam.py:725-903) minus the VAE decode.

        `noise_video` / `noise_action` are the initial N(0,1) draws, passed in rather
        than sampled here: upstream draws them from a seeded `torch.Generator` on CPU,
        which no JAX PRNG reproduces, so every parity test supplies its own.

        Args:
            first_frame_latents: [B, 48, 1, H_l, W_l] — the encoded current observation.
            context / context_mask: [B, 128, 4096] / [B, 128] if `proprio` is given
                (the proprio token is appended here), else the final [B, 129, ...].

        Returns: {"action": [B, Sa, a_dim], "latents_video": [B, 48, T_l, H_l, W_l]}.
        """
        if proprio is not None:
            context, context_mask = self.append_proprio(context, context_mask, proprio)
        if first_frame_latents.shape[2] != 1:
            raise FastWAMError(
                f"first_frame_latents must have exactly one latent frame, got {first_frame_latents.shape}"
            )

        video_cfg, action_cfg = self.config.video_scheduler, self.config.action_scheduler
        shift_v = video_cfg.infer_shift if sigma_shift is None else sigma_shift
        shift_a = action_cfg.infer_shift if sigma_shift is None else sigma_shift
        t_video_all, delta_video_all = flow_match.build_inference_schedule(
            num_inference_steps, shift_v, video_cfg.num_train_timesteps
        )
        t_action_all, delta_action_all = flow_match.build_inference_schedule(
            num_inference_steps, shift_a, action_cfg.num_train_timesteps
        )

        batch = noise_video.shape[0]
        latents_video = noise_video.at[:, :, 0:1].set(first_frame_latents)
        latents_action = noise_action

        for index in range(num_inference_steps):
            t_video = jnp.broadcast_to(t_video_all[index], (batch,))
            t_action = jnp.broadcast_to(t_action_all[index], (batch,))
            pred_video, pred_action = self.predict(
                latents_video, latents_action, t_video, t_action, context, context_mask
            )
            latents_video = flow_match.step(pred_video, delta_video_all[index], latents_video)
            latents_action = flow_match.step(pred_action, delta_action_all[index], latents_action)
            # Re-applied after EVERY step (spec §9.5 item 25) — the first frame is the
            # observation, not something the model is allowed to move.
            latents_video = latents_video.at[:, :, 0:1].set(first_frame_latents)

        return {"action": latents_action, "latents_video": latents_video}

    def denoise_action(
        self,
        *,
        first_frame_latents: jax.Array,
        noise_action: jax.Array,
        context: jax.Array,
        context_mask: jax.Array,
        proprio: jax.Array | None = None,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
    ) -> jax.Array:
        """`infer_action` (fastwam.py:905-1048) WITHOUT the KV cache.

        The video tower is fed the single observed latent frame at t=0 on every step
        instead of being prefilled once. That is the same computation the cache
        performs — the video branch's per-layer K/V do not depend on the action tokens
        or on the step index — just without reusing it, so this is `infer_action`'s
        result at `infer_action`'s cost only up to the 10x video-tower redundancy that
        M3's cache removes.

        Returns: [B, Sa, action_dim].
        """
        if proprio is not None:
            context, context_mask = self.append_proprio(context, context_mask, proprio)
        action_cfg = self.config.action_scheduler
        shift = action_cfg.infer_shift if sigma_shift is None else sigma_shift
        timesteps, deltas = flow_match.build_inference_schedule(
            num_inference_steps, shift, action_cfg.num_train_timesteps
        )

        batch = noise_action.shape[0]
        latents_action = noise_action
        # Latent frame 0 is declared clean, so the video timestep is exactly 0 and
        # `build_token_timesteps` zeroes it anyway; passing zeros makes that explicit.
        t_video = jnp.zeros((batch,), jnp.float32)
        for index in range(num_inference_steps):
            _, pred_action = self.predict(
                first_frame_latents,
                latents_action,
                t_video,
                jnp.broadcast_to(timesteps[index], (batch,)),
                context,
                context_mask,
            )
            latents_action = flow_match.step(pred_action, deltas[index], latents_action)
        return latents_action

    def infer_action(
        self,
        *,
        first_frame_latents: jax.Array,
        noise_action: jax.Array,
        context: jax.Array,
        context_mask: jax.Array,
        proprio: jax.Array | None = None,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
    ) -> jax.Array:
        """`infer_action` (fastwam.py:905-1048) WITH the KV cache. The deployed path.

        Identical arguments and identical result to :meth:`denoise_action`; the only
        difference is that the video tower runs once instead of `num_inference_steps`
        times. At the released widths that is 30 layers over 120 tokens through a 5 B
        expert saved 9 times out of 10, which is most of the per-replan cost.

        The equivalence is not an approximation and it is not a claim about the weights:
        the video branch cannot see the action tokens (the joint mask's video->action
        block is all False) and its timestep is pinned to 0 by `build_token_timesteps`,
        so its per-layer K/V are literally the same tensors on every step. What the cache
        removes is recomputation, nothing else. `tests/test_inference_cache.py::
        test_cached_matches_uncached_denoise` measures the residual, which is fp32
        reassociation only.

        Returns: [B, Sa, action_dim] — still NORMALIZED. Denormalizing is the policy's
        job (inference/policy.py) because that needs `dataset_stats.json`, which the
        model does not carry.
        """
        if proprio is not None:
            context, context_mask = self.append_proprio(context, context_mask, proprio)
        if first_frame_latents.ndim != 5 or first_frame_latents.shape[2] != 1:
            raise FastWAMError(
                f"first_frame_latents must be [B, C, 1, H_l, W_l] — one encoded observation, "
                f"got {first_frame_latents.shape}"
            )
        if noise_action.ndim != 3 or noise_action.shape[0] != first_frame_latents.shape[0]:
            raise FastWAMError(
                f"noise_action must be [B, Sa, action_dim] matching the latent batch "
                f"{first_frame_latents.shape[0]}, got {noise_action.shape}"
            )

        action_cfg = self.config.action_scheduler
        shift = action_cfg.infer_shift if sigma_shift is None else sigma_shift
        timesteps, deltas = flow_match.build_inference_schedule(
            num_inference_steps, shift, action_cfg.num_train_timesteps
        )

        batch = noise_action.shape[0]
        video_state = self.mot.video.pre_dit(
            first_frame_latents, jnp.zeros((batch,), jnp.float32), context, context_mask
        )
        video_seq_len = video_state.tokens.shape[1]
        if video_seq_len != video_state.tokens_per_frame:
            raise FastWAMError(
                f"the prefill must see exactly one latent frame: {video_seq_len} tokens for "
                f"{video_state.tokens_per_frame} tokens/frame"
            )
        attention_mask = build_mot_attention_mask(
            video_seq_len, noise_action.shape[1], video_state.tokens_per_frame
        )
        cache = self.mot.prefill_video_cache(
            video_state, attention_mask[:video_seq_len, :video_seq_len]
        )

        latents_action = noise_action
        for index in range(num_inference_steps):
            action_state = self.mot.action.pre_dit(
                latents_action,
                jnp.broadcast_to(timesteps[index], (batch,)),
                context,
                context_mask,
            )
            tokens = self.mot.forward_action_with_video_cache(
                action_state, cache, attention_mask, video_seq_len
            )
            pred_action = self.mot.action.post_dit(tokens)
            latents_action = flow_match.step(pred_action, deltas[index], latents_action)
        return latents_action
