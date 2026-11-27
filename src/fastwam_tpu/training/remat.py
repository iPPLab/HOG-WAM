"""Rematerialization policy: the memory/compute dial for the MoT block stack.

Ported from
    codebases/TinyWAM-v4/tinywam/models/remat.py — RematPolicy enum, from_str,
        check_checkpoint_names, apply_remat.
    reference/maxdiffusion/src/maxdiffusion/models/gradient_checkpoint.py —
        GradientCheckpointType and the host-offload variants.

What is actually available here, and why the enum is shorter than TinyWAM's
--------------------------------------------------------------------------
TinyWAM's six tiers exist because its WAMBlock plants `checkpoint_name` tags on the joint
q/k/v, the attention output and the FFN activation, and the selective policies are
defined over those names. **This model plants no tags** (`grep checkpoint_name
src/fastwam_tpu/models` finds nothing outside this file), and `models/mot.py` accepts
only `REMAT_POLICIES = ("none", "full")`.

A tag-less selective policy is not a degraded version of a working one — it silently
saves NOTHING and is byte-identical to `full` while reading like a tuning knob. That is
the failure `check_checkpoint_names` was written to prevent, so the tiers that need tags
raise here instead of being offered. Adding one back is two steps that must happen
together: tag the tensor in `mot.py::_layer`, then list the name in
`KNOWN_CHECKPOINT_NAMES`.

Note on the reference: both official RoboTwin/LIBERO task configs set
`mot_checkpoint_mixed_attn: false`, i.e. the PyTorch run trains with no remat at all. On
TPU with 6 B trainable params that is not reproducible and does not need to be — remat
changes memory and speed, not numerics.

Pitfalls already paid for
    * splash / jax.checkpoint do not compose inside shard_map (v6e campaign). Nothing
      here uses shard_map, and `training/sharding.py` explains why.
    * remat with a naive double forward measured as a real slowdown on v6e; measure with
      a profiler before believing any policy is better.
"""

import enum

__all__ = [
    "RematPolicy",
    "KNOWN_CHECKPOINT_NAMES",
    "apply_remat",
    "check_checkpoint_names",
    "resolve",
]

# Every `checkpoint_name` tag the model plants. Empty on purpose: see the module
# docstring. A selective policy validated against an empty tuple raises, which is the
# loud version of "this tier would have done nothing".
KNOWN_CHECKPOINT_NAMES: tuple[str, ...] = ()

# The tiers that need tags, mapped to the names they would have needed. Kept so the error
# can say what is missing rather than just "unsupported".
_NEEDS_TAGS = {
    "save_qkv": ("joint_q", "joint_k", "joint_v"),
    "save_attn_out": ("joint_attn_out", "joint_attn_lse"),
    "offload_hidden": ("hidden_v", "hidden_a"),
    "custom": ("<caller-supplied>",),
}


class RematPolicy(enum.Enum):
    """`tpu.remat_policy`. Only NONE and FULL are implemented — see the module docstring.

    NONE  no checkpointing; XLA keeps whatever it wants. Fastest per step, most HBM.
    FULL  only the block inputs survive the forward (`nnx.remat` with policy=None);
          everything inside a MoT layer is recomputed in the backward.
    """

    NONE = "none"
    FULL = "full"
    SAVE_QKV = "save_qkv"
    SAVE_ATTN_OUT = "save_attn_out"
    OFFLOAD_HIDDEN = "offload_hidden"
    CUSTOM = "custom"

    @classmethod
    def from_str(cls, value: str) -> "RematPolicy":
        """Case-insensitive parse; an unknown name lists the valid ones."""
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            valid = ", ".join(policy.value for policy in cls)
            raise ValueError(f"tpu.remat_policy: unknown policy {value!r} (valid: {valid})") from None

    @property
    def implemented(self) -> bool:
        return self in (RematPolicy.NONE, RematPolicy.FULL)

    def model_policy(self) -> str:
        """The string `models/mot.py` accepts. Raises for the tag-based tiers."""
        if self.implemented:
            return self.value
        wanted = ", ".join(_NEEDS_TAGS[self.value])
        raise NotImplementedError(
            f"tpu.remat_policy='{self.value}' needs checkpoint_name tags ({wanted}) that this "
            "model does not plant, so it would save nothing and be silently identical to 'full'. "
            "Use 'none' or 'full'; to add this tier, tag the tensors in models/mot.py::_layer AND "
            "list the names in remat.KNOWN_CHECKPOINT_NAMES."
        )

    def jax_policy(self, saved: tuple[str, ...] = (), offloaded: tuple[str, ...] = ()):
        """The `jax.checkpoint` policy. None == save-nothing, `nnx.remat`'s default."""
        if self is RematPolicy.NONE:
            raise ValueError("RematPolicy.NONE has no jax policy; apply_remat skips remat entirely")
        if self is RematPolicy.FULL:
            return None
        check_checkpoint_names(*saved, *offloaded)  # always raises while the tag set is empty
        raise AssertionError("unreachable: check_checkpoint_names admits nothing today")


def check_checkpoint_names(*names: str) -> None:
    """Fail loudly on a name the model never tags (a typo silently saves nothing)."""
    unknown = [name for name in names if name not in KNOWN_CHECKPOINT_NAMES]
    if unknown:
        raise ValueError(
            f"unknown checkpoint name(s) {unknown}; this model tags "
            f"{list(KNOWN_CHECKPOINT_NAMES) or 'NOTHING'} — a selective remat policy over an "
            "untagged model saves nothing at all. Tag the tensors first (models/mot.py::_layer)."
        )


def resolve(remat_policy: str) -> RematPolicy:
    """Config string -> tier, refusing anything that would be a no-op in disguise."""
    policy = RematPolicy.from_str(remat_policy)
    policy.model_policy()  # raises for the unimplemented tiers, with the how-to
    return policy


def apply_remat(fn, policy: RematPolicy, *, prevent_cse: bool = True, saved=(), offloaded=()):
    """Wrap a per-layer function according to `policy`.

    NONE returns `fn` unchanged (a zero-cost off switch). `prevent_cse` must be False
    under `nnx.scan`: the scan boundary already stops CSE from merging the forward and
    the recomputation, and leaving it True costs compile time for nothing. That pairing
    lives in `models/mot.py::_layer_body`, the only caller inside the model; this
    function exists for tools and tests that wrap a layer directly.
    """
    from flax import nnx

    if policy is RematPolicy.NONE:
        return fn
    return nnx.remat(fn, policy=policy.jax_policy(saved, offloaded), prevent_cse=prevent_cse)
