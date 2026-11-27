"""Action / proprio normalisation, on the official constants.

We do not recompute the statistics, we copy the official `dataset_stats.json` and read
four 14-vectors out of it. Two reasons, both from `m4_data_design.md` §Q3:

* The official file is what the released checkpoint was trained with. Any other set of
  constants is a pure affine misalignment of the action space — no exception, no log
  line, just a model that "seems to have learned the wrong thing".
* Reproducing the file would mean reproducing its quirks: the means are **per-episode
  equal-weight** rather than per-row (`base_lerobot_dataset.py:328-335`, the comment
  says "assume that each minibatch has equal number of samples"), the action stream is
  passed through `sliding_window_with_replication(a, 32)` *before* the statistics are
  taken (`:165`), and `global_q01` is an `amin` of per-episode 1st percentiles rather
  than a percentile. Our own honest recomputation would disagree with it and the
  disagreement would read like a bug.

So the whole normaliser is `clamp(x * scale + offset, -5, 5)` with
`scale = 1/(std + 1e-8)`, `offset = -mean/(std + 1e-8)` (`utils/normalizer.py:100-133`),
action and proprio each on their own constants. The clamp is **forward only** — the
inverse used at deployment has none, so a clamped sample is information the model can
never be asked to give back.

Arithmetic order follows the official exactly: json -> float64 -> cast to float32 ->
compute scale/offset in float32. Deriving them in float64 and casting afterwards moves
the last couple of mantissa bits, which is not worth a second set of numbers.
"""

import hashlib
import json

import numpy as np

__all__ = ["Normalizer", "OfficialStats", "load_official_stats"]

_STD_REG = 1e-8
_CLAMP = 5.0
_FIELDS = ("action", "state")
_KEY = "default"


class Normalizer:
    """One field's affine map. `forward` clamps, `backward` does not."""

    __slots__ = ("scale", "offset", "mean", "std")

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = np.ascontiguousarray(mean, np.float32)
        self.std = np.ascontiguousarray(std, np.float32)
        if self.mean.shape != self.std.shape or self.mean.ndim != 1:
            raise ValueError(f"mean {self.mean.shape} and std {self.std.shape} must be one 1-D pair")
        self.scale = (np.float32(1.0) / (self.std + np.float32(_STD_REG))).astype(np.float32)
        self.offset = (-self.mean / (self.std + np.float32(_STD_REG))).astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def forward(self, x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        """Raw values `[..., dim]` -> normalised, clamped to +-5."""
        x = np.asarray(x, np.float32)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape}")
        out = np.multiply(x, self.scale, out=out, dtype=np.float32)
        out += self.offset
        return np.clip(out, -_CLAMP, _CLAMP, out=out)

    def backward(self, x: np.ndarray) -> np.ndarray:
        """Normalised -> raw. No clamp, matching `normalizer.py:131-133`."""
        x = np.asarray(x, np.float32)
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected last dim {self.dim}, got {x.shape}")
        return ((x - self.offset) / self.scale).astype(np.float32)


class OfficialStats:
    """The four vectors the recipe actually consumes, plus what the file claims about
    the dataset it was computed on (used to prove our episode set is the same one)."""

    __slots__ = ("action", "proprio", "num_episodes", "num_transition", "sha256", "source")

    def __init__(self, action: Normalizer, proprio: Normalizer, num_episodes: int,
                 num_transition: int, sha256: str, source: str):
        self.action = action
        self.proprio = proprio
        self.num_episodes = int(num_episodes)
        self.num_transition = int(num_transition)
        self.sha256 = sha256
        self.source = source

    def describe(self) -> str:
        return (
            f"official dataset_stats: {self.source} sha256={self.sha256[:16]} "
            f"episodes={self.num_episodes} transitions={self.num_transition:,} "
            f"z-score(global) + clamp(+-{_CLAMP:g}) forward only"
        )


def load_official_stats(
    blob: bytes, *, source: str = "<bytes>", expect_sha256: str | None = None
) -> OfficialStats:
    """Parse `dataset_stats.json`. Anything unexpected raises; nothing is defaulted.

    `use_stepwise_action_norm` is False in every shipped RoboTwin config, so the
    `stepwise_*` arrays in the file are dead weight and are deliberately not read — if a
    future config wants them it has to say so and get a matching error here first.
    """
    digest = hashlib.sha256(blob).hexdigest()
    if expect_sha256 is not None and digest != expect_sha256:
        raise ValueError(
            f"{source}: sha256 {digest} does not match the configured {expect_sha256}; the "
            "normalisation constants are not the ones this run was set up with"
        )
    raw = json.loads(blob)
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected a json object, got {type(raw).__name__}")

    norms = {}
    for field in _FIELDS:
        if field not in raw or _KEY not in raw[field]:
            raise ValueError(f"{source}: missing {field}.{_KEY}")
        stats = raw[field][_KEY]
        vectors = {}
        for name in ("global_mean", "global_std"):
            if name not in stats:
                raise ValueError(f"{source}: {field}.{_KEY} has no {name}")
            v = np.asarray(stats[name], np.float64)
            if v.ndim != 1:
                raise ValueError(f"{source}: {field}.{_KEY}.{name} must be 1-D, got shape {v.shape}")
            if not np.isfinite(v).all():
                raise ValueError(f"{source}: {field}.{_KEY}.{name} contains non-finite values")
            vectors[name] = v.astype(np.float32)
        if (vectors["global_std"] <= 0).any():
            raise ValueError(f"{source}: {field}.{_KEY}.global_std has a non-positive entry")
        norms[field] = Normalizer(vectors["global_mean"], vectors["global_std"])

    for name in ("num_episodes", "num_transition"):
        if name not in raw:
            raise ValueError(f"{source}: missing {name}")

    return OfficialStats(
        action=norms["action"],
        proprio=norms["state"],  # the official calls proprio "state" throughout
        num_episodes=raw["num_episodes"],
        num_transition=raw["num_transition"],
        sha256=digest,
        source=source,
    )
