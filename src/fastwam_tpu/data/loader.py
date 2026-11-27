"""Streaming loader: L3 `.twp` on GCS -> the tensors the training step consumes.

There is no offline latent cache. The plan used to call for one; `m4_data_design.md` §Q4
overturned that and the plan now agrees (`fastwam_tpu_plan.md` M4). The short version:
every LeRobot row is its own window start, and the 9 stride-4 frames of two neighbouring
starts land in completely different Wan-VAE temporal groups, so latents cannot be shared
between windows. Materialising them per window is 782 GiB; materialising them densely
per episode and slicing is the exact shape of the mistake that produced a 0% closed loop
in `project_fastwam_coupling`. The official pipeline runs the VAE inside the training
step, so doing the same is not a compromise, it *is* the faithful path — and it lets us
read the L3 pixels that already exist, in-region, without writing a byte.

Per sample (spec §4.5, RoboTwin column; `video` replaces `latents` because the VAE now
runs in the step):

    video        [3, 9, 384, 320]   float32   [-1, 1], the official 3-camera canvas
    action       [32, 14]           float32   z-scored on official stats, clamp(+-5)
    proprio      [32, 14]           float32   same, on the state constants; index 0 used
    context      [128, 4096]        float32   UMT5-XXL; the step casts to bf16
    context_mask [128]              bool      all True by construction (spec §2.8)
    image_is_pad [9]                bool
    action_is_pad[32]               bool

Layers: `twp` (the file format, vendored), `pool` (which episodes are resident),
`windows` (which indices a start implies), `canvas` (pixels), `normalize` (constants),
`text` (conditioning). This module only assembles.

numpy, gcsfs and a JPEG decoder — no jax and no torch below `global_batches`, so the
data plane can saturate 240 host cores without touching an accelerator context.
"""

import numpy as np

from fastwam_tpu.data import canvas as canvas_mod
from fastwam_tpu.data import windows
from fastwam_tpu.data.normalize import load_official_stats
from fastwam_tpu.data.pool import TwpIndex, TwpPool, open_filesystem, stable_seed
from fastwam_tpu.data.text import build_text_source

__all__ = ["ARRAY_KEYS", "RobotwinTwpDataset", "global_batches", "make_iterator", "split_episodes"]

# Batch entries that are per-sample tensors and become global jax.Arrays. Everything else
# in the batch dict is host-side provenance (instructions, episode ids) for logging.
ARRAY_KEYS = (
    "video",
    "action",
    "proprio",
    "context",
    "context_mask",
    "image_is_pad",
    "action_is_pad",
)


def split_episodes(n_episodes: int, val_proportion: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Our own train/val split over the L3 index.

    The official's split cannot be reproduced: it shuffles LeRobot episode numbers that
    only exist inside its own conversion (`base_lerobot_dataset.py:94-102`,
    `m4_data_design.md` R6). We keep its *shape* — shuffle, take the first
    `int(n*(1-p))` for train — and record the seed, so the split is at least reproducible
    on our side. It does not affect the A line, and the official statistics were computed
    over all 27,500 episodes anyway (`num_episodes` in the stats file says so), so the
    normalisation constants do not depend on where the line falls.
    """
    if not 0.0 <= val_proportion < 1.0:
        raise ValueError(f"val_set_proportion must be in [0, 1), got {val_proportion}")
    order = np.random.default_rng(stable_seed(seed, "split")).permutation(n_episodes)
    if val_proportion < 1e-6:
        return order.astype(np.int64), np.empty(0, np.int64)
    cut = int(n_episodes * (1.0 - val_proportion))
    return order[:cut].astype(np.int64), order[cut:].astype(np.int64)


class RobotwinTwpDataset:
    """Windows over L3 `.twp` episodes, assembled into official-shaped samples."""

    def __init__(
        self,
        prefix: str,
        *,
        stats_blob: bytes,
        stats_source: str = "<bytes>",
        stats_sha256: str | None = None,
        num_frames: int = 33,
        action_video_freq_ratio: int = 4,
        video_size: tuple[int, int] = canvas_mod.CANVAS_HW,
        action_shift: int = windows.ACTION_SHIFT_DEFAULT,
        source_color_order: str = canvas_mod.SOURCE_COLOR_ORDER,
        text_mode: str = "sidetable",
        text_dir: str | None = None,
        context_len: int = 128,
        task_filter: "tuple[str, ...] | list[str] | None" = None,
        val_set_proportion: float = 0.01,
        split: str = "train",
        episode_weighting: str = "transitions",
        pool_size: int = 4096,
        pool_rate: float = 1.0,
        pool_prefetch_threads: int = 8,
        decode_threads: int = 16,
        seed: int = 42,
        host_id: int = 0,
        n_hosts: int = 1,
        fs=None,
        expect_index_fingerprint: str | None = None,
    ):
        import concurrent.futures as cf

        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if (num_frames - 1) % action_video_freq_ratio:
            raise ValueError(
                f"(num_frames-1)={num_frames - 1} must divide by action_video_freq_ratio="
                f"{action_video_freq_ratio}"
            )
        self.video_frames = (num_frames - 1) // action_video_freq_ratio + 1
        if (self.video_frames - 1) % 4:
            raise ValueError(
                f"the {self.video_frames} sampled video frames must satisfy (T-1)%4==0 for the VAE"
            )
        self.action_horizon = num_frames - 1
        self.video_stride = int(action_video_freq_ratio)
        self.action_shift = int(action_shift)
        self.source_color_order = source_color_order
        self.split = split
        canvas_mod.assert_canvas_matches(video_size, len(canvas_mod.TILES))

        self.fs = fs if fs is not None else open_filesystem(prefix)
        full = TwpIndex(prefix, self.fs)
        self.aligned = full.spec.get("action_shift") == 0 and full.spec.get("color_order") == "rgb"
        if self.aligned and (self.action_shift != 0 or self.source_color_order != "rgb"):
            raise ValueError("aligned RoboTwin data requires data.robotwin_packing: aligned")
        if expect_index_fingerprint is not None and full.fingerprint != expect_index_fingerprint:
            raise ValueError(
                f"L3 index fingerprint {full.fingerprint} does not match the configured "
                f"{expect_index_fingerprint}; the dataset was rebuilt or episodes were excluded"
            )
        self.stats = load_official_stats(stats_blob, source=stats_source, expect_sha256=stats_sha256)

        # Gate 1: our own window count over the full set against the official transition
        # count. These two numbers were produced by two pipelines that share nothing, so
        # equality is the proof that the episodes are the same episodes.
        self.full_windows = full.num_windows
        self.transition_match = self.full_windows == self.stats.num_transition

        # Task subset, THEN split. Both orders give a reproducible split; this one keeps the
        # split a function of the subset alone, so `val_set_proportion` means the same
        # fraction of what is actually trained on however few tasks are named. (Splitting
        # first would leave the small run's val set at the mercy of how the global 1%
        # happened to fall across the tasks — for a two-task filter it can be empty.)
        # The two gates above deliberately run on the FULL manifest: a subset run still has
        # to prove it is reading the official 27,500-episode set before it narrows.
        self.task_filter = None if task_filter is None else tuple(dict.fromkeys(task_filter))
        self.full_tasks = len(set(full.task_key))
        pool_index = full if self.task_filter is None else full.select_tasks(self.task_filter)
        self.filtered_windows = pool_index.num_windows

        train_idx, val_idx = split_episodes(len(pool_index), val_set_proportion, seed)
        keep = train_idx if split == "train" else val_idx
        if keep.size == 0:
            raise ValueError(
                f"the {split} split is empty (val_set_proportion={val_set_proportion}, "
                f"{len(pool_index)} episodes after task_filter={self.task_filter})"
            )
        self.index = pool_index.select(np.sort(keep))

        missing = [r for r, _ in canvas_mod.TILES if r not in self.index.camera_roles]
        if missing:
            raise ValueError(
                f"the canvas needs camera roles {missing}, but L3 only has {self.index.camera_roles}"
            )
        self._cam_column = {role: self.index.camera_roles.index(role) for role, _ in canvas_mod.TILES}

        self.text = build_text_source(
            text_mode,
            index=self.index,
            text_dir=text_dir,
            fs=self.fs,
            expect_fingerprint=full.fingerprint,
        )
        if self.text.context_len != context_len:
            raise ValueError(
                f"text source gives context_len {self.text.context_len}, config says {context_len}"
            )
        self.context_len = context_len
        self.d_model = self.text.d_model

        # Shapes come from a real file's preamble, never from a second copy of the
        # numbers in code. spec.json is written by a person, the preamble by a machine.
        probe = twp_probe(self.fs, self.index)
        self.action_dim = probe.action_dim
        self.n_cams = probe.n_cams
        if self.n_cams != len(self.index.cameras):
            raise ValueError(f"file says n_cams={self.n_cams}, spec.json lists {self.index.cameras}")
        if text_mode == "baked" and (probe.text_len, probe.text_dim) != (context_len, self.d_model):
            raise ValueError(
                f"spec.json declares baked text [{context_len},{self.d_model}] but the file "
                f"carries [{probe.text_len},{probe.text_dim}]"
            )
        for name, norm in (("action", self.stats.action), ("proprio", self.stats.proprio)):
            if norm.dim != self.action_dim:
                raise ValueError(
                    f"{name} stats are {norm.dim}-dimensional but the data is {self.action_dim}"
                )

        if episode_weighting == "transitions":
            weights = (self.index.n_frames - 1).astype(np.float64)
        elif episode_weighting == "uniform":
            weights = None
        else:
            raise ValueError(
                f"episode_weighting must be 'transitions' or 'uniform', got {episode_weighting!r}"
            )
        self.episode_weighting = episode_weighting

        self._px_pool = cf.ThreadPoolExecutor(int(decode_threads), thread_name_prefix="twp-px")
        self.pool = TwpPool(
            self.index,
            size=min(int(pool_size), len(self.index)),
            rate=pool_rate,
            seed=seed,
            host_id=host_id,
            n_hosts=n_hosts,
            weights=weights,
            prefetch_threads=pool_prefetch_threads,
            sidecar=self.text.sidecar if self.text.needs_sidecar else None,
        )
        self._step = 0
        print(self.describe(), flush=True)

        import atexit

        atexit.register(self.close)

    # ---------------- provenance

    def data_spec(self) -> dict:
        """Everything a future reader would otherwise have to guess. Printed at startup
        and meant to be written into the checkpoint next to the weights: every entry is a
        convention that has cost somebody a debugging session somewhere."""
        return {
            "canvas": list(canvas_mod.CANVAS_HW),
            "tiles": {role: list(box) for role, box in canvas_mod.TILES},
            "camera_roles": list(self.index.camera_roles),
            "source_color_order": self.source_color_order,
            "color_order_applied": "flipped to RGB" if self.source_color_order == "bgr" else "none",
            "video_frames": self.video_frames,
            "video_stride": self.video_stride,
            "action_horizon": self.action_horizon,
            "action_index_convention": (
                f"action[j] = qpos[min(start+j, T-2) + {self.action_shift}], "
                f"proprio[j] = qpos[min(start+j, T-2)]  (D5, action_shift={self.action_shift})"
            ),
            "normalization": "z-score(global) + clamp(+-5), forward only",
            "stats_source": self.stats.source,
            "stats_sha256": self.stats.sha256,
            "jpeg_decoder": canvas_mod.decoder_name,
            "resize": "bilinear antialias (torch _upsample_bilinear2d_aa weights), float32, W then H",
            "text": self.text.describe(),
            "l3_prefix": self.index.prefix,
            "l3_index_fingerprint": self.index.fingerprint,
            "split": self.split,
            "episodes": len(self.index),
            "windows": self.index.num_windows,
            "episode_weighting": self.episode_weighting,
            "task_filter": None if self.task_filter is None else list(self.task_filter),
            "tasks": len(set(self.index.task_key)),
        }

    def describe(self) -> str:
        spec = self.data_spec()
        head = (
            f"[data] L3 {spec['l3_prefix']} split={self.split} "
            f"episodes={spec['episodes']:,} windows={spec['windows']:,} "
            f"fingerprint={spec['l3_index_fingerprint']}"
        )
        match = (
            f"sum(T-1) over the full set = {self.full_windows:,} "
            + (
                "== official num_transition (same episodes)"
                if self.transition_match
                else f"!= official num_transition {self.stats.num_transition:,} — NOT the official "
                "episode set"
            )
        )
        # The subset line is printed whether or not a filter is set. "no task_filter" in the
        # log of a run somebody later thinks was a subset is worth more than the two words
        # it costs; a line that only appears in one of the two cases proves nothing by its
        # absence, because absence also looks like an older build.
        if self.task_filter is None:
            subset = f"[data] task_filter: none — all {self.full_tasks} tasks in the L3 set"
        else:
            subset = (
                f"[data] task_filter: {len(self.task_filter)} of the {self.full_tasks} task keys -> "
                f"{self.filtered_windows:,} windows before the split "
                f"({100 * self.filtered_windows / max(1, self.full_windows):.1f}% of the full set); "
                f"tasks = {', '.join(self.task_filter)}"
            )
        lines = [
            head,
            f"[data] {match}",
            subset,
            f"[data] canvas {spec['canvas'][0]}x{spec['canvas'][1]} from "
            + ", ".join(f"{r}@{tuple(b)}" for r, b in spec["tiles"].items()),
            f"[data] colour: decoded as {self.source_color_order.upper()}, {spec['color_order_applied']} "
            f"(the eval adapter must NOT flip again)",
            f"[data] window: {spec['video_frames']} frames stride {spec['video_stride']}, "
            f"{spec['action_horizon']} actions; {spec['action_index_convention']}",
            f"[data] {self.stats.describe()}",
            f"[data] {spec['text']}",
            f"[data] decoder={spec['jpeg_decoder']}  weighting={self.episode_weighting}  "
            f"pool={self.pool.K} episodes @ rate {self.pool.rate:g} "
            f"(turnover {self.pool.K / self.pool.rate:,.0f} steps)",
        ]
        return "\n".join(lines)

    # ---------------- lifecycle

    def seek(self, step: int):
        """Resume. **Before** `start()`: the prefill has to begin at the right admission."""
        if self.pool._ready.is_set():
            raise RuntimeError("seek must precede start(); the pool is already filled from the old point")
        self._step = int(step)
        self.pool.start_step = int(step)
        return self

    def start(self):
        self.pool.start()
        return self

    def close(self):
        try:
            self.pool.stop()
        finally:
            self._px_pool.shutdown(wait=False)

    # ---------------- sampling

    def sample_batch(self, rng: np.random.Generator, batch_size: int) -> dict:
        """One host-local batch. Deterministic given (pool seed, step, rng state)."""
        self.pool.advance(self._step)
        self._step += 1

        h, w = canvas_mod.CANVAS_HW
        n_v, n_a, dim = self.video_frames, self.action_horizon, self.action_dim
        video = np.empty((batch_size, 3, n_v, h, w), np.float32)
        action = np.empty((batch_size, n_a, dim), np.float32)
        proprio = np.empty((batch_size, n_a, dim), np.float32)
        context = np.empty((batch_size, self.context_len, self.d_model), np.float32)
        image_is_pad = np.empty((batch_size, n_v), bool)
        action_is_pad = np.empty((batch_size, n_a), bool)

        # Every random draw happens here, in order, before any threading. Drawing inside
        # the workers would make the batch depend on thread completion order: the same
        # (seed, step) would replay differently, silently, and this loader's whole resume
        # story rests on the batch being a pure function of (seed, step).
        drawn = self.pool.draw(rng, batch_size)
        starts = [int(rng.integers(windows.num_windows(a.episode.n_frames))) for a in drawn]

        def _pixels(item):
            b, admission, start = item
            episode = admission.episode
            idx, pad = windows.video_indices(
                start, episode.n_frames, video_frames=n_v, video_stride=self.video_stride
            )
            streams = {
                role: np.stack([canvas_mod.decode_jpeg(episode.jpeg(int(t), col)) for t in idx])
                for role, col in self._cam_column.items()
            }
            frames = canvas_mod.compose_canvas(streams, source_color_order=self.source_color_order)
            video[b] = np.transpose(frames, (3, 0, 1, 2))
            image_is_pad[b] = pad

        list(self._px_pool.map(_pixels, zip(range(batch_size), drawn, starts)))

        instructions = []
        episode_ids = np.empty(batch_size, np.int64)
        for b, (admission, start) in enumerate(zip(drawn, starts)):
            episode = admission.episode
            qpos = episode.actions
            a_idx, a_pad = windows.action_indices(
                start, episode.n_frames, action_horizon=n_a, action_shift=self.action_shift
            )
            p_idx, _ = windows.proprio_indices(start, episode.n_frames, action_horizon=n_a)
            self.stats.action.forward(qpos[a_idx], out=action[b])
            states = episode.states
            if self.aligned and states is None:
                raise ValueError(f"aligned episode {episode.episode_id} is missing its states section")
            self.stats.proprio.forward((states if self.aligned else qpos)[p_idx], out=proprio[b])
            action_is_pad[b] = a_pad
            context[b] = self.text.context_for(admission)
            episode_ids[b] = episode.episode_id
            # The string the conditioning was actually built from: templated for the
            # side table, bare for the baked row. A batch whose logged prompt differs
            # from its embedding is how a text mismatch stays invisible for a week.
            instructions.append(self.text.prompt_for(episode.seen[0]))

        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "context": context,
            # The official zeroes the padded rows and replaces the mask with ones
            # (`robot_video_dataset.py:220-221`); the zeroing is baked into the source.
            "context_mask": np.ones((batch_size, self.context_len), bool),
            "image_is_pad": image_is_pad,
            "action_is_pad": action_is_pad,
            "instructions": instructions,
            "episode_id": episode_ids,
            "window_start": np.asarray(starts, np.int64),
        }


def twp_probe(fs, index: TwpIndex):
    """Read one preamble + section table. Shapes come from the data, never from a second
    copy of the numbers in code."""
    from fastwam_tpu.data import twp

    head = fs.cat_file(index.path(0), start=0, end=twp.PREAMBLE + 8 * twp.ENTRY + 4)
    return twp.Episode(head)


# --------------------------------------------------------------------------- #
# training entry points
# --------------------------------------------------------------------------- #
def make_iterator(config, split: str, mesh=None, *, host_id: int = 0, n_hosts: int = 1, fs=None):
    """Yield batches for `split`, as global `jax.Array`s when a mesh is given.

    Reads `config.data.l3` for everything about the L3 side and `config.data.<split>` for
    the window geometry, so the official numbers stay where they were transcribed.
    """
    node = getattr(config.data, split)
    l3 = config.data.l3
    fs = fs if fs is not None else open_filesystem(l3.prefix)
    stats_path = l3.stats_path.removeprefix("gs://")
    dataset = RobotwinTwpDataset(
        l3.prefix,
        stats_blob=fs.cat_file(stats_path),
        stats_source=l3.stats_path,
        stats_sha256=l3.stats_sha256,
        num_frames=node.num_frames,
        action_video_freq_ratio=node.action_video_freq_ratio,
        video_size=node.video_size,
        action_shift=l3.action_shift,
        source_color_order=l3.source_color_order,
        text_mode=l3.text_mode,
        text_dir=l3.text_dir,
        context_len=node.context_len,
        task_filter=l3.task_filter,
        val_set_proportion=node.val_set_proportion,
        split=split,
        episode_weighting=l3.episode_weighting,
        pool_size=l3.pool_size,
        pool_rate=l3.pool_rate,
        pool_prefetch_threads=l3.pool_prefetch_threads,
        decode_threads=l3.decode_threads,
        seed=config.seed,
        host_id=host_id,
        n_hosts=n_hosts,
        fs=fs,
        expect_index_fingerprint=l3.index_fingerprint,
    ).start()

    rng = np.random.default_rng(stable_seed(config.seed, "batch", split, host_id))
    # The selected TPU shape determines process_count at runtime. Keeping a per-host batch
    # in YAML duplicates cloud topology and is wrong as soon as the same recipe moves from
    # v6e-16 to v6e-64. micro_batch_size is global across hosts, so derive the local share.
    if config.tpu.micro_batch_size % n_hosts:
        raise ValueError(
            f"tpu.micro_batch_size {config.tpu.micro_batch_size} does not divide over "
            f"{n_hosts} TPU workers"
        )
    per_host = config.tpu.micro_batch_size // n_hosts
    try:
        while True:
            batch = dataset.sample_batch(rng, per_host)
            yield batch if mesh is None else global_batches(batch, mesh)
    finally:
        dataset.close()


def global_batches(batch: dict, mesh) -> dict:
    """Host-local numpy -> global `jax.Array`s sharded on the batch axis.

    The ~30 lines of `multihost_dataloading.py` maxdiffusion actually needs, and none of
    the tf.data stack around it. jax is imported here and nowhere above, so the data
    plane stays importable on a host with no accelerator.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec

    sharding = NamedSharding(mesh, PartitionSpec(mesh.axis_names))
    n_local = len(mesh.local_devices)
    out = dict(batch)
    for key in ARRAY_KEYS:
        array = batch[key]
        global_shape = (jax.process_count() * array.shape[0], *array.shape[1:])
        if array.shape[0] % n_local:
            raise ValueError(
                f"batch key {key!r}: host batch {array.shape[0]} does not divide over "
                f"{n_local} local devices"
            )
        shards = np.split(array, n_local, axis=0)
        buffers = jax.device_put(shards, mesh.local_devices)
        out[key] = jax.make_array_from_single_device_arrays(global_shape, sharding, buffers)
    return out
