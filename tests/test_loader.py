"""End-to-end loader gate on a synthetic L3 directory.

The fixture writes real `.twp` files (through the vendored writer, so the format is
exercised and not mocked) with hand-chosen colours and joint poses, which makes every
assertion below computable by hand: which camera went into which tile, which qpos row
became which action row, and where the padding starts.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastwam_tpu.data import canvas as cv  # noqa: E402
from fastwam_tpu.data import twp, windows  # noqa: E402
from fastwam_tpu.data.loader import ARRAY_KEYS, RobotwinTwpDataset, split_episodes  # noqa: E402
from fastwam_tpu.data.pool import LocalFileSystem, TwpIndex, TwpPool  # noqa: E402
from tools.prepare_text_sidetable import dedup, write_sidetable  # noqa: E402

SRC_HW = (48, 64)  # small stand-in for the real 240x320; the resize itself is pinned in test_canvas
CONTEXT_LEN, D_MODEL = 8, 4
CAMERA_ROLES = ["head", "left_wrist", "right_wrist"]
LENGTHS = [40, 12, 55, 33]  # one episode shorter than a window on purpose
FULL_CONFIG = "training/full.example.yaml"


def _jpeg(colour) -> bytes:
    img = np.tile(np.asarray(colour, np.uint8), (*SRC_HW, 1))
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=100, subsampling=0)
    return buf.getvalue()


def episode_colour(episode_id: int, frame: int, cam: int) -> tuple[int, int, int]:
    """A colour that identifies (episode, frame, camera) uniquely, in a channel order
    that is asymmetric so a channel flip is visible."""
    return (
        10 + 3 * cam,
        40 + (frame * 2) % 150,
        200 - 7 * episode_id,
    )


def episode_qpos(episode_id: int, n_frames: int, dim: int = 14) -> np.ndarray:
    """qpos[t, d] = t + d/100 + episode_id*1000 — every entry identifies its own index,
    so a one-frame slip in the action window is arithmetically visible."""
    t = np.arange(n_frames, dtype=np.float32)[:, None]
    d = np.arange(dim, dtype=np.float32)[None, :]
    return (t + d / 100.0 + episode_id * 1000.0).astype(np.float32)


def make_l3(root: Path, lengths=LENGTHS, *, with_text: bool = True, aligned: bool = False) -> Path:
    prefix = root / "pixel_v3"
    prefix.mkdir(parents=True, exist_ok=True)
    manifest = {"layer": "L3-train-pixel", "prefix": "gs://stale/path/nobody/should/read",
                "n_episodes": len(lengths), "episodes": []}
    for episode_id, n_frames in enumerate(lengths):
        blobs = [
            _jpeg(episode_colour(episode_id, f, c)) for f in range(n_frames) for c in range(3)
        ]
        text = (
            np.full((1, CONTEXT_LEN, D_MODEL), episode_id + 1, np.float16) if with_text else None
        )
        raw = twp.build(
            episode_id,
            [len(b) for b in blobs],
            b"".join(blobs),
            episode_qpos(episode_id, n_frames),
            [f"instruction for episode {episode_id}", "an alternative phrasing"],
            ["held-out phrasing"],
            n_cams=3,
            text_emb=text,
            states=episode_qpos(episode_id, n_frames) + 1 if aligned else None,
        )
        (prefix / f"episode_{episode_id:06d}.twp").write_bytes(raw)
        manifest["episodes"].append(
            {
                "episode_id": episode_id,
                "task_key": f"task_{episode_id % 2}",
                "n_frames": n_frames,
                "payload_end": twp.Episode(raw).payload_end(),
                "bad_jpeg": 0,
                "nonfinite_actions": 0,
            }
        )
    (prefix / "MANIFEST.json").write_text(json.dumps(manifest))
    spec = {
        "format": "TWP3",
        "cameras": ["head_camera", "left_camera", "right_camera"],
        "camera_roles": CAMERA_ROLES,
        "color_order": "rgb" if aligned else "bgr_baked",
        "action_shift": 0 if aligned else 1,
        "action_dim": 14,
    }
    if with_text:
        spec["text_emb"] = {"max_len": CONTEXT_LEN, "d_model": D_MODEL, "row_source": "seen[0]",
                            "source": "test", "rows_per_episode": 1}
    (prefix / "spec.json").write_text(json.dumps(spec))
    return prefix


# Constants wide enough that nothing the fixture produces reaches the +-5 clamp (qpos
# spans 0..3055), and offset by 1 between the two fields so a swap is visible.
ACTION_MEAN, ACTION_STD = 1500.0, 1000.0
PROPRIO_MEAN, PROPRIO_STD = 1501.0, 1000.0


def affine(mean: float, std: float):
    """The official's own scale/offset (`utils/normalizer.py:100-103`), so the
    expectations below test which rows were selected, not the arithmetic — that is
    pinned separately in test_normalize.py."""
    scale = np.float32(1.0) / (np.float32(std) + np.float32(1e-8))
    offset = -np.float32(mean) / (np.float32(std) + np.float32(1e-8))
    return scale, offset


def stats_blob(dim: int = 14) -> bytes:
    def field(mean, std):
        return {
            "global_mean": [mean] * dim,
            "global_std": [std] * dim,
            "global_min": [-1e6] * dim,
            "global_max": [1e6] * dim,
        }

    return json.dumps(
        {
            "action": {"default": field(ACTION_MEAN, ACTION_STD)},
            "state": {"default": field(PROPRIO_MEAN, PROPRIO_STD)},
            "num_episodes": len(LENGTHS),
            "num_transition": sum(n - 1 for n in LENGTHS),
        }
    ).encode()


def build_dataset(prefix: Path, **kwargs) -> RobotwinTwpDataset:
    params = dict(
        stats_blob=stats_blob(),
        stats_source="test",
        text_mode="baked",
        context_len=CONTEXT_LEN,
        val_set_proportion=0.0,
        pool_size=len(LENGTHS),
        pool_rate=1.0,
        decode_threads=2,
        pool_prefetch_threads=2,
        seed=1234,
        fs=LocalFileSystem(),
    )
    params.update(kwargs)
    return RobotwinTwpDataset(str(prefix), **params)


@pytest.fixture(scope="module")
def l3(tmp_path_factory) -> Path:
    return make_l3(tmp_path_factory.mktemp("l3"))


# --------------------------------------------------------------------------- #
# index and split
# --------------------------------------------------------------------------- #
def test_index_ignores_the_manifests_stale_prefix(l3):
    """MANIFEST's own `prefix` field is a pre-move path that points at nothing
    (`m4_data_design.md` R5); only the caller's prefix is ever used."""
    index = TwpIndex(str(l3), LocalFileSystem())
    assert index.prefix == str(l3).rstrip("/")
    assert len(index) == len(LENGTHS)
    assert index.num_windows == sum(n - 1 for n in LENGTHS)


def test_aligned_profile_uses_packed_state_and_unshifted_action(tmp_path):
    prefix = make_l3(tmp_path, aligned=True)
    ds = build_dataset(prefix, action_shift=0, source_color_order="rgb").start()
    try:
        batch = ds.sample_batch(np.random.default_rng(0), 1)
        # The packed state is action+1, while its published mean is also action_mean+1.
        # Equal normalized values therefore verify that the separate state section was used.
        np.testing.assert_allclose(batch["proprio"][0, 0], batch["action"][0, 0], atol=1e-6)
        assert batch["video"].shape[2] == 9
    finally:
        ds.close()


def test_legacy_twp_without_a_state_section_is_readable(tmp_path):
    prefix = make_l3(tmp_path)
    episode = twp.Episode((prefix / "episode_000000.twp").read_bytes())
    assert episode.states is None


def test_fingerprint_tracks_the_episode_set(l3):
    index = TwpIndex(str(l3), LocalFileSystem())
    subset = index.select(np.array([0, 2]))
    assert len(subset) == 2
    assert subset.fingerprint != index.fingerprint
    assert subset.num_windows == (LENGTHS[0] - 1) + (LENGTHS[2] - 1)


def test_task_filter_keeps_exactly_the_named_tasks(l3):
    """The fixture alternates `task_0` / `task_1` by episode id, so which episodes survive
    a filter is computable by hand."""
    index = TwpIndex(str(l3), LocalFileSystem())
    only_0 = index.select_tasks(["task_0"])
    assert only_0.episode_id.tolist() == [0, 2]
    assert only_0.num_windows == (LENGTHS[0] - 1) + (LENGTHS[2] - 1)
    assert set(only_0.task_key) == {"task_0"}
    # Naming both is the whole set again — same episodes, and therefore the same
    # fingerprint, whichever order they were asked for in.
    both = index.select_tasks(["task_1", "task_0"])
    assert both.episode_id.tolist() == list(range(len(LENGTHS)))
    assert both.fingerprint == index.fingerprint


def test_a_task_name_that_matches_nothing_raises(l3):
    """The failure this exists to prevent: a typo'd task name quietly shrinking the
    dataset, leaving a run that looks perfectly healthy on a fraction of the data."""
    index = TwpIndex(str(l3), LocalFileSystem())
    with pytest.raises(ValueError, match="does not have"):
        index.select_tasks(["task_0", "task_typo"])
    with pytest.raises(ValueError, match="empty"):
        index.select_tasks([])


def test_task_filter_none_is_byte_for_byte_the_full_set(l3):
    """The default must be a no-op, not merely an equivalent-looking path: every existing
    checkpoint and every published number was produced without a filter."""
    unfiltered = build_dataset(l3)
    filtered_to_everything = build_dataset(l3, task_filter=["task_0", "task_1"])
    try:
        assert unfiltered.data_spec()["task_filter"] is None
        assert unfiltered.index.fingerprint == filtered_to_everything.index.fingerprint
        assert unfiltered.index.num_windows == filtered_to_everything.index.num_windows
        assert unfiltered.data_spec()["tasks"] == 2
    finally:
        unfiltered.close()
        filtered_to_everything.close()


def test_the_filter_runs_before_the_split(l3):
    """Both splits must be drawn from the subset alone. If the split ran first, `val` here
    would be one of the four episodes at large and could easily be a `task_1` one — a val
    set from outside the training distribution, with nothing raising."""
    kept = {0, 2}  # the task_0 episodes
    train = build_dataset(l3, task_filter=["task_0"], val_set_proportion=0.5, split="train")
    val = build_dataset(l3, task_filter=["task_0"], val_set_proportion=0.5, split="val")
    try:
        train_ids = set(train.index.episode_id.tolist())
        val_ids = set(val.index.episode_id.tolist())
        assert train_ids | val_ids == kept
        assert not train_ids & val_ids
        assert len(train_ids) == len(val_ids) == 1  # 50% of the SUBSET, not of the four
        assert train.data_spec()["task_filter"] == ["task_0"]
    finally:
        train.close()
        val.close()


def test_the_split_is_stable_for_a_given_filter(l3):
    """Same filter, same seed, same episodes — the property a resume depends on."""
    first = build_dataset(l3, task_filter=["task_0"], val_set_proportion=0.5)
    again = build_dataset(l3, task_filter=["task_0"], val_set_proportion=0.5)
    try:
        assert first.index.episode_id.tolist() == again.index.episode_id.tolist()
        assert first.index.fingerprint == again.index.fingerprint
    finally:
        first.close()
        again.close()


def test_changing_the_filter_refuses_to_resume(l3):
    """The pool's resume guard keys off the index fingerprint, so a run that changes its
    task list cannot pick up a checkpoint written under the old one and silently train on
    different data."""
    full = build_dataset(l3)
    subset = build_dataset(l3, task_filter=["task_0"])
    try:
        with pytest.raises(ValueError, match="refusing to resume"):
            TwpPool.assert_same(full.pool.state(), subset.pool.state())
    finally:
        full.close()
        subset.close()


def test_a_filter_smaller_than_the_pool_still_runs(l3):
    """`pool_size` is a config constant sized for the full set; narrowing the filter below
    it must clamp rather than raise, or every subset would need its own pool_size."""
    ds = build_dataset(l3, task_filter=["task_0"], pool_size=len(LENGTHS)).start()
    try:
        assert ds.pool.K == 2  # clamped to the two task_0 episodes
        batch = ds.sample_batch(np.random.default_rng(0), 3)
        assert set(batch["episode_id"].tolist()) <= {0, 2}
    finally:
        ds.close()


def test_the_official_dataset_gates_run_on_the_full_set_not_the_subset(l3):
    """A subset run still has to prove it is reading the official episode set before it
    narrows: both gates are computed over the whole manifest."""
    ds = build_dataset(l3, task_filter=["task_0"])
    try:
        assert ds.full_windows == sum(n - 1 for n in LENGTHS)
        assert ds.transition_match
        assert ds.filtered_windows == (LENGTHS[0] - 1) + (LENGTHS[2] - 1)
        # The banner reports the subset against the size of the real set, not a literal 50.
        assert "task_filter: 1 of the 2 task keys" in ds.describe()
    finally:
        ds.close()


def test_split_is_deterministic_and_disjoint():
    train, val = split_episodes(1000, 0.01, seed=42)
    assert len(train) == 990 and len(val) == 10
    assert not set(train.tolist()) & set(val.tolist())
    again, _ = split_episodes(1000, 0.01, seed=42)
    np.testing.assert_array_equal(train, again)
    other, _ = split_episodes(1000, 0.01, seed=43)
    assert not np.array_equal(train, other)
    full, empty = split_episodes(1000, 0.0, seed=42)
    assert len(full) == 1000 and len(empty) == 0


# --------------------------------------------------------------------------- #
# pool
# --------------------------------------------------------------------------- #
def test_pool_contents_are_a_pure_function_of_seed_and_step(l3):
    index = TwpIndex(str(l3), LocalFileSystem())
    a = TwpPool(index, size=2, rate=1.0, seed=7)
    b = TwpPool(index, size=2, rate=1.0, seed=7)
    assert [a.pool_at(s) for s in range(6)] == [b.pool_at(s) for s in range(6)]
    # A different seed has to give a different admission sequence. Compared over a run of
    # steps rather than one, because with a 4-episode fixture two seeds can agree on the
    # first pool by chance.
    other = TwpPool(index, size=2, rate=1.0, seed=8)
    assert [other.pool_at(s) for s in range(12)] != [a.pool_at(s) for s in range(12)]


def test_one_lap_admits_every_episode_exactly_once(l3):
    index = TwpIndex(str(l3), LocalFileSystem())
    for n_hosts in (1, 3):  # a host count that does not divide the episode count
        admitted = [
            TwpPool(index, size=2, rate=1.0, seed=3, host_id=h, n_hosts=n_hosts).admission(j)
            for j in range(len(index))
            for h in range(n_hosts)
        ][: len(index)]
        assert sorted(admitted) == list(range(len(index)))


def test_resume_guard_raises_on_any_drift(l3):
    index = TwpIndex(str(l3), LocalFileSystem())
    pool = TwpPool(index, size=2, rate=1.0, seed=7)
    TwpPool.assert_same(pool.state(), pool.state())
    moved = dict(pool.state(), n_hosts=4)
    with pytest.raises(ValueError, match="refusing to resume"):
        TwpPool.assert_same(pool.state(), moved)


# --------------------------------------------------------------------------- #
# batch shapes and provenance
# --------------------------------------------------------------------------- #
def test_batch_matches_the_official_tensor_spec(l3):
    ds = build_dataset(l3).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(0), 4)
    finally:
        ds.close()
    assert batch["video"].shape == (4, 3, 9, 384, 320) and batch["video"].dtype == np.float32
    assert batch["action"].shape == (4, 32, 14) and batch["action"].dtype == np.float32
    assert batch["proprio"].shape == (4, 32, 14)
    assert batch["context"].shape == (4, CONTEXT_LEN, D_MODEL)
    assert batch["context_mask"].shape == (4, CONTEXT_LEN) and batch["context_mask"].all()
    assert batch["image_is_pad"].shape == (4, 9) and batch["image_is_pad"].dtype == bool
    assert batch["action_is_pad"].shape == (4, 32) and batch["action_is_pad"].dtype == bool
    assert set(ARRAY_KEYS) <= set(batch)
    assert batch["video"].min() >= -1.001 and batch["video"].max() <= 1.001
    assert len(batch["instructions"]) == 4


def test_window_count_is_compared_against_the_official_transition_count(l3):
    ds = build_dataset(l3)
    try:
        assert ds.full_windows == sum(n - 1 for n in LENGTHS)
        assert ds.transition_match  # the fixture's stats say the same number
    finally:
        ds.close()

    bad = json.loads(stats_blob())
    bad["num_transition"] += 1
    ds2 = build_dataset(l3, stats_blob=json.dumps(bad).encode())
    try:
        assert not ds2.transition_match
        assert "NOT the official episode set" in ds2.describe()
    finally:
        ds2.close()


def test_data_spec_records_every_convention(l3):
    ds = build_dataset(l3)
    try:
        spec = ds.data_spec()
    finally:
        ds.close()
    assert spec["canvas"] == [384, 320]
    assert spec["video_frames"] == 9 and spec["video_stride"] == 4 and spec["action_horizon"] == 32
    assert spec["color_order_applied"] == "flipped to RGB"
    assert "action_shift=1" in spec["action_index_convention"]
    assert spec["normalization"] == "z-score(global) + clamp(+-5), forward only"
    assert spec["jpeg_decoder"] == cv.decoder_name
    assert spec["l3_index_fingerprint"] and spec["stats_sha256"]


# --------------------------------------------------------------------------- #
# values
# --------------------------------------------------------------------------- #
def test_action_and_proprio_are_the_hand_computed_rows(l3):
    """Every qpos entry encodes its own (episode, frame, dim), so a one-frame slip shows
    up as an exact arithmetic mismatch rather than a plausible-looking tensor."""
    ds = build_dataset(l3).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(11), 6)
    finally:
        ds.close()

    for b in range(6):
        episode_id = int(batch["episode_id"][b])
        start = int(batch["window_start"][b])
        n_frames = LENGTHS[episode_id]
        qpos = episode_qpos(episode_id, n_frames)

        a_idx, a_pad = windows.action_indices(start, n_frames, action_horizon=32, action_shift=1)
        p_idx, _ = windows.proprio_indices(start, n_frames, action_horizon=32)
        a_scale, a_offset = affine(ACTION_MEAN, ACTION_STD)
        p_scale, p_offset = affine(PROPRIO_MEAN, PROPRIO_STD)
        np.testing.assert_allclose(batch["action"][b], qpos[a_idx] * a_scale + a_offset, atol=1e-6)
        np.testing.assert_allclose(batch["proprio"][b], qpos[p_idx] * p_scale + p_offset, atol=1e-6)
        np.testing.assert_array_equal(batch["action_is_pad"][b], a_pad)

        _, v_pad = windows.video_indices(start, n_frames, video_frames=9, video_stride=4)
        np.testing.assert_array_equal(batch["image_is_pad"][b], v_pad)


def test_action_leads_proprio_by_exactly_one_frame(l3):
    """The D5 convention, read straight off a batch: at an unpadded step the action row
    is the proprio row of the next frame."""
    ds = build_dataset(l3).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(5), 8)
    finally:
        ds.close()
    a_scale, a_offset = affine(ACTION_MEAN, ACTION_STD)
    p_scale, p_offset = affine(PROPRIO_MEAN, PROPRIO_STD)
    checked = 0
    for b in range(8):
        pad = batch["action_is_pad"][b]
        for j in range(31):
            if pad[j] or pad[j + 1]:
                continue
            action_j = (batch["action"][b, j] - a_offset) / a_scale
            proprio_next = (batch["proprio"][b, j + 1] - p_offset) / p_scale
            # Raw qpos runs to ~3000, so a float32 round trip through two different
            # affine maps is good to ~1e-4 relative. A one-frame slip would be 1.0 off.
            np.testing.assert_allclose(action_j, proprio_next, rtol=1e-4)
            checked += 1
    assert checked > 0


def test_action_shift_zero_makes_action_equal_proprio(l3):
    ds = build_dataset(l3, action_shift=0).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(5), 4)
    finally:
        ds.close()
    # same rows, so the only difference left is the 1.0 between the two means
    delta = (PROPRIO_MEAN - ACTION_MEAN) / ACTION_STD
    np.testing.assert_allclose(batch["action"] - batch["proprio"], delta, atol=1e-5)


def test_clamp_bites_only_where_it_should(l3):
    """Stats with a tiny std push everything past +-5; the clamp must be exactly the
    official's saturating one and must not touch anything inside the range."""
    tight = json.loads(stats_blob())
    tight["action"]["default"]["global_std"] = [0.01] * 14
    ds = build_dataset(l3, stats_blob=json.dumps(tight).encode()).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(2), 4)
    finally:
        ds.close()
    assert np.abs(batch["action"]).max() <= 5.0
    assert (np.abs(batch["action"]) == 5.0).any()


def test_each_camera_reaches_its_own_tile_with_the_channels_flipped(l3):
    """Colours are per (episode, frame, camera), so this reads back which stream landed
    where AND which channel order it landed in — the two failures that never raise."""
    ds = build_dataset(l3).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(3), 3)
    finally:
        ds.close()

    for b in range(3):
        episode_id = int(batch["episode_id"][b])
        start = int(batch["window_start"][b])
        idx, _ = windows.video_indices(start, LENGTHS[episode_id], video_frames=9, video_stride=4)
        for j, frame in enumerate(idx.tolist()):
            for cam, (role, (y0, x0, th, tw)) in enumerate(cv.TILES):
                stored = np.asarray(episode_colour(episode_id, frame, cam), np.float32)
                expected = (2.0 * (stored[::-1] / 255.0) - 1.0)  # bgr -> rgb
                got = batch["video"][b, :, j, y0 + th // 2, x0 + tw // 2]
                np.testing.assert_allclose(got, expected, atol=0.03, err_msg=f"{role} frame {frame}")


def test_video_frames_follow_the_stride_4_window(l3):
    """Distinct per-frame colours mean a wrong stride, a reversed window or a repeated
    frame all show up as the wrong green channel."""
    ds = build_dataset(l3).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(9), 4)
    finally:
        ds.close()
    for b in range(4):
        episode_id = int(batch["episode_id"][b])
        start = int(batch["window_start"][b])
        idx, _ = windows.video_indices(start, LENGTHS[episode_id], video_frames=9, video_stride=4)
        for j, frame in enumerate(idx.tolist()):
            green = episode_colour(episode_id, frame, 0)[1] / 255.0 * 2.0 - 1.0
            assert abs(float(batch["video"][b, 1, j, 100, 100]) - green) < 0.03


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def test_two_instances_with_the_same_seed_produce_identical_batches(l3):
    def run():
        ds = build_dataset(l3).start()
        try:
            rng = np.random.default_rng(99)
            return [ds.sample_batch(rng, 3) for _ in range(3)]
        finally:
            ds.close()

    for first, second in zip(run(), run()):
        for key in ARRAY_KEYS:
            np.testing.assert_array_equal(first[key], second[key], err_msg=key)
        np.testing.assert_array_equal(first["window_start"], second["window_start"])
        assert first["instructions"] == second["instructions"]


def test_seek_must_precede_start(l3):
    ds = build_dataset(l3).start()
    try:
        ds.pool.wait_ready(timeout=30)
        with pytest.raises(RuntimeError, match="seek must precede start"):
            ds.seek(10)
    finally:
        ds.close()


# --------------------------------------------------------------------------- #
# text
# --------------------------------------------------------------------------- #
def test_baked_text_comes_from_the_twp_row(l3):
    ds = build_dataset(l3).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(4), 4)
    finally:
        ds.close()
    for b in range(4):
        np.testing.assert_allclose(batch["context"][b], int(batch["episode_id"][b]) + 1)
    assert "BARE instruction" in ds.text.describe()


def test_sidetable_text_is_fetched_per_episode(l3, tmp_path):
    """The side table is written by the real tool function and read by the real loader,
    so the on-disk layout is covered end to end rather than in two agreeing mocks."""
    from fastwam_tpu.data.text import format_prompt

    episode_ids = np.arange(len(LENGTHS))
    prompts = [format_prompt(f"instruction for episode {e}") for e in episode_ids]
    rows, texts = dedup(prompts)
    emb = np.stack(
        [np.full((CONTEXT_LEN, D_MODEL), 100 + r, np.float16) for r in range(len(texts))]
    )
    mask = np.ones((len(texts), CONTEXT_LEN), np.uint8)
    mask[:, -2:] = 0
    emb[:, -2:, :] = 0  # what the tool does: fold the mask into the values
    directory = tmp_path / "text"
    meta = write_sidetable(
        directory, episode_ids=episode_ids, rows=rows, texts=texts, emb=emb, mask=mask,
        encoder="test-encoder", l3_prefix=str(l3),
        l3_index_fingerprint=TwpIndex(str(l3), LocalFileSystem()).fingerprint,
    )
    assert meta["n_rows"] == len(texts) == len(LENGTHS)  # every episode phrases it differently

    ds = build_dataset(l3, text_mode="sidetable", text_dir=str(directory)).start()
    try:
        batch = ds.sample_batch(np.random.default_rng(6), 4)
    finally:
        ds.close()
    for b in range(4):
        row = rows[int(batch["episode_id"][b])]
        np.testing.assert_allclose(batch["context"][b, 0], 100 + row)
        np.testing.assert_allclose(batch["context"][b, -1], 0.0)  # padded rows pre-zeroed
    assert batch["context_mask"].all()  # ...and the mask the model sees is all True


def test_sidetable_with_a_different_template_is_rejected(l3, tmp_path):
    from fastwam_tpu.data.text import TextSideTable

    directory = tmp_path / "wrong_template"
    write_sidetable(
        directory, episode_ids=[0], rows=[0], texts=["hello"],
        emb=np.zeros((1, CONTEXT_LEN, D_MODEL), np.float16),
        mask=np.ones((1, CONTEXT_LEN), np.uint8), encoder="test",
        template="{task}",
    )
    with pytest.raises(ValueError, match="different prompt template"):
        TextSideTable(LocalFileSystem(), str(directory))


def test_text_mode_must_be_named(l3):
    with pytest.raises(ValueError, match="text_mode"):
        build_dataset(l3, text_mode="in_loop")
    with pytest.raises(ValueError, match="text_dir"):
        build_dataset(l3, text_mode="sidetable", text_dir=None)


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def test_wrong_index_fingerprint_refuses_to_run(l3):
    with pytest.raises(ValueError, match="index fingerprint"):
        build_dataset(l3, expect_index_fingerprint="0" * 16)


def test_wrong_stats_sha_refuses_to_run(l3):
    with pytest.raises(ValueError, match="sha256"):
        build_dataset(l3, stats_sha256="0" * 64)


def test_stats_dimension_must_match_the_data(l3):
    from tests.test_normalize import _stats_blob

    with pytest.raises(ValueError, match="stats are 3-dimensional"):
        build_dataset(l3, stats_blob=_stats_blob(dim=3))


def test_missing_camera_role_refuses_to_run(tmp_path):
    prefix = make_l3(tmp_path / "two_cam")
    spec = json.loads((prefix / "spec.json").read_text())
    spec["camera_roles"] = ["head", "left_wrist", "front"]
    (prefix / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="camera roles"):
        build_dataset(prefix)


def test_episode_id_mismatch_is_caught(tmp_path):
    """A file whose name and content disagree — the pool passes `expect_id` for exactly
    this, because the alternative is training on silently mislabelled episodes."""
    prefix = make_l3(tmp_path / "swapped")
    a, b = prefix / "episode_000000.twp", prefix / "episode_000001.twp"
    a.write_bytes(b.read_bytes())
    ds = build_dataset(prefix)
    with pytest.raises(ValueError, match="集号错位|episode"):
        ds.pool._fetch(0)
    ds.close()


# --------------------------------------------------------------------------- #
# training entry points
# --------------------------------------------------------------------------- #
def _local_config(l3: Path, stats_file: Path, batch_size: int = 2):
    """The real Config, repointed at the fixture, so `make_iterator`'s config plumbing is
    exercised against the same dataclasses the TPU run will hand it."""
    import dataclasses

    from fastwam_tpu import load_config

    cfg = load_config(FULL_CONFIG)
    l3_cfg = dataclasses.replace(
        cfg.data.l3,
        prefix=str(l3),
        index_fingerprint=None,
        stats_path=str(stats_file),
        stats_sha256=None,
        text_mode="baked",
        text_dir=None,
        pool_size=len(LENGTHS),
        pool_rate=1.0,
        decode_threads=2,
        pool_prefetch_threads=2,
    )
    node = dataclasses.replace(cfg.data.train, context_len=CONTEXT_LEN, val_set_proportion=0.0)
    data = dataclasses.replace(cfg.data, train=node, l3=l3_cfg)
    tpu = dataclasses.replace(
        cfg.tpu, global_batch_size=batch_size, micro_batch_size=batch_size
    )
    return dataclasses.replace(cfg, data=data, tpu=tpu)


def test_make_iterator_reads_the_config(l3, tmp_path):
    from fastwam_tpu.data.loader import make_iterator

    stats_file = tmp_path / "dataset_stats.json"
    stats_file.write_bytes(stats_blob())
    cfg = _local_config(l3, stats_file)
    it = make_iterator(cfg, "train", fs=LocalFileSystem())
    try:
        batch = next(it)
    finally:
        it.close()
    per_host = cfg.tpu.micro_batch_size
    assert batch["video"].shape == (per_host, 3, 9, 384, 320)
    assert batch["action"].shape == (per_host, 32, 14)
    assert batch["context"].shape == (per_host, CONTEXT_LEN, D_MODEL)


def test_global_batches_shards_on_the_batch_axis(l3, tmp_path):
    """The ~30 lines borrowed from maxdiffusion's multihost_dataloading: host-local
    numpy in, one global jax.Array per tensor key out, sharded over every mesh axis."""
    jax = pytest.importorskip("jax")
    from jax.sharding import Mesh

    from fastwam_tpu.data.loader import make_iterator

    stats_file = tmp_path / "dataset_stats.json"
    stats_file.write_bytes(stats_blob())
    cfg = _local_config(l3, stats_file)
    mesh = Mesh(np.array(jax.devices()).reshape(1, 1), ("data", "fsdp"))
    it = make_iterator(cfg, "train", mesh, fs=LocalFileSystem())
    try:
        batch = next(it)
    finally:
        it.close()
    for key in ARRAY_KEYS:
        assert isinstance(batch[key], jax.Array), key
        assert batch[key].shape[0] == cfg.tpu.micro_batch_size
    assert isinstance(batch["instructions"], list)  # provenance stays on the host


def test_sidetable_missing_a_row_fails_at_startup_not_in_the_fetch_thread(l3, tmp_path):
    """The pool retries fetch failures forever by design, so a permanently missing text
    row has to be caught before the pool starts or the run hangs instead of crashing."""
    directory = tmp_path / "short_table"
    write_sidetable(
        directory, episode_ids=[0, 1], rows=[0, 0], texts=["only one prompt"],
        emb=np.zeros((1, CONTEXT_LEN, D_MODEL), np.float16),
        mask=np.ones((1, CONTEXT_LEN), np.uint8), encoder="test",
        l3_index_fingerprint=TwpIndex(str(l3), LocalFileSystem()).fingerprint,
    )
    with pytest.raises(ValueError, match="has no row for"):
        build_dataset(l3, text_mode="sidetable", text_dir=str(directory))


def test_sidetable_built_against_another_l3_is_rejected(l3, tmp_path):
    directory = tmp_path / "other_l3"
    write_sidetable(
        directory, episode_ids=list(range(len(LENGTHS))), rows=[0] * len(LENGTHS),
        texts=["a prompt"], emb=np.zeros((1, CONTEXT_LEN, D_MODEL), np.float16),
        mask=np.ones((1, CONTEXT_LEN), np.uint8), encoder="test",
        l3_index_fingerprint="0123456789abcdef",
    )
    with pytest.raises(ValueError, match="but this run reads"):
        build_dataset(l3, text_mode="sidetable", text_dir=str(directory))


def test_sidetable_without_a_recorded_fingerprint_is_rejected(l3, tmp_path):
    """"Built by hand, provenance unknown" is not a pass — an unprovable table is
    exactly the case the fingerprint exists to catch."""
    directory = tmp_path / "no_fingerprint"
    write_sidetable(
        directory, episode_ids=list(range(len(LENGTHS))), rows=[0] * len(LENGTHS),
        texts=["a prompt"], emb=np.zeros((1, CONTEXT_LEN, D_MODEL), np.float16),
        mask=np.ones((1, CONTEXT_LEN), np.uint8), encoder="test",
    )
    with pytest.raises(ValueError, match="no L3 fingerprint at all"):
        build_dataset(l3, text_mode="sidetable", text_dir=str(directory))


def test_baked_text_geometry_must_match_the_file(l3, tmp_path):
    """spec.json is written by a person and the preamble by a machine; when they
    disagree the file wins and the run stops."""
    prefix = make_l3(tmp_path / "text_drift")
    spec = json.loads((prefix / "spec.json").read_text())
    spec["text_emb"]["d_model"] = D_MODEL + 1
    (prefix / "spec.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="but the file carries"):
        build_dataset(prefix, context_len=CONTEXT_LEN)


def test_logged_prompt_matches_what_was_encoded(l3, tmp_path):
    """`instructions` is provenance, so it has to be the string the embedding came from:
    the bare instruction under `baked`, the templated one under `sidetable`."""
    from fastwam_tpu.data.text import format_prompt

    ds = build_dataset(l3).start()
    try:
        bare = ds.sample_batch(np.random.default_rng(1), 2)["instructions"]
    finally:
        ds.close()
    assert all(p.startswith("instruction for episode") for p in bare)

    episode_ids = np.arange(len(LENGTHS))
    prompts = [format_prompt(f"instruction for episode {e}") for e in episode_ids]
    rows, texts = dedup(prompts)
    directory = tmp_path / "prompt_text"
    write_sidetable(
        directory, episode_ids=episode_ids, rows=rows, texts=texts,
        emb=np.zeros((len(texts), CONTEXT_LEN, D_MODEL), np.float16),
        mask=np.ones((len(texts), CONTEXT_LEN), np.uint8), encoder="test",
        l3_index_fingerprint=TwpIndex(str(l3), LocalFileSystem()).fingerprint,
    )
    ds = build_dataset(l3, text_mode="sidetable", text_dir=str(directory)).start()
    try:
        templated = ds.sample_batch(np.random.default_rng(1), 2)["instructions"]
    finally:
        ds.close()
    assert all(p.startswith("A video recorded from a robot's point of view") for p in templated)
    assert templated != bare
