"""Where a sample's `context [128, 4096]` comes from.

The official conditions on **one instruction per episode, wrapped in a fixed template**
(`robot_video_dataset.py:23, 216`):

    "A video recorded from a robot's point of view executing the following instruction: {task}"

UMT5-XXL encodes the formatted prompt. The model sees exactly zero padding and
an all-true mask: `context[~mask] = 0; mask = ones` (`:218-221`).

The configuration selects one text source:

* `sidetable` looks up the formatted instruction in the separately prepared
  table. Use `tools/prepare_robotwin_text.py` for the release recipes.
* `baked` reads the embedding stored inside each legacy TWP episode. It can
  represent a bare instruction and is not interchangeable with the side table.
"""

import json

import numpy as np

__all__ = ["DEFAULT_PROMPT", "BakedText", "TextSideTable", "build_text_source", "format_prompt"]

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)

META_NAME = "meta.json"
ROWS_NAME = "rows.i32"
EMB_NAME = "emb.f16"
MASK_NAME = "mask.u8"
TEXTS_NAME = "texts.json"


def format_prompt(task: str) -> str:
    """The official template. One place, so the tool and any eval adapter agree."""
    return DEFAULT_PROMPT.format(task=task)


class BakedText:
    """The `TEXT_EMB` row that already travels inside each `.twp`."""

    mode = "baked"
    needs_sidecar = False

    def __init__(self, spec: dict, prefix: str):
        section = spec.get("text_emb")
        if not section:
            raise ValueError(
                f"{prefix}/spec.json has no text_emb section — this L3 predates baked text; "
                "rebuild it or use text_mode='sidetable'"
            )
        self.context_len = int(section["max_len"])
        self.d_model = int(section["d_model"])
        self.row_source = section.get("row_source", "?")
        self.encoder = section.get("source", "?")

    def describe(self) -> str:
        return (
            f"text=baked  [{self.context_len},{self.d_model}] from the .twp TEXT_EMB section "
            f"(row={self.row_source}, encoder={self.encoder}) — BARE instruction, no official "
            f"template, Wan2.1 encoder. Not comparable with the official checkpoint's "
            f"conditioning; use text_mode='sidetable' for the B line."
        )

    def sidecar(self, episode_id: int):
        return None

    def prompt_for(self, instruction: str) -> str:
        """What was actually encoded — the bare instruction, no template."""
        return instruction

    def context_for(self, admission) -> np.ndarray:
        emb = admission.episode.text_emb
        if emb.shape[0] < 1:
            raise ValueError(f"episode {admission.episode.episode_id} carries no baked text row")
        return np.asarray(emb[0], np.float32)


class TextSideTable:
    """Per-episode embeddings in a side object, built by `tools/prepare_text_sidetable.py`.

    Layout under `<dir>/`:
        meta.json   geometry, encoder id, template, provenance, sha256 of emb.f16
        rows.i32    int32[n_episode_slots], episode_id -> row, -1 where absent
        emb.f16     float16[n_rows, max_len, d_model], padded positions already zeroed
        mask.u8     uint8[n_rows, max_len], kept for verification only
        texts.json  the n_rows prompt strings, for provenance

    One row is 1 MiB at [128, 4096], so it is fetched when its episode is admitted to the
    pool (one GET per admission, ~1 per step) rather than per sample.
    """

    mode = "sidetable"
    needs_sidecar = True

    def __init__(self, fs, directory: str, *, expect_fingerprint: str | None = None):
        self.fs = fs
        self.dir = directory.removeprefix("gs://").rstrip("/")
        self.meta = json.loads(fs.cat_file(f"{self.dir}/{META_NAME}"))
        for key in ("max_len", "d_model", "dtype", "n_rows", "template", "encoder", "granularity"):
            if key not in self.meta:
                raise ValueError(f"{directory}/{META_NAME}: missing {key!r}")
        if self.meta["dtype"] != "float16":
            raise ValueError(f"{directory}: only float16 rows are supported, got {self.meta['dtype']!r}")
        if self.meta["template"] != DEFAULT_PROMPT:
            raise ValueError(
                f"{directory}: the table was built with a different prompt template:\n"
                f"  table: {self.meta['template']!r}\n  code : {DEFAULT_PROMPT!r}"
            )
        self.context_len = int(self.meta["max_len"])
        self.d_model = int(self.meta["d_model"])
        self.n_rows = int(self.meta["n_rows"])
        self._row_bytes = self.context_len * self.d_model * 2
        fingerprint = self.meta.get("l3_index_fingerprint")
        if expect_fingerprint is not None and fingerprint != expect_fingerprint:
            built_against = (
                "no L3 fingerprint at all (built before the field existed, or by hand)"
                if not fingerprint
                else f"L3 index {fingerprint!r}"
            )
            raise ValueError(
                f"{directory}: the table records {built_against} but this run reads "
                f"{expect_fingerprint!r}; episode ids may not line up. Rebuild it with "
                f"tools/prepare_text_sidetable.py collect against this L3."
            )
        rows = np.frombuffer(fs.cat_file(f"{self.dir}/{ROWS_NAME}"), np.int32)
        self._rows = rows
        if (rows >= self.n_rows).any():
            raise ValueError(f"{directory}/{ROWS_NAME}: a row index exceeds n_rows={self.n_rows}")
        # Metadata/rows can arrive before the multi-GB embedding object. Detect an
        # incomplete publication on the main thread, not in asynchronous prefill,
        # where an exception would otherwise leave wait_ready() blocked forever.
        embedding_path = f"{self.dir}/{EMB_NAME}"
        if not fs.exists(embedding_path):
            raise FileNotFoundError(
                f"incomplete text side table: missing {embedding_path}; "
                "publish and verify the original embedding object before training"
            )

    def describe(self) -> str:
        return (
            f"text=sidetable  [{self.context_len},{self.d_model}] {self.meta['granularity']} "
            f"from {self.dir} (encoder={self.meta['encoder']}, rows={self.n_rows}, "
            f"official template applied, padded positions zeroed)"
        )

    def row_for(self, episode_id: int) -> int:
        if episode_id < 0 or episode_id >= self._rows.shape[0]:
            raise KeyError(f"episode {episode_id} is outside the side table's {self._rows.shape[0]} slots")
        row = int(self._rows[episode_id])
        if row < 0:
            raise KeyError(f"episode {episode_id} has no row in the text side table")
        return row

    def assert_covers(self, episode_ids) -> None:
        """Fail at startup if any episode lacks a row.

        Not optional: `sidecar` runs on the pool's fetch thread, and that thread retries
        every failure forever on purpose (skipping an admission would fork the replay
        from the recorded one). A permanently missing row would therefore not crash, it
        would hang the run behind a log line every five seconds.
        """
        ids = np.asarray(episode_ids, np.int64)
        outside = ids[(ids < 0) | (ids >= self._rows.shape[0])]
        missing = ids[(ids >= 0) & (ids < self._rows.shape[0])]
        missing = missing[self._rows[missing] < 0]
        bad = np.concatenate([outside, missing])
        if bad.size:
            raise ValueError(
                f"the text side table at {self.dir} has no row for {bad.size} of {ids.size} "
                f"episodes (first few: {bad[:5].tolist()}); rebuild it against this L3"
            )

    def sidecar(self, episode_id: int) -> np.ndarray:
        """Fetch one row. Called by the pool as the episode is admitted."""
        row = self.row_for(episode_id)
        start = row * self._row_bytes
        blob = self.fs.cat_file(f"{self.dir}/{EMB_NAME}", start=start, end=start + self._row_bytes)
        if len(blob) != self._row_bytes:
            raise ValueError(
                f"short read for episode {episode_id} row {row}: {len(blob)} of {self._row_bytes} bytes"
            )
        return np.frombuffer(blob, np.float16).reshape(self.context_len, self.d_model)

    def prompt_for(self, instruction: str) -> str:
        """What was actually encoded — the instruction inside the official template."""
        return format_prompt(instruction)

    def context_for(self, admission) -> np.ndarray:
        if admission.sidecar is None:
            raise RuntimeError("the pool was built without the text sidecar; wire build_text_source in")
        return np.asarray(admission.sidecar, np.float32)


def build_text_source(mode: str, *, index, text_dir: str | None, fs, expect_fingerprint: str | None = None):
    """Config `text_mode` -> a text source. Unknown mode raises; there is no default.

    `expect_fingerprint` is the FULL L3 index's fingerprint, not the split's: the side
    table is built over every episode, so comparing it against a train-only subset would
    fail for the wrong reason.
    """
    if mode == "baked":
        return BakedText(index.spec, index.prefix)
    if mode == "sidetable":
        if not text_dir:
            raise ValueError("text_mode='sidetable' needs data.l3.text_dir")
        table = TextSideTable(fs, text_dir, expect_fingerprint=expect_fingerprint)
        table.assert_covers(index.episode_id)
        return table
    raise ValueError(f"text_mode must be 'sidetable' or 'baked', got {mode!r}")
