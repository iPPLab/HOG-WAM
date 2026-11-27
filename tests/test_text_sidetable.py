"""`tools/prepare_text_sidetable.py` without a GPU.

The UMT5 call is injected, so everything around it — dedup, shard resume, the
mask-into-values fold, the on-disk layout — is covered before anyone spends 10 minutes
of 3090 time, and covered against the real reader in `data/text.py` rather than a mock.
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastwam_tpu.data import twp  # noqa: E402
from fastwam_tpu.data.pool import LocalFileSystem, TwpIndex  # noqa: E402
from fastwam_tpu.data.text import DEFAULT_PROMPT, TextSideTable, format_prompt  # noqa: E402
from tests.test_loader import CONTEXT_LEN, D_MODEL, LENGTHS, make_l3  # noqa: E402
from tools.prepare_text_sidetable import (  # noqa: E402
    collect,
    dedup,
    encode_all,
    read_first_instruction,
    write_sidetable,
)

MAX_LEN, DIM = 6, 4


def stub_encoder(true_len=4):
    """Deterministic stand-in. The value depends only on the TEXT, never on when it was
    encoded, so a resumed run and a fresh run have to agree exactly — a stub keyed on
    call order would hide precisely the bug resume can introduce."""
    calls = []

    def encode(texts):
        calls.append(list(texts))
        ctx = np.stack([np.full((MAX_LEN, DIM), float(len(t) + ord(t[-1]))) for t in texts])
        mask = np.zeros((len(texts), MAX_LEN), bool)
        mask[:, :true_len] = True
        return ctx.astype(np.float32), mask

    return encode, calls


def test_dedup_is_first_appearance_order():
    rows, texts = dedup(["b", "a", "b", "c", "a"])
    assert texts == ["b", "a", "c"]
    assert rows == [0, 1, 0, 2, 1]


def test_encode_folds_the_mask_into_the_values(tmp_path):
    """The official does `context[~mask] = 0` and then hands the model an all-True mask
    (`robot_video_dataset.py:220-221`), so the zeros have to be baked in at build time —
    the loader has no mask left to apply them with."""
    encode, _ = stub_encoder(true_len=4)
    emb, mask = encode_all(
        ["a", "b", "c"], encode, max_len=MAX_LEN, d_model=DIM, shard_rows=8,
        shard_dir=tmp_path / "shards", log=lambda *_: None,
    )
    assert emb.shape == (3, MAX_LEN, DIM) and emb.dtype == np.float16
    assert (emb[:, :4] != 0).all()
    assert (emb[:, 4:] == 0).all()
    assert mask[:, :4].all() and not mask[:, 4:].any()


def test_encode_resumes_from_shards_without_re_encoding(tmp_path):
    encode, calls = stub_encoder()
    kwargs = dict(max_len=MAX_LEN, d_model=DIM, shard_rows=2, shard_dir=tmp_path / "shards",
                  log=lambda *_: None)
    texts = [f"prompt {i}" for i in range(5)]
    first, _ = encode_all(texts, encode, **kwargs)
    assert len(calls) == 3  # 2 + 2 + 1

    # Delete one shard: only that one is re-encoded, and the result is unchanged.
    (tmp_path / "shards" / "shard_00000002.npz").unlink()
    encode2, calls2 = stub_encoder()
    second, _ = encode_all(texts, encode2, **kwargs)
    assert calls2 == [["prompt 2", "prompt 3"]]
    np.testing.assert_array_equal(first, second)


def test_encode_rejects_a_wrong_shaped_encoder(tmp_path):
    def bad(texts):
        n = len(texts)
        return np.zeros((n, MAX_LEN + 1, DIM), np.float32), np.ones((n, MAX_LEN + 1), bool)

    with pytest.raises(ValueError, match="encoder returned"):
        encode_all(["a"], bad, max_len=MAX_LEN, d_model=DIM, shard_rows=8,
                   shard_dir=tmp_path / "shards", log=lambda *_: None)


def test_write_then_read_round_trips_through_the_real_reader(tmp_path):
    episode_ids = [0, 3, 7]  # deliberately sparse: the row array is dense over episode id
    rows = [0, 1, 0]
    texts = ["one", "two"]
    emb = np.stack([np.full((MAX_LEN, DIM), 10.0 + r) for r in range(2)]).astype(np.float16)
    mask = np.ones((2, MAX_LEN), np.uint8)
    meta = write_sidetable(
        tmp_path, episode_ids=episode_ids, rows=rows, texts=texts, emb=emb, mask=mask,
        encoder="stub", l3_index_fingerprint="abc123",
    )
    assert meta["n_rows"] == 2 and meta["n_episode_slots"] == 8
    assert meta["template"] == DEFAULT_PROMPT

    table = TextSideTable(LocalFileSystem(), str(tmp_path), expect_fingerprint="abc123")
    assert [table.row_for(e) for e in episode_ids] == rows
    np.testing.assert_allclose(table.sidecar(3), 11.0)
    with pytest.raises(KeyError, match="no row"):
        table.row_for(1)
    table.assert_covers(episode_ids)
    with pytest.raises(ValueError, match="has no row for"):
        table.assert_covers([0, 1])


def test_write_rejects_inconsistent_inputs(tmp_path):
    emb = np.zeros((1, MAX_LEN, DIM), np.float16)
    mask = np.ones((1, MAX_LEN), np.uint8)
    with pytest.raises(ValueError, match="must line up"):
        write_sidetable(tmp_path, episode_ids=[0, 1], rows=[0], texts=["x"], emb=emb, mask=mask,
                        encoder="stub")
    with pytest.raises(ValueError, match="outside the embedding table"):
        write_sidetable(tmp_path, episode_ids=[0], rows=[5], texts=["x"], emb=emb, mask=mask,
                        encoder="stub")
    with pytest.raises(ValueError, match="do not match"):
        write_sidetable(tmp_path, episode_ids=[0], rows=[0], texts=["x", "y"], emb=emb, mask=mask,
                        encoder="stub")


# --------------------------------------------------------------------------- #
# collect, against real .twp files
# --------------------------------------------------------------------------- #
def test_read_first_instruction_uses_two_small_range_reads(tmp_path):
    """It must not pull the 12 MiB payload to fetch a string, so the read has to stop at
    the end of the SEEN section."""
    prefix = make_l3(tmp_path / "l3")
    fs = LocalFileSystem()
    index = TwpIndex(str(prefix), fs)
    assert read_first_instruction(fs, index, 0) == "instruction for episode 0"

    head = fs.cat_file(index.path(0), start=0, end=twp.PREAMBLE + 8 * twp.ENTRY)
    n_sec = struct.unpack("<I", head[8:12])[0]
    ends = {}
    for k in range(n_sec):
        tag, off, ln = struct.unpack(
            "<IQQ", head[twp.PREAMBLE + k * twp.ENTRY : twp.PREAMBLE + (k + 1) * twp.ENTRY]
        )
        ends[tag] = off + ln
    assert ends[twp.SEEN] < ends[twp.PAYLOAD]


def test_collect_templates_every_episode(tmp_path):
    payload = collect(str(make_l3(tmp_path / "l3")), tmp_path / "out", workers=2)
    assert payload["episode_ids"] == list(range(len(LENGTHS)))
    assert payload["template"] == DEFAULT_PROMPT
    assert payload["l3_index_fingerprint"]
    assert payload["texts"][0] == format_prompt("instruction for episode 0")
    assert len(payload["texts"]) == len(LENGTHS)  # every fixture episode phrases it differently

    written = json.loads((tmp_path / "out" / "prompts.json").read_text())
    assert written == payload
    assert (CONTEXT_LEN, D_MODEL) == (8, 4)  # fixture geometry, kept in one place
