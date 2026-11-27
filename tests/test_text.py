"""The deployment-side text contract: the template, the cleaning rule, the disk cache.

No torch here. `Umt5Encoder` needs 11 GB of weights and a venv this repo does not have,
so unit tests verify the prompt/cache contract without requiring private assets.
Regenerate the paper's text embeddings from the pinned public encoder. What IS unit
tested is everything around it, because that is where the wiring mistakes live:

  * the template string is the official one, character for character;
  * the cache round-trips, and the zero-padding convention is reconstructed on load
    rather than stored;
  * the cache key covers the encoder repo and max_len, so a cache built for a different
    setting cannot serve a run silently — it would have exactly the right shape.
"""

import json

import numpy as np
import pytest

from fastwam_tpu.inference import text as textmod


def test_the_template_is_the_official_string():
    """`robot_video_dataset.py:23`. It is duplicated in `data/` (training) and here
    (deployment), and the two conditioning paths must be the same sentence."""
    from fastwam_tpu.data.text import DEFAULT_PROMPT as TRAINING_PROMPT

    assert textmod.DEFAULT_PROMPT == TRAINING_PROMPT
    assert textmod.DEFAULT_PROMPT == (
        "A video recorded from a robot's point of view executing the following "
        "instruction: {task}"
    )
    assert textmod.build_prompt("Pick up the shoe").endswith("instruction: Pick up the shoe")


def test_cleaning_collapses_whitespace_and_unescapes_html():
    """`whitespace_clean(basic_clean(text))` — the tokenizer's `clean="whitespace"`."""
    assert textmod.clean_text("  a   b\n\tc  ") == "a b c"
    assert textmod.clean_text("a &amp; b") == "a & b"
    # Double unescape, as the reference does it (`html.unescape` twice).
    assert textmod.clean_text("a &amp;amp; b") == "a & b"


def test_build_prompt_refuses_an_empty_instruction():
    for bad in ("", "   ", None, 42):
        with pytest.raises(textmod.TextError):
            textmod.build_prompt(bad)


def test_prompt_key_covers_the_encoder_and_the_length():
    """A cache built at max_len 512, or with another encoder, must not serve this run.
    Keying on the prompt alone would let it: the arrays are the right shape either way."""
    prompt = textmod.build_prompt("Lift the pot")
    base = textmod.prompt_key(prompt, repo="A", max_len=128)
    assert base != textmod.prompt_key(prompt, repo="B", max_len=128)
    assert base != textmod.prompt_key(prompt, repo="A", max_len=512)
    assert base == textmod.prompt_key(prompt, repo="A", max_len=128)
    assert len(base) == 64


def test_cache_round_trip_reconstructs_the_zero_padding(tmp_path):
    cache = textmod.TextEmbeddingCache(str(tmp_path), max_len=8, text_dim=4)
    prompt = textmod.build_prompt("Lift the pot")
    rows = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) + 1.0

    assert cache.get(prompt) is None
    stored = cache.put(prompt, rows, 3)
    assert stored.shape == (8, 4)
    assert stored.dtype == np.float32
    assert np.allclose(stored[:3], rows)
    # The padded rows are zeros BY THE CONVENTION, so they are reconstructed, not stored.
    assert not stored[3:].any()

    fresh = textmod.TextEmbeddingCache(str(tmp_path), max_len=8, text_dim=4)
    hit = fresh.get(prompt)
    assert hit is not None and np.allclose(hit, stored)
    assert fresh.hits == 1 and fresh.misses == 0

    with np.load(cache.path_for(prompt), allow_pickle=False) as payload:
        assert payload["rows"].shape == (3, 4), "only the real rows are on disk"
        assert json.loads(str(payload["meta"]))["prompt"] == prompt


def test_cache_mask_is_all_true_at_the_declared_length(tmp_path):
    cache = textmod.TextEmbeddingCache(str(tmp_path), max_len=128, text_dim=4096)
    mask = cache.mask()
    assert mask.shape == (128,) and mask.dtype == np.bool_ and mask.all()


def test_cache_treats_a_corrupt_or_foreign_entry_as_a_miss(tmp_path):
    """A cache that can wedge a campaign is worse than no cache — but it must never
    serve the WRONG embedding either, so a meta mismatch is a miss, not a hit."""
    cache = textmod.TextEmbeddingCache(str(tmp_path), max_len=8, text_dim=4)
    prompt = textmod.build_prompt("Lift the pot")
    cache.put(prompt, np.ones((2, 4), np.float32), 2)

    path = cache.path_for(prompt)
    with open(path, "wb") as handle:
        handle.write(b"not an npz")
    assert textmod.TextEmbeddingCache(str(tmp_path), max_len=8, text_dim=4).get(prompt) is None

    # Right file name, wrong contents: a meta that claims another prompt must not be served.
    np.savez(path, rows=np.ones((2, 4), np.float16), meta=np.asarray(json.dumps(
        {"prompt": "something else", "repo": cache.repo, "max_len": 8, "seq_len": 2}
    )))
    assert textmod.TextEmbeddingCache(str(tmp_path), max_len=8, text_dim=4).get(prompt) is None


def test_cache_rejects_a_bad_put(tmp_path):
    cache = textmod.TextEmbeddingCache(str(tmp_path), max_len=8, text_dim=4)
    prompt = textmod.build_prompt("Lift the pot")
    with pytest.raises(textmod.TextError):
        cache.put(prompt, np.ones((2, 5), np.float32), 2)  # wrong width
    with pytest.raises(textmod.TextError):
        cache.put(prompt, np.ones((2, 4), np.float32), 9)  # seq_len past max_len
    with pytest.raises(textmod.TextError):
        cache.put(prompt, np.full((2, 4), np.nan, np.float32), 2)
