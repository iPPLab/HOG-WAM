"""Prepare deduplicated UMT5 embeddings for the RoboTwin training instructions."""

import argparse
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastwam_tpu.data import twp  # noqa: E402
from fastwam_tpu.data.pool import TwpIndex, open_filesystem  # noqa: E402
from fastwam_tpu.data.text import (  # noqa: E402
    DEFAULT_PROMPT,
    EMB_NAME,
    MASK_NAME,
    META_NAME,
    ROWS_NAME,
    TEXTS_NAME,
    format_prompt,
)
from fastwam_tpu.inference.text import DEFAULT_REPO  # noqa: E402

# Share the text encoder with the RoboTwin inference service.
DEFAULT_ENCODER = DEFAULT_REPO
DEFAULT_MAX_LEN = 128
DEFAULT_D_MODEL = 4096
SHARD_ROWS = 512
PROMPTS_NAME = "prompts.json"


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def read_first_instruction(fs, index: TwpIndex, i: int) -> str:
    """`seen[0]` of one episode, in two small range reads.

    The `.twp` section table sits at a fixed offset, so the first read learns where the
    instructions end and the second takes exactly that much. Reading the whole episode
    would be 12 MiB for a string.
    """
    head_len = twp.PREAMBLE + 8 * twp.ENTRY
    head = fs.cat_file(index.path(i), start=0, end=head_len)
    n_sec = struct.unpack("<I", head[8:12])[0]
    end = 0
    for k in range(n_sec):
        entry = head[twp.PREAMBLE + k * twp.ENTRY : twp.PREAMBLE + (k + 1) * twp.ENTRY]
        tag, off, ln = struct.unpack("<IQQ", entry)
        if tag == twp.SEEN:
            end = off + ln
    if end == 0:
        raise ValueError(f"{index.path(i)}: no SEEN section")
    blob = fs.cat_file(index.path(i), start=0, end=end)
    episode = twp.Episode(blob, expect_id=int(index.episode_id[i]))
    seen = episode.seen
    if not seen:
        raise ValueError(f"{index.path(i)}: empty seen list")
    return seen[0]


def collect(l3_prefix: str, out_dir: Path, *, workers: int = 32, limit: int | None = None) -> dict:
    """L3 -> `prompts.json`: one templated prompt per episode, plus the dedup mapping."""
    import concurrent.futures as cf

    fs = open_filesystem(l3_prefix)
    index = TwpIndex(l3_prefix, fs)
    n = len(index) if limit is None else min(limit, len(index))
    print(f"[collect] {n} episodes from {l3_prefix} (fingerprint {index.fingerprint})", flush=True)

    instructions: list[str] = [""] * n
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, text in zip(range(n), ex.map(lambda i: read_first_instruction(fs, index, i), range(n))):
            instructions[i] = text
            if (i + 1) % 2000 == 0:
                print(f"[collect] {i + 1}/{n}", flush=True)

    prompts = [format_prompt(t) for t in instructions]
    rows, unique = dedup(prompts)
    payload = {
        "l3_prefix": index.prefix,
        "l3_index_fingerprint": index.fingerprint,
        "template": DEFAULT_PROMPT,
        "granularity": "per_episode",
        "row_source": "seen[0]",
        "episode_ids": [int(v) for v in index.episode_id[:n]],
        "rows": [int(r) for r in rows],
        "texts": unique,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(out_dir / PROMPTS_NAME, json.dumps(payload, ensure_ascii=False).encode())
    print(
        f"[collect] {n} episodes -> {len(unique)} distinct prompts "
        f"({100 * len(unique) / max(1, n):.1f}%) -> {out_dir / PROMPTS_NAME}",
        flush=True,
    )
    return payload


def dedup(prompts: list[str]) -> tuple[list[int], list[str]]:
    """Prompts -> (row per prompt, unique prompts). Order is first appearance, so the
    mapping is a deterministic function of the episode order."""
    rows, unique, seen = [], [], {}
    for p in prompts:
        if p not in seen:
            seen[p] = len(unique)
            unique.append(p)
        rows.append(seen[p])
    return rows, unique


# --------------------------------------------------------------------------- #
# encode
# --------------------------------------------------------------------------- #
def umt5_encoder(model_id: str, max_len: int, device: str, batch_size: int,
                 dtype: str = "bfloat16"):
    """A callable `texts -> (context float32 [n, L, D], mask bool [n, L])`.

    Kept behind a factory so `encode_all` can be exercised with a stub in tests and the
    heavy import only happens on the machine that has the GPU.

    `dtype` is a knob rather than a constant because the encoder runs on two very
    different machines. On a GPU, bfloat16 is the obvious choice. On CPU it is not: bf16
    matmul falls off the optimised kernels, and float32 is what the A-line eval sidecar
    uses -- encoding the B line's conditioning in the same precision the A line reads it
    back in keeps the two comparable instead of differing by an unmeasured amount.

    Loading is deferred to the first call. `encode_all` skips shards that already exist,
    so the assembly pass re-runs this command with every shard on disk and encodes
    nothing; eagerly materialising a 23 GB fp32 encoder to then do no work would make
    assembly need a machine it does not otherwise need.
    """
    import torch
    from transformers import AutoTokenizer, UMT5EncoderModel

    loaded: dict = {}

    def _model():
        if not loaded:
            torch_dtype = getattr(torch, dtype)
            loaded["tok"] = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
            loaded["net"] = UMT5EncoderModel.from_pretrained(
                model_id, subfolder="text_encoder", torch_dtype=torch_dtype
            ).to(device).eval()
        return loaded["tok"], loaded["net"]

    @torch.no_grad()
    def run(texts: list[str]):
        tokenizer, model = _model()
        out_ctx, out_mask = [], []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = tokenizer(
                chunk, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
            ).to(device)
            hidden = model(enc.input_ids, attention_mask=enc.attention_mask).last_hidden_state
            out_ctx.append(hidden.float().cpu().numpy())
            out_mask.append(enc.attention_mask.bool().cpu().numpy())
        return np.concatenate(out_ctx), np.concatenate(out_mask)

    return run


def encode_all(texts: list[str], encode, *, max_len: int, d_model: int, shard_rows: int,
               shard_dir: Path, log=print) -> tuple[np.ndarray, np.ndarray]:
    """Encode every prompt, one shard file at a time, skipping shards already on disk.

    The official folds the mask into the values before the model ever sees them
    (`robot_video_dataset.py:220-221`: `context[~mask] = 0`, then the mask becomes all
    ones), so the zeroing happens here and the stored rows are already what the model
    consumes. The mask is written alongside for verification only.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    n = len(texts)
    emb = np.zeros((n, max_len, d_model), np.float16)
    mask = np.zeros((n, max_len), np.uint8)
    for start in range(0, n, shard_rows):
        stop = min(start + shard_rows, n)
        path = shard_dir / f"shard_{start:08d}.npz"
        if path.exists():
            with np.load(path) as z:
                emb[start:stop], mask[start:stop] = z["emb"], z["mask"]
            continue
        ctx, m = encode(texts[start:stop])
        if ctx.shape != (stop - start, max_len, d_model):
            raise ValueError(f"encoder returned {ctx.shape}, expected {(stop - start, max_len, d_model)}")
        ctx = np.where(m[:, :, None], ctx, 0.0)
        if not np.isfinite(ctx).all():
            raise ValueError(f"non-finite embeddings in rows {start}..{stop}")
        truncated = int(m.all(axis=1).sum())
        if truncated:
            log(f"[encode] {truncated} of {stop - start} prompts fill all {max_len} tokens (truncated?)")
        emb[start:stop] = ctx.astype(np.float16)
        mask[start:stop] = m.astype(np.uint8)
        _atomic_write_npz(path, emb=emb[start:stop], mask=mask[start:stop])
        log(f"[encode] {stop}/{n}")
    return emb, mask


# --------------------------------------------------------------------------- #
# write / publish
# --------------------------------------------------------------------------- #
def write_sidetable(directory: Path, *, episode_ids, rows, texts, emb, mask, encoder: str,
                    template: str = DEFAULT_PROMPT, granularity: str = "per_episode",
                    row_source: str = "seen[0]", l3_prefix: str = "", l3_index_fingerprint: str = "") -> dict:
    """Write the four files the loader reads, plus the provenance meta.

    `rows.i32` is dense over episode id (-1 where an episode has no row) so a lookup is
    an array index and never a hash of a string that might have been re-templated.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    episode_ids = np.asarray(episode_ids, np.int64)
    rows = np.asarray(rows, np.int64)
    emb = np.ascontiguousarray(emb, np.float16)
    mask = np.ascontiguousarray(mask, np.uint8)
    if episode_ids.shape != rows.shape:
        raise ValueError(f"episode_ids {episode_ids.shape} and rows {rows.shape} must line up")
    if emb.ndim != 3 or emb.shape[0] != len(texts) or mask.shape != emb.shape[:2]:
        raise ValueError(f"emb {emb.shape} / mask {mask.shape} do not match {len(texts)} texts")
    if rows.max() >= emb.shape[0] or rows.min() < 0:
        raise ValueError("a row index falls outside the embedding table")

    dense = np.full(int(episode_ids.max()) + 1, -1, np.int32)
    dense[episode_ids] = rows.astype(np.int32)

    _atomic_write(directory / EMB_NAME, emb.tobytes())
    _atomic_write(directory / MASK_NAME, mask.tobytes())
    _atomic_write(directory / ROWS_NAME, dense.tobytes())
    _atomic_write(directory / TEXTS_NAME, json.dumps(texts, ensure_ascii=False).encode())
    meta = {
        "encoder": encoder,
        "max_len": int(emb.shape[1]),
        "d_model": int(emb.shape[2]),
        "dtype": "float16",
        "n_rows": int(emb.shape[0]),
        "n_episodes": int(episode_ids.shape[0]),
        "n_episode_slots": int(dense.shape[0]),
        "template": template,
        "granularity": granularity,
        "row_source": row_source,
        "padding": "masked positions zeroed at build time; the loader emits an all-True mask",
        "emb_sha256": hashlib.sha256(emb.tobytes()).hexdigest(),
        "l3_prefix": l3_prefix,
        "l3_index_fingerprint": l3_index_fingerprint,
    }
    _atomic_write(directory / META_NAME, json.dumps(meta, ensure_ascii=False, indent=2).encode())
    return meta


def _atomic_write(path: Path, blob: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(blob)
    os.replace(tmp, path)


def _atomic_write_npz(path: Path, **arrays) -> None:
    tmp = path.with_suffix(".npz.part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _cmd_collect(args) -> None:
    collect(args.l3, Path(args.out), workers=args.workers, limit=args.limit)


def _cmd_encode(args) -> None:
    directory = Path(args.dir)
    payload = json.loads((directory / PROMPTS_NAME).read_text())
    texts = payload["texts"]
    print(f"[encode] {len(texts)} distinct prompts, {args.encoder}, L={args.max_len}, "
          f"{args.device}/{args.dtype}", flush=True)
    encode = umt5_encoder(args.encoder, args.max_len, args.device, args.batch_size, args.dtype)
    emb, mask = encode_all(
        texts, encode, max_len=args.max_len, d_model=args.d_model,
        shard_rows=args.shard_rows, shard_dir=directory / "shards",
    )
    meta = write_sidetable(
        directory,
        episode_ids=payload["episode_ids"],
        rows=payload["rows"],
        texts=texts,
        emb=emb,
        mask=mask,
        encoder=args.encoder,
        template=payload["template"],
        granularity=payload["granularity"],
        row_source=payload["row_source"],
        l3_prefix=payload["l3_prefix"],
        l3_index_fingerprint=payload["l3_index_fingerprint"],
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def _cmd_publish(args) -> None:
    import subprocess

    directory = Path(args.dir)
    files = [META_NAME, ROWS_NAME, EMB_NAME, MASK_NAME, TEXTS_NAME]
    missing = [f for f in files if not (directory / f).exists()]
    if missing:
        raise SystemExit(f"nothing to publish: {missing} are not in {directory}")
    # gsutil rather than gcsfs: a host with working gcloud credentials does not
    # necessarily have application-default credentials, and this is a one-shot upload.
    cmd = ["gsutil", "-m", "cp"] + [str(directory / f) for f in files] + [args.dst.rstrip("/") + "/"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="L3 -> prompts.json (no GPU)")
    c.add_argument("--l3", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--workers", type=int, default=32)
    c.add_argument("--limit", type=int, default=None, help="first N episodes, for a smoke run")
    c.set_defaults(func=_cmd_collect)

    e = sub.add_parser("encode", help="prompts.json -> emb.f16 + meta.json (GPU)")
    e.add_argument("--dir", required=True)
    e.add_argument("--encoder", default=DEFAULT_ENCODER)
    e.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, dest="max_len")
    e.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL, dest="d_model")
    e.add_argument("--device", default="cuda")
    e.add_argument("--dtype", default="float32", help="use float32 for the paper text-encoding profile")
    e.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    e.add_argument("--shard-rows", type=int, default=SHARD_ROWS, dest="shard_rows")
    e.set_defaults(func=_cmd_encode)

    p = sub.add_parser("publish", help="upload the finished table")
    p.add_argument("--dir", required=True)
    p.add_argument("--dst", required=True)
    p.set_defaults(func=_cmd_publish)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
