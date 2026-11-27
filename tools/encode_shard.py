"""Encode one host's slice of the text side table, for the 16-host CPU fan-out.

Why this exists: the side table's `encode` stage wants a GPU, and the only GPU in reach
was blocked. The v6e slice has 16 hosts idling at 180 cores each while the TPU chips sit
unused, so the encode runs there instead -- on CPU, in float32, which is also the
precision the A-line eval sidecar reads embeddings back in.

This wrapper is deliberately thin. It does NOT reimplement the side table:

    * it calls `prepare_text_sidetable.umt5_encoder`, the same encoder factory,
    * it writes `shards/shard_%08d.npz` with the same names, dtypes and mask-zeroing as
      `encode_all`,
    * assembly is then the UNMODIFIED `prepare_text_sidetable.py encode` command run over
      a directory holding all the shards: it finds every shard on disk, encodes nothing,
      and writes emb.f16 / mask / rows / texts / meta.json through its own code path.

That last point is the whole design. The on-disk layout the reader expects is produced by
the code that has always produced it, so there is no second implementation to keep in
sync -- this file only has to get the shard files right.

    # on each host, HID in 0..15
    python tools/encode_shard.py --dir /path/to/text_sidetable \\
        --host-id $HID --num-hosts 16 --device cpu --dtype float32

Shards are assigned round-robin (`shard_index % num_hosts == host_id`) rather than in
contiguous blocks: the prompts are grouped by task, so a contiguous split would give one
host all the long prompts and another all the short ones.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from prepare_text_sidetable import (  # noqa: E402
    DEFAULT_D_MODEL,
    DEFAULT_ENCODER,
    DEFAULT_MAX_LEN,
    PROMPTS_NAME,
    SHARD_ROWS,
    _atomic_write_npz,
    umt5_encoder,
)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="holds prompts.json; shards/ is written here")
    ap.add_argument("--host-id", type=int, required=True, dest="host_id")
    ap.add_argument("--num-hosts", type=int, required=True, dest="num_hosts")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, dest="max_len")
    ap.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL, dest="d_model")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    ap.add_argument("--shard-rows", type=int, default=SHARD_ROWS, dest="shard_rows")
    args = ap.parse_args(argv)

    if not 0 <= args.host_id < args.num_hosts:
        raise SystemExit(f"host-id {args.host_id} out of range for {args.num_hosts} hosts")

    directory = Path(args.dir)
    payload = json.loads((directory / PROMPTS_NAME).read_text())
    texts = payload["texts"]
    n = len(texts)

    starts = list(range(0, n, args.shard_rows))
    mine = [s for i, s in enumerate(starts) if i % args.num_hosts == args.host_id]
    shard_dir = directory / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(f"[shard {args.host_id}/{args.num_hosts}] {n} prompts, {len(starts)} shards total, "
          f"{len(mine)} mine, rows={args.shard_rows}, {args.device}/{args.dtype}", flush=True)

    encode = umt5_encoder(args.encoder, args.max_len, args.device, args.batch_size, args.dtype)

    done = 0
    for start in mine:
        stop = min(start + args.shard_rows, n)
        path = shard_dir / f"shard_{start:08d}.npz"
        if path.exists():
            print(f"[shard {args.host_id}] {path.name} already present, skipping", flush=True)
            done += 1
            continue

        t0 = time.time()
        ctx, m = encode(texts[start:stop])
        if ctx.shape != (stop - start, args.max_len, args.d_model):
            raise ValueError(
                f"encoder returned {ctx.shape}, expected {(stop - start, args.max_len, args.d_model)}"
            )
        # Same as encode_all: the official zeroes masked positions before the model ever
        # sees them, so the stored rows are already what gets consumed.
        ctx = np.where(m[:, :, None], ctx, 0.0)
        if not np.isfinite(ctx).all():
            raise ValueError(f"non-finite embeddings in rows {start}..{stop}")
        _atomic_write_npz(path, emb=ctx.astype(np.float16), mask=m.astype(np.uint8))
        done += 1
        dt = time.time() - t0
        print(f"[shard {args.host_id}] {path.name} rows {start}..{stop} in {dt:.1f}s "
              f"({dt / (stop - start):.2f}s/row)  {done}/{len(mine)}", flush=True)

    print(f"[shard {args.host_id}] SHARD_DONE {done}/{len(mine)}", flush=True)


if __name__ == "__main__":
    main()
