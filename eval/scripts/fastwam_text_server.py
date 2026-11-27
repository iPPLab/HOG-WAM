"""Serve one UMT5 encoder to the FastWAM evaluation workers.

Requests use length-prefixed JSON. The service caches embeddings on disk
so multiple model servers can reuse them.
"""

import argparse
import base64
import json
import os
import socket
import sys
import threading
import time
import traceback
from typing import Any

import numpy as np

# eval/scripts/ -> eval/ -> repo root. Derived from this file so the repo can be cloned
# anywhere; FASTWAM_JAX_PATH still wins for an out-of-tree checkout.
FASTWAM_JAX = os.environ.get(
    "FASTWAM_JAX_PATH",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
sys.path.insert(0, os.path.join(FASTWAM_JAX, "src"))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "data": base64.b64encode(obj.tobytes()).decode("ascii"),
                "dtype": str(obj.dtype),
                "shape": obj.shape,
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def json_to_numpy(text: str) -> Any:
    def hook(payload):
        if "__numpy_array__" in payload:
            return np.frombuffer(
                base64.b64decode(payload["data"]), dtype=payload["dtype"]
            ).reshape(payload["shape"])
        return payload

    return json.loads(text, object_hook=hook)


def _recv_exact(sock, count):
    chunks, got = [], 0
    while got < count:
        chunk = sock.recv(min(count - got, 65536))
        if not chunk:
            raise ConnectionError("client closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


class TextService:
    def __init__(self, *, cache_dir: str, max_len: int, dtype: str, repo: str | None):
        from fastwam_tpu.inference.text import DEFAULT_REPO, TextEmbeddingCache, Umt5Encoder

        available = _mem_available_gb()
        need = 26 if dtype == "float32" else 14
        if available < need:
            raise SystemExit(
                f"only {available} GB of RAM available; loading UMT5 in {dtype} needs ~{need} GB."
            )
        started = time.time()
        self.encoder = Umt5Encoder(repo=repo or DEFAULT_REPO, max_len=max_len, dtype=dtype)
        self.cache = TextEmbeddingCache(
            cache_dir, max_len=max_len, text_dim=self.encoder.text_dim
        )
        self.encoded = 0
        print(f"[text] {self.encoder.describe()}", flush=True)
        print(f"[text] loaded in {time.time() - started:.1f}s; cache {self.cache.cache_dir}", flush=True)

    def ping(self):
        return "ok"

    def encode(self, obs):
        prompt = obs.get("prompt") if isinstance(obs, dict) else obs
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"encode needs a non-empty 'prompt', got {prompt!r}")
        rows, seq_len = self.encoder.encode_one(prompt)
        # Write through so the model servers' own cache lookups hit next time and a
        # restart of this process costs nothing already paid for.
        self.cache.put(prompt, rows, seq_len)
        self.encoded += 1
        if self.encoded % 25 == 1:
            print(f"[text] {self.encoded} encoded; {self.cache.stats()}", flush=True)
        return {"rows": rows.astype(np.float32), "seq_len": int(seq_len)}


def _mem_available_gb() -> int:
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(int(line.split()[1]) / 1048576)
    return 0


def _client_loop(client, address, service, lock):
    print(f"[text] client {address}", flush=True)
    with client:
        while True:
            try:
                size = int.from_bytes(_recv_exact(client, 4), "big")
                request = json_to_numpy(_recv_exact(client, size).decode("utf-8"))
                method = getattr(service, request.get("cmd"))
                obs = request.get("obs")
                with lock:  # one torch forward at a time; the model is not thread-safe
                    result = method(obs) if obs is not None else method()
                payload = json.dumps({"res": result}, cls=NumpyEncoder).encode("utf-8")
                client.sendall(len(payload).to_bytes(4, "big"))
                client.sendall(payload)
            except (ConnectionError, ConnectionResetError, BrokenPipeError):
                print(f"[text] client {address} disconnected", flush=True)
                return
            except Exception as exc:  # noqa: BLE001
                print(f"[text] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
                payload = json.dumps({"error": str(exc)}, cls=NumpyEncoder).encode("utf-8")
                try:
                    client.sendall(len(payload).to_bytes(4, "big"))
                    client.sendall(payload)
                except Exception:
                    return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=6035)
    parser.add_argument("--cache", required=True, help="shared sha256(prompt) cache dir")
    parser.add_argument(
        "--repo",
        default=None,
        help="local or Hugging Face UMT5 Diffusers root; defaults to the canonical Wan repo",
    )
    parser.add_argument("--max-len", type=int, default=128, help="tokenizer_max_len; 128")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16", "float16"],
        help="UMT5 encoder dtype. Defaults to float32 for CPU execution.",
    )
    args = parser.parse_args(argv)

    service = TextService(
        cache_dir=args.cache,
        max_len=args.max_len,
        dtype=args.dtype,
        repo=args.repo,
    )
    lock = threading.Lock()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.port))
    server.listen(8)
    print(f"[text] listening on 127.0.0.1:{args.port}", flush=True)
    while True:
        client, address = server.accept()
        threading.Thread(
            target=_client_loop, args=(client, address, service, lock), daemon=True
        ).start()


if __name__ == "__main__":
    main()
