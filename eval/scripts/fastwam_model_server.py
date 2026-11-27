"""Run a RoboTwin model process using length-prefixed JSON RPC over TCP."""

import argparse
from dataclasses import replace
import base64
import hashlib
import itertools
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


# --------------------------------------------------------------------------- #
# wire protocol — byte-identical to policy_model_server.py / tinywam_model_server.py
# --------------------------------------------------------------------------- #
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


def rpc(address: str, cmd: str, obs, timeout: float = 600.0):
    """One request against another server speaking this protocol (the text sidecar)."""
    host, _, port = address.partition(":")
    with socket.create_connection((host or "127.0.0.1", int(port)), timeout=timeout) as sock:
        payload = json.dumps({"cmd": cmd, "obs": obs}, cls=NumpyEncoder).encode("utf-8")
        sock.sendall(len(payload).to_bytes(4, "big"))
        sock.sendall(payload)
        size = int.from_bytes(_recv_exact(sock, 4), "big")
        reply = json_to_numpy(_recv_exact(sock, size).decode("utf-8"))
    if "error" in reply:
        raise RuntimeError(f"text server error: {reply['error']}")
    return reply["res"]


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #
class FastWamRobotwinModel:
    """The official uncond checkpoint behind the RoboTwin protocol."""

    def __init__(
        self,
        *,
        ckpt: str,
        vae_ckpt: str,
        stats_path: str,
        config_path: str,
        text_cache_dir: str,
        text_server: str | None,
        text_encoder: str,
        denoise: int,
        horizon: int | None,
        noise_mode: str,
        noise_seed: int,
        dtype: str,
        stats_sha256: str | None,
        text_template: str = "official",
    ):
        started = time.time()
        import jax

        from fastwam_tpu import load_config
        from fastwam_tpu.inference.load import DTYPES, build_policy, describe_assets
        from fastwam_tpu.inference.text import (
            DEFAULT_MAX_LEN,
            TextEmbeddingCache,
            build_prompt,
            clean_text,
        )

        self.config = load_config(config_path)
        self.config = replace(self.config, model=replace(self.config.model, training_towers="rgb_action"))
        shapes = self.config.shapes
        print(f"[server] jax devices: {jax.devices()}", flush=True)
        print(f"[server] assets: {describe_assets(ckpt, vae_ckpt, stats_path)}", flush=True)

        policy_kwargs = dict(
            num_inference_steps=int(denoise),
            noise_mode=noise_mode,
            noise_seed=int(noise_seed),
        )
        if horizon is not None and int(horizon) != shapes.action_tokens:
            raise SystemExit(
                f"--horizon {horizon} contradicts the config's action_tokens "
                f"{shapes.action_tokens}. The horizon is derived (num_frames - 1), not a "
                "free knob; changing it changes the model's output width."
            )
        self.policy = build_policy(
            ckpt_root=ckpt,
            vae_ckpt=vae_ckpt,
            stats_path=stats_path,
            config=self.config,
            dtype=DTYPES[dtype],
            stats_sha256=stats_sha256,
            **policy_kwargs,
        )
        # Which text convention this checkpoint was trained under. NOT cosmetic: the
        # official ckpt saw
        #     "A video recorded from a robot's point of view executing the following
        #      instruction: {task}"
        # while a `text_mode: baked` B-line ckpt saw the BARE instruction encoded by
        # Wan2.1's UMT5 (fastwam_jax/data/text.py, BakedText.describe(): "BARE
        # instruction, no official template ... Not comparable with the official
        # checkpoint's conditioning"). Feeding the wrong one scores a correct model on
        # text it never saw and looks like a modelling failure.
        #
        # Default is "official" so nothing about the A line changes. `clean_text` is
        # applied in BOTH modes: it is `whitespace_clean(basic_clean(...))`, what
        # HuggingfaceTokenizer does anyway, so dropping it would introduce a difference
        # rather than remove one.
        if text_template not in ("official", "bare"):
            raise SystemExit(f"--text-template must be official|bare, got {text_template!r}")
        self.text_template = text_template
        if text_template == "bare":
            def _bare_prompt(instruction: str) -> str:
                if not isinstance(instruction, str) or not instruction.strip():
                    raise ValueError(
                        "instruction must be a non-empty string; an empty one means the "
                        "client never called set_task and the episode would run on the "
                        "wrong text."
                    )
                return clean_text(instruction)

            self.build_prompt = _bare_prompt
        else:
            self.build_prompt = build_prompt
        self.text = TextEmbeddingCache(
            text_cache_dir,
            max_len=DEFAULT_MAX_LEN,
            text_dim=self.config.model.video_dit_config.text_dim,
        )
        self.text_server = text_server
        self.text_encoder_mode = text_encoder
        self._encoder = None  # lazily built, only in "inprocess" mode
        self.canvas_hw = self.policy.canvas_hw

        # Per-episode state. `context is None` is the guard that makes a missing set_task
        # an error instead of an episode silently conditioned on nothing: the v3 campaign
        # lost a whole ablation to exactly that, as a warning nobody read.
        self.context = None
        self.context_mask = self.text.mask()
        self.instruction = None
        self.episodes = 0

        print(f"[server] loaded in {time.time() - started:.1f}s", flush=True)
        print(self.banner(), flush=True)
        self._warmup()

    # ---- conventions, said out loud -------------------------------------- #
    def banner(self) -> str:
        lines = [
            "================ FastWAM conventions (M3) ================",
            *self.policy.describe().splitlines(),
            f"text: {self.text_encoder_mode}"
            + (f" via {self.text_server}" if self.text_server else "")
            + f", cache {self.text.cache_dir}",
            f"text template: {self.text_template}"
            + ("  (official wrapper: 'A video recorded ... instruction: {task}')"
               if self.text_template == "official"
               else "  (BARE instruction, no wrapper -- for text_mode:baked ckpts)"),
            # Printed so a REUSED process cannot pass for a patched one: the campaign
            # launchers reuse a live server when the ckpt matches, so "I edited the file"
            # and "the server running is the edited file" are different claims.
            "set_task encode: lock-free (the UMT5 encode runs outside the global lock; a "
            "cold prompt stalls only its own lane)",
            "NOT present (and must never be): BGR swap, robotwin_triple 320x512 canvas, "
            "delta/anchor denormalizer, state_tokens, TW_CHUNK_EXEC, TW_ANCHOR_PREV",
            "==========================================================",
        ]
        return "\n".join(lines)

    def _warmup(self) -> None:
        """Compile every jitted shape BEFORE listen().

        Mandatory, not an optimization: the client's socket timeout is 300 s and the
        first `get_action` otherwise compiles 30 unrolled DiT layers while a real episode
        waits on it. The dummy context is zeros, which has the same shape and dtype as a
        real one, so the compiled executable is the one the loop will use.
        """
        started = time.time()
        dummy_context = np.zeros((self.text.max_len, self.text.text_dim), np.float32)
        canvas = np.zeros((*self.canvas_hw, 3), np.uint8)
        qpos = np.zeros((self.policy.proprio_dim,), np.float32)
        chunk = self.policy.infer_action(canvas, qpos, dummy_context, self.context_mask)
        if chunk.shape != (self.policy.action_horizon, self.policy.action_dim):
            raise SystemExit(f"warmup produced {chunk.shape}, expected the official chunk shape")
        if not np.isfinite(chunk).all():
            raise SystemExit("warmup produced non-finite actions; the checkpoint is not usable")
        self.policy.reset()
        print(f"[server] jit warmup done in {time.time() - started:.1f}s", flush=True)

    # ---- text ------------------------------------------------------------ #
    def _encode_prompt(self, prompt: str) -> np.ndarray:
        cached = self.text.get(prompt)
        if cached is not None:
            return cached
        if self.text_server:
            reply = rpc(self.text_server, "encode", {"prompt": prompt})
            rows, seq_len = reply["rows"], int(reply["seq_len"])
            return self.text.put(prompt, np.asarray(rows, np.float32), seq_len)
        if self.text_encoder_mode == "inprocess":
            if self._encoder is None:
                from fastwam_tpu.inference.text import Umt5Encoder

                self._encoder = Umt5Encoder(max_len=self.text.max_len)
                print(f"[server] {self._encoder.describe()}", flush=True)
            rows, seq_len = self._encoder.encode_one(prompt)
            return self.text.put(prompt, rows, seq_len)
        raise RuntimeError(
            f"cache miss on {prompt!r} and no encoder available. Start "
            "scripts/fastwam_text_server.py and pass --text-server host:port, or use "
            "--text-encoder inprocess (needs torch + transformers in this venv)."
        )

    # ---- RPC methods ------------------------------------------------------ #
    def reset_model(self):
        self.context = None
        self.instruction = None
        self.episodes += 1
        self.policy.reset()
        return "ok"

    def set_task(self, obs):
        """`{"instruction": "<the episode's drawn instruction>"}`.

        The official passes `TASK_ENV.get_instruction()`, a random draw from the
        **unseen** pool, so this is a real sentence and not a canonical task name. An
        empty one is refused rather than defaulted: a defaulted instruction would run the
        whole episode on the wrong text and score like a modelling problem.
        """
        instruction = obs.get("instruction") if isinstance(obs, dict) else obs
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"set_task needs a non-empty 'instruction' string, got {instruction!r}. "
                "The FastWAM client sends TASK_ENV.get_instruction()."
            )
        return self.apply_task(*self.resolve_task(obs))

    def resolve_task(self, obs):
        """Resolve text context without holding the shared model lock.

        A cache miss can wait for the UMT5-XXL service without blocking other clients'
        action requests. Only `apply_task` updates the instruction and context.
        """
        instruction = obs.get("instruction") if isinstance(obs, dict) else obs
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                f"set_task needs a non-empty 'instruction' string, got {instruction!r}. "
                "The FastWAM client sends TASK_ENV.get_instruction()."
            )
        prompt = self.build_prompt(instruction)
        return instruction, self._encode_prompt(prompt), hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()[:12]

    def apply_task(self, instruction, context, digest):
        """`set_task`'s fast half: publish the resolved text. Must hold the lock."""
        self.context = context
        self.instruction = instruction
        print(
            f"[server] set_task sha={digest} |ctx|={float(np.abs(self.context).max()):.3f} "
            f"{instruction!r}",
            flush=True,
        )
        return "ok"

    def get_action(self, payload):
        if self.context is None:
            raise RuntimeError(
                "get_action before set_task. Every episode must pin its instruction first "
                "— the text is read on the very first (and only) video prefill, so a late "
                "set_task would condition the chunk on nothing."
            )
        canvas = payload["canvas"]
        qpos = np.asarray(payload["qpos"], np.float32)
        return self.policy.infer_action(canvas, qpos, self.context, self.context_mask)


# --------------------------------------------------------------------------- #
# sessions and serving
# --------------------------------------------------------------------------- #
class _Session:
    """One connection = one episode. Owns the episode's text and noise state; borrows
    the weights. A single lock serializes model calls, so per-lane results are identical
    to a dedicated server and the concurrency win is the ~90 % of wall time each client
    spends rendering in its own process.
    """

    _SEQ = itertools.count(1)

    def __init__(self, model: FastWamRobotwinModel, lock: threading.Lock):
        self._model = model
        self._lock = lock
        self._seq = next(_Session._SEQ)
        self.context = None
        self.instruction = None
        # Each connection owns a band of episode indices, so "resample" noise differs per
        # episode instead of every connection replaying stream #1 (the bug the TinyWAM
        # server hit when the harness went from one-connection-per-task to per-episode).
        self.episodes = self._seq * 1000
        self.noise = model.policy.noise
        self.replans = 0

    def _swap_in(self):
        model = self._model
        model.context, model.instruction = self.context, self.instruction
        model.policy.noise = self.noise
        model.policy.episodes = self.episodes
        model.policy.replans = self.replans

    def _swap_out(self):
        model = self._model
        self.context, self.instruction = model.context, model.instruction
        self.noise = model.policy.noise
        self.episodes = model.policy.episodes
        self.replans = model.policy.replans

    def call(self, cmd, obs):
        # `set_task` resolves its text BEFORE taking the lock. The encode is a pure
        # function of the prompt (see FastWamRobotwinModel.resolve_task) and on a cache
        # miss it blocks for ~100 s, which under the lock stalled every other lane's
        # `get_action`. Only the publish half needs the lock, and it is three assignments.
        if cmd == "set_task":
            resolved = self._model.resolve_task(obs)
            with self._lock:
                self._swap_in()
                try:
                    return self._model.apply_task(*resolved)
                finally:
                    self._swap_out()
        with self._lock:
            self._swap_in()
            try:
                method = getattr(self._model, cmd)
                return method(obs) if obs is not None else method()
            finally:
                self._swap_out()


def _client_loop(client, address, handler):
    print(f"[server] client {address}", flush=True)
    with client:
        while True:
            try:
                size = int.from_bytes(_recv_exact(client, 4), "big")
                request = json_to_numpy(_recv_exact(client, size).decode("utf-8"))
                result = handler(request.get("cmd"), request.get("obs"))
                payload = json.dumps({"res": result}, cls=NumpyEncoder).encode("utf-8")
                client.sendall(len(payload).to_bytes(4, "big"))
                client.sendall(payload)
            except (ConnectionError, ConnectionResetError, BrokenPipeError):
                print(f"[server] client {address} disconnected", flush=True)
                return
            except Exception as exc:  # noqa: BLE001 — reported to the client and the log
                print(f"[server] ERROR: {exc}\n{traceback.format_exc()}", flush=True)
                payload = json.dumps({"error": str(exc)}, cls=NumpyEncoder).encode("utf-8")
                try:
                    client.sendall(len(payload).to_bytes(4, "big"))
                    client.sendall(payload)
                except Exception:
                    return


def serve(model, port, max_clients):
    lock = threading.Lock()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen(max_clients + 2)
    print(f"[server] listening on 127.0.0.1:{port} (max {max_clients} lanes)", flush=True)
    while True:
        client, address = server.accept()
        session = _Session(model, lock)
        threading.Thread(
            target=_client_loop, args=(client, address, session.call), daemon=True
        ).start()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True, help="6033/6034 — NOT 6023/6024")
    parser.add_argument("--ckpt", required=True, help="converted official Orbax root")
    parser.add_argument("--vae", required=True, help="Wan2.2 VAE Orbax dir")
    parser.add_argument("--stats", required=True, help="official dataset_stats.json")
    parser.add_argument("--stats-sha256", default=None, help="refuse a different stats file")
    parser.add_argument("--config", required=True)
    parser.add_argument("--denoise", type=int, default=10, help="official num_inference_steps")
    parser.add_argument("--horizon", type=int, default=None, help="cross-check only; derived")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32", "float16"])
    parser.add_argument(
        "--noise-mode",
        default="fixed",
        choices=["fixed", "resample"],
        help="fixed reproduces the official property (one seeded draw reused on every "
        "replan of every episode); resample is the A/B (design memo D4)",
    )
    parser.add_argument("--noise-seed", type=int, default=42, help="fixed training seed")
    parser.add_argument("--text-cache", required=True, help="shared sha256(prompt) cache dir")
    parser.add_argument(
        "--text-template", default="official", choices=["official", "bare"],
        help="official = the released ckpt's wrapper; bare = the raw instruction, for a "
             "checkpoint trained with text_mode:baked. Wrong choice conditions the model "
             "on text it never saw and reads as a modelling failure.")
    parser.add_argument(
        "--text-server", default=None, help="host:port of scripts/fastwam_text_server.py"
    )
    parser.add_argument(
        "--text-encoder",
        default="sidecar",
        choices=["sidecar", "inprocess"],
        help="inprocess needs torch+transformers HERE and 11 GB of RAM per server",
    )
    parser.add_argument("--max-clients", type=int, default=2)
    args = parser.parse_args(argv)

    if args.text_encoder == "sidecar" and not args.text_server:
        parser.error(
            "--text-encoder sidecar needs --text-server host:port. Cache hits alone cannot "
            "serve an unseen-instruction campaign: the prompt pool is a template x object "
            "cross product and nearly every episode draws a fresh one."
        )
    model = FastWamRobotwinModel(
        ckpt=args.ckpt,
        vae_ckpt=args.vae,
        stats_path=args.stats,
        config_path=args.config,
        text_cache_dir=args.text_cache,
        text_server=args.text_server,
        text_encoder=args.text_encoder,
        denoise=args.denoise,
        horizon=args.horizon,
        noise_mode=args.noise_mode,
        noise_seed=args.noise_seed,
        dtype=args.dtype,
        stats_sha256=args.stats_sha256,
        text_template=args.text_template,
    )
    serve(model, args.port, args.max_clients)


if __name__ == "__main__":
    main()
