"""Protocol check for fastwam_model_server.py: framing, codec, sessions — no weights.

Run it with any python that has numpy; it starts the real `serve()` on a free port
against a stub model and talks to it with the client's own encoder, so what is being
checked is the actual wire path, not a description of it.

What this catches that a unit test would not:
  * the uint8 canvas surviving base64 + json + reshape unchanged. The payload is 368 KB
    of pixels per replan and a dtype or shape slip here would arrive as a *valid* image
    of the wrong thing.
  * `_Session` isolation: two connections are two episodes, and one setting its task
    must not move the other's. The TinyWAM server needed a per-connection RNG band for
    exactly this reason once the harness went to one connection per episode.
  * errors coming back as `{"error": ...}` rather than killing the connection — the
    harness treats a dead socket as an infra failure and retries, which would turn a
    real refusal (e.g. get_action before set_task) into three wasted episodes.

    python scripts/check_protocol.py
"""

import os
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastwam_model_server as srv  # noqa: E402

CANVAS_HW = (384, 320)
HORIZON, ACTION_DIM = 32, 14


class StubModel:
    """The three RPC methods, with the same guards the real model has."""

    def __init__(self):
        self.context = None
        self.instruction = None
        self.policy = _StubPolicy()
        self.episodes = 0
        self.seen = []

    def reset_model(self):
        self.context = None
        self.instruction = None
        self.episodes += 1
        return "ok"

    # `_Session.call` special-cases set_task into resolve (pure, outside the lock) then
    # apply (three assignments, under it), so the stub has to offer the same two halves
    # or the check dies on a missing attribute against a perfectly healthy server.
    def resolve_task(self, obs):
        instruction = obs.get("instruction") if isinstance(obs, dict) else obs
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("set_task needs a non-empty 'instruction' string")
        # Pure: reads and writes no model state. That is the property the split exists
        # for, and doing it wrong here would hide a real regression in the server.
        return instruction, np.full((128, 8), float(len(instruction)), np.float32), "stub"

    def apply_task(self, instruction, context, digest):
        self.instruction = instruction
        self.context = context
        return "ok"

    def set_task(self, obs):
        return self.apply_task(*self.resolve_task(obs))

    def get_action(self, payload):
        if self.context is None:
            raise RuntimeError("get_action before set_task")
        canvas, qpos = payload["canvas"], np.asarray(payload["qpos"], np.float32)
        self.seen.append((canvas.shape, str(canvas.dtype), int(canvas.sum()), self.instruction))
        # Encode what arrived into the reply so the caller can verify the round trip.
        chunk = np.zeros((HORIZON, ACTION_DIM), np.float32)
        chunk[0, 0] = float(canvas.sum())
        chunk[0, 1] = float(qpos.sum())
        chunk[0, 2] = float(len(self.instruction))
        return chunk


class _StubPolicy:
    noise = np.zeros(1)
    episodes = 0
    replans = 0

    def reset(self):
        self.episodes += 1


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Client:
    """The RoboTwin client's `ModelClient`, minimally."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)

    def call(self, cmd, obs=None):
        import json

        payload = json.dumps({"cmd": cmd, "obs": obs}, cls=srv.NumpyEncoder).encode("utf-8")
        self.sock.sendall(len(payload).to_bytes(4, "big"))
        self.sock.sendall(payload)
        size = int.from_bytes(srv._recv_exact(self.sock, 4), "big")
        reply = srv.json_to_numpy(srv._recv_exact(self.sock, size).decode("utf-8"))
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply["res"]

    def close(self):
        self.sock.close()


def main():
    model = StubModel()
    port = free_port()
    threading.Thread(target=srv.serve, args=(model, port, 4), daemon=True).start()
    for _ in range(100):
        try:
            probe = Client(port)
            probe.close()
            break
        except OSError:
            time.sleep(0.02)
    else:
        raise SystemExit("server never came up")

    rng = np.random.default_rng(0)
    canvas = rng.integers(0, 256, (*CANVAS_HW, 3), dtype=np.uint8)
    qpos = np.linspace(-1, 1, ACTION_DIM, dtype=np.float32)

    a = Client(port)
    assert a.call("reset_model") == "ok"

    # 1. get_action before set_task must be a clean protocol error, not a dropped socket.
    try:
        a.call("get_action", {"canvas": canvas, "qpos": qpos})
        raise SystemExit("FAIL: get_action before set_task was allowed")
    except RuntimeError as exc:
        assert "set_task" in str(exc), exc
    print("ok   get_action before set_task -> {'error': ...}, connection survives")

    # 2. an empty instruction is refused (a defaulted one would run the whole episode
    #    on the wrong text and score like a modelling problem).
    for bad in ("", "   "):
        try:
            a.call("set_task", {"instruction": bad})
            raise SystemExit(f"FAIL: empty instruction {bad!r} accepted")
        except RuntimeError:
            pass
    print("ok   empty instruction refused")

    # 3. the real round trip.
    instruction = "Pick up the shoe with lace holes and move it to the mat"
    assert a.call("set_task", {"instruction": instruction}) == "ok"
    chunk = np.asarray(a.call("get_action", {"canvas": canvas, "qpos": qpos}), np.float32)
    assert chunk.shape == (HORIZON, ACTION_DIM), chunk.shape
    assert chunk.dtype == np.float32
    shape, dtype, total, seen_instruction = model.seen[-1]
    assert shape == (*CANVAS_HW, 3), f"canvas arrived as {shape}"
    assert dtype == "uint8", f"canvas arrived as {dtype}"
    assert total == int(canvas.sum()), "canvas pixels changed in flight"
    assert seen_instruction == instruction
    # The pixel sum is ~4.7e7, past float32's exact-integer range, so compare in float32
    # on both sides — the *exact* check is `total` above, on the server's own ints.
    assert chunk[0, 0] == np.float32(canvas.sum())
    assert chunk[0, 2] == np.float32(len(instruction))
    print(f"ok   canvas {shape} {dtype} round-tripped byte-exact ({total} pixel sum)")
    print(f"ok   chunk {chunk.shape} {chunk.dtype} returned")

    # 4. two connections are two episodes.
    b = Client(port)
    b.call("reset_model")
    b.call("set_task", {"instruction": "Lift the pot"})
    b.call("get_action", {"canvas": canvas, "qpos": qpos})
    assert model.seen[-1][3] == "Lift the pot"
    a.call("get_action", {"canvas": canvas, "qpos": qpos})
    assert model.seen[-1][3] == instruction, (
        f"session bleed: connection A saw {model.seen[-1][3]!r} after B set its own task"
    )
    print("ok   two connections keep separate instructions (session isolation)")

    a.close()
    b.close()
    print("\nPASS: framing, numpy codec, session isolation and error replies all behave")


if __name__ == "__main__":
    main()
    # `serve()` never returns — it is an accept loop, which is correct for a server and
    # wrong for a check. Its daemon threads are still inside `socket.recv` when the
    # interpreter starts tearing down, and CPython aborts (SIGABRT, exit 134) about half
    # the time. That made this script an intermittently "failing" preflight gate while
    # every assertion above had already passed. `os._exit` skips the teardown entirely,
    # which is exactly what a process whose only remaining job is to report should do.
    sys.stdout.flush()
    os._exit(0)
