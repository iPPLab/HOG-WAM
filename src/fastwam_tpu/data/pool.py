"""Episode index + the streaming FIFO pool that keeps the TPU fed from L3.

Ported from `codebases/TinyWAM-v4/tinywam/pool_data.py:61-293`, which has run on v4-32
and v6e-32; the design is `GC_TPU/robotwin_data_arch.md` §2.4. The mechanism, unchanged:

* **One episode is one file**, read in a single range read of `[0, payload_end)`. That
  range already contains the baked text embedding (the `.twp` section order puts
  `TEXT_EMB` before `PAYLOAD`) and stops before the `unseen` instructions, which the
  training process is not allowed to touch.
* **A pool of K resident episodes, `rate` of them replaced per step.** Reads scale with
  the number of steps, not with K: the whole run costs `hosts * (K + rate * steps)`
  GETs. At K=4096 that is ~50 GiB of RAM per host, against 400 GiB available on v4-32.
* **Admission order is a global shuffled permutation, striped by host**, so one lap
  admits every episode exactly once — no Poisson clumping from sampling with
  replacement (which pushes the per-episode appearance CV from 11.1% to 28.5%).
* **The pool at step t is a pure function of (seed, t)**, so resume rebuilds it with K
  hashes. It also depends implicitly on the host count, the weighting and the episode
  set, so `state()` carries those and `assert_same()` refuses to resume when any of them
  moved. That check raises; a warning here would go unread.

What this port adds:

* `TwpIndex.select` for our own train/val split. The official's split cannot be
  reproduced (it shuffles LeRobot episode numbers we do not have, `m4_data_design.md`
  R6), so we make our own and record it. Selecting changes the index fingerprint, which
  is the point: a resume across a different split must fail.
* An optional `sidecar` fetch that rides along with each admission. The per-episode text
  embeddings live in a separate object; fetching them here costs one extra GET per
  admission (≈1 per step) instead of one per sample (≈batch per step), and the FIFO ring
  is already an exact cache with exactly the right lifetime.
"""

import hashlib
import json
import queue
import threading
import time
from typing import Callable

import numpy as np

from fastwam_tpu.data import twp

__all__ = ["Admission", "LocalFileSystem", "TwpIndex", "TwpPool", "open_filesystem", "stable_seed"]


def stable_seed(*parts) -> int:
    """A seed that is identical across processes and python builds. `hash()` is salted."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


class LocalFileSystem:
    """The two `cat_file` calls we make, against a local directory.

    gcsfs needs application-default credentials, which a host with a working `gsutil`
    does not necessarily have; this makes the loader runnable against a `gsutil cp`-ed
    sample, which is how the shape and pixel gates get run without a TPU.
    """

    def cat_file(self, path: str, start: int | None = None, end: int | None = None) -> bytes:
        with open(path, "rb") as fh:
            if start:
                fh.seek(int(start))
            if end is None:
                return fh.read()
            return fh.read(int(end) - int(start or 0))

    def exists(self, path: str) -> bool:
        import os

        return os.path.exists(path)


def open_filesystem(prefix: str):
    """gcsfs for `gs://`, plain files otherwise. Import is lazy so tests need neither."""
    if prefix.startswith("gs://"):
        import gcsfs

        return gcsfs.GCSFileSystem()
    return LocalFileSystem()


class TwpIndex:
    """The episode list of one L3 prefix, from `MANIFEST.json`.

    json and not parquet on purpose: the TPU environment has no pyarrow.
    """

    def __init__(self, prefix: str, fs, *, _from: "TwpIndex | None" = None, _keep: np.ndarray | None = None):
        if _from is not None:
            self.prefix, self.fs, self.spec = _from.prefix, _from.fs, _from.spec
            self.cameras, self.camera_roles = _from.cameras, _from.camera_roles
            self.n_excluded = _from.n_excluded
            self.episode_id = _from.episode_id[_keep]
            self.n_frames = _from.n_frames[_keep]
            self.payload_end = _from.payload_end[_keep]
            self.task_key = [_from.task_key[i] for i in _keep]
        else:
            # The config's paths carry `gs://` (the cross-region guard keys off that
            # prefix); gcsfs wants them without the scheme.
            self.prefix = prefix.removeprefix("gs://").rstrip("/")
            self.fs = fs
            man = json.loads(fs.cat_file(f"{self.prefix}/MANIFEST.json"))
            self.spec = json.loads(fs.cat_file(f"{self.prefix}/spec.json"))
            # MANIFEST's own `prefix` field is a pre-move path and points at nothing;
            # only the caller's prefix is ever trusted (`m4_data_design.md` R5).
            eps = [
                e
                for e in man["episodes"]
                if e.get("bad_jpeg", 0) == 0 and e.get("nonfinite_actions", 0) == 0
            ]
            eps.sort(key=lambda e: e["episode_id"])
            self.episode_id = np.array([e["episode_id"] for e in eps], np.int64)
            self.n_frames = np.array([e["n_frames"] for e in eps], np.int64)
            self.payload_end = np.array([e["payload_end"] for e in eps], np.int64)
            self.task_key = [e["task_key"] for e in eps]
            self.n_excluded = len(man["episodes"]) - len(eps)
            self.cameras = list(self.spec["cameras"])
            self.camera_roles = list(self.spec["camera_roles"])
            if len(self.cameras) != len(self.camera_roles):
                raise ValueError(
                    f"{prefix}/spec.json: cameras {self.cameras} and camera_roles "
                    f"{self.camera_roles} have different lengths"
                )

    def __len__(self) -> int:
        return len(self.episode_id)

    @property
    def fingerprint(self) -> str:
        """Identifies the exact episode set. Excluding an episode or reshuffling the
        split changes it, and resume compares it, because the same (seed, step) over a
        different set gives a different pool with no other symptom.

        The dtypes below are part of the definition, not an implementation detail: they
        are int64/int32 so this keeps producing the value TinyWAM's L3 build recorded
        (the full pixel_v3 set hashes to `cd818fceb45f7736`), which is what makes our
        fingerprint comparable with the one written into existing artifacts.
        """
        return hashlib.sha256(
            self.episode_id.astype(np.int64).tobytes() + self.n_frames.astype(np.int32).tobytes()
        ).hexdigest()[:16]

    @property
    def num_windows(self) -> int:
        """Total sample starts = sum(T-1). Over the full set this must equal the
        official `num_transition`; that identity is what proves the two pipelines see
        the same episodes."""
        return int((self.n_frames - 1).sum())

    def select(self, keep: np.ndarray) -> "TwpIndex":
        """A view over a subset (train/val). Shares the manifest, gets a new fingerprint."""
        keep = np.asarray(keep, np.int64)
        if keep.size == 0:
            raise ValueError("refusing to build an empty episode index")
        if keep.min() < 0 or keep.max() >= len(self):
            raise ValueError(f"select indices out of range for {len(self)} episodes")
        return TwpIndex("", None, _from=self, _keep=keep)

    def select_tasks(self, names) -> "TwpIndex":
        """A view over the episodes whose `task_key` is one of `names`.

        `task_key` is the upstream task directory name and the one natural key of the
        RoboTwin set (`robotwin_data_arch.md` §5.4): 50 keys, 550 episodes each. Selecting
        on it is how a small-scale run trains on a subset of tasks.

        A name that matches nothing RAISES rather than being dropped. The alternative is a
        typo silently shrinking the dataset — the run would look completely healthy on less
        data than it was asked for, and only the final score would know. The order of the returned
        episodes is the index's own (episode_id ascending), NOT the order of `names`, so the
        selection is a pure function of the set of names: every host computes the same
        index and the fingerprint below is comparable across processes.
        """
        wanted = list(dict.fromkeys(names))
        if not wanted:
            raise ValueError("task_filter is empty; use None to mean 'every task'")
        available = set(self.task_key)
        unknown = [n for n in wanted if n not in available]
        if unknown:
            raise ValueError(
                f"task_filter names {unknown} that this L3 index does not have. "
                f"It carries {len(available)} task keys, e.g. {sorted(available)[:5]}"
            )
        keep_set = set(wanted)
        keep = np.array([i for i, key in enumerate(self.task_key) if key in keep_set], np.int64)
        return self.select(keep)

    def path(self, i: int) -> str:
        return f"{self.prefix}/episode_{int(self.episode_id[i]):06d}.twp"


class Admission:
    """One resident slot: the episode plus whatever the sidecar fetched with it."""

    __slots__ = ("slot", "episode", "sidecar")

    def __init__(self, slot: int, episode: "twp.Episode", sidecar):
        self.slot = int(slot)  # position in the index, so weights and ids stay resolvable
        self.episode = episode
        self.sidecar = sidecar


class TwpPool:
    """A K-slot FIFO ring whose contents at step t are a pure function of (seed, t)."""

    def __init__(
        self,
        index: TwpIndex,
        *,
        size: int,
        rate: float,
        seed: int,
        host_id: int = 0,
        n_hosts: int = 1,
        prefetch: int = 4,
        weights: np.ndarray | None = None,
        start_step: int = 0,
        prefetch_threads: int = 8,
        sidecar: Callable[[int], object] | None = None,
    ):
        if size > len(index):
            raise ValueError(f"pool size={size} exceeds the {len(index)} available episodes")
        if not (0 < rate <= size):
            raise ValueError(f"pool rate={rate} must lie in (0, size={size}]")
        self.ix, self.K, self.seed = index, int(size), int(seed)
        # Only the ratio matters. The knobs used to be (every n steps, swap k episodes),
        # but k=4,n=4 and k=1,n=1 produce the same pool sequence, so exposing both let
        # you "change two settings and see no effect". rate fixes both the turnover
        # period (size/rate steps) and the read count (hosts * (size + rate*steps)).
        self.rate = float(rate)
        self.prefetch_threads = int(prefetch_threads)
        self.h, self.H = int(host_id), int(n_hosts)
        self.w = None if weights is None else np.asarray(weights, np.float64)
        if self.w is not None and self.w.shape != (len(index),):
            raise ValueError(f"weights must be one per episode ({len(index)}), got {self.w.shape}")
        self.sidecar = sidecar
        self._perm_cache: dict[int, np.ndarray] = {}
        self._ring: list[Admission | None] = [None] * self.K
        self.start_step = int(start_step)
        self._admitted = 0
        self._q: queue.Queue = queue.Queue(maxsize=prefetch)
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def _advanced(self, step: int) -> int:
        """How many extra admissions have happened by `step`.

        The epsilon is for rates that are clean in decimal and not in binary: without it
        `floor(0.1*30)` can be 2 instead of 3 and the pool silently runs half a beat late.
        """
        return int(self.rate * step + 1e-9)

    # ---------------- pure part: pool contents depend only on (seed, step)

    def _perm(self, g: int) -> np.ndarray:
        """Lap `g`'s global permutation. Deliberately **unweighted** — a weighted
        permutation makes heavy episodes cluster at the start of every lap, so the pool's
        composition oscillates with the lap period."""
        if g not in self._perm_cache:
            self._perm_cache[g] = np.random.default_rng(
                stable_seed(self.seed, "perm", g)
            ).permutation(len(self.ix))
            if len(self._perm_cache) > 3:
                self._perm_cache.pop(min(self._perm_cache))
        return self._perm_cache[g]

    def admission(self, j: int) -> int:
        """Index of this host's `j`-th admission.

        The global sequence number is `j*H + h`, so one lap of `len(index)` admissions
        covers every episode exactly once whether or not the host count divides it.
        """
        g_idx = j * self.H + self.h
        return int(self._perm(g_idx // len(self.ix))[g_idx % len(self.ix)])

    def pool_at(self, step: int) -> list[int]:
        """Episode indices resident at `step`, oldest first. Pure; K hashes to rebuild."""
        lo = self._advanced(step)
        return [self.admission(j) for j in range(lo, lo + self.K)]

    # ---------------- fetching

    def _fetch(self, i: int) -> Admission:
        buf = self.ix.fs.cat_file(self.ix.path(i), start=0, end=int(self.ix.payload_end[i]))
        # expect_id catches the file-name-says-one-thing-content-says-another case.
        episode = twp.Episode(buf, expect_id=int(self.ix.episode_id[i]))
        side = None if self.sidecar is None else self.sidecar(int(self.ix.episode_id[i]))
        return Admission(i, episode, side)

    def _worker(self):
        """Fetch concurrently, deliver **strictly in `j` order**.

        A single GCS stream gives about 8 MB/s, so serial fetching starves the pool and
        the symptom is `advance()` blocking on the queue. Ordering is not optional: the
        delivery order decides which ring slot an episode lands in (`_admitted % K`), so
        out-of-order delivery desynchronises the ring from `pool_at(step)` — silently,
        and the pure-function resume guarantee goes with it.
        """
        import concurrent.futures as cf

        j = self._admitted
        with cf.ThreadPoolExecutor(self.prefetch_threads, thread_name_prefix="twp-fetch") as ex:
            inflight: dict[int, cf.Future] = {}
            while not self._stop.is_set():
                try:
                    while len(inflight) < self.prefetch_threads:
                        nxt = j + len(inflight)
                        inflight[nxt] = ex.submit(self._fetch, self.admission(nxt))
                    fut = inflight[j]
                    item = fut.result(timeout=120)
                    self._q.put(item, timeout=5)
                    del inflight[j]
                    j += 1
                except queue.Full:
                    continue
                except Exception as e:
                    # Never skip: a skipped admission forks the replay from the recorded one.
                    print(f"[pool] fetch failed at j={j}: {e!r} — retrying in 5s", flush=True)
                    inflight.pop(j, None)
                    self._stop.wait(5)

    def _prefill(self):
        import concurrent.futures as cf

        t0 = time.time()
        lo = self._advanced(self.start_step)
        js = list(range(lo, lo + self.K))
        with cf.ThreadPoolExecutor(32) as ex:
            for j, item in zip(js, ex.map(lambda j: self._fetch(self.admission(j)), js)):
                self._ring[j % self.K] = item  # slot = j % K, same order advance() evicts in
        self._admitted = lo + self.K
        mib = sum(a.episode.payload_end() for a in self._ring if a) / 2**20
        print(
            f"[pool] prefilled {self.K} episodes (from step {self.start_step}, admission {lo}): "
            f"{mib / 1024:.1f} GiB in {time.time() - t0:.0f}s "
            f"({mib / max(1e-9, time.time() - t0):.0f} MiB/s)",
            flush=True,
        )
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._ready.set()

    def start(self):
        """**Non-blocking.** The prefill overlaps model build and cold compile: K=4096 is
        ~50 GiB and takes minutes even in-region, and on spot capacity those minutes are
        the whole budget."""
        threading.Thread(target=self._prefill, name="twp-prefill", daemon=True).start()
        return self

    def wait_ready(self, timeout: float | None = None):
        if not self._ready.wait(timeout):
            raise TimeoutError(f"pool prefill did not finish within {timeout}s")

    def advance(self, step: int):
        """Roll forward to `step`: evict the oldest, admit `rate` per step."""
        self.wait_ready()
        want = self.K + self._advanced(step)
        while self._admitted < want:
            self._ring[self._admitted % self.K] = self._q.get()
            self._admitted += 1

    def draw(self, rng: np.random.Generator, k: int) -> list[Admission]:
        """Draw `k` residents with replacement, honouring the weights.

        Unweighted means uniform over episodes; weights proportional to the window count
        mean uniform over samples, which is what the official row-level sampler does.
        Either way it is a decision that was made on purpose, not a side effect.
        """
        live = [a for a in self._ring if a is not None]
        if not live:
            raise RuntimeError("the pool is empty; call start() and wait_ready() first")
        if self.w is None:
            return [live[i] for i in rng.integers(len(live), size=k)]
        pr = np.array([self.w[a.slot] for a in live], np.float64)
        pr /= pr.sum()
        return [live[i] for i in rng.choice(len(live), size=k, p=pr)]

    def stop(self):
        """Must be called when training ends: otherwise the worker keeps hammering a
        closed gcsfs session, logging every 5s, and can hold the process open."""
        self._stop.set()
        try:
            while not self._q.empty():
                self._q.get_nowait()
        except Exception:
            pass

    def state(self) -> dict:
        """Goes in the checkpoint. If any of this moves, the same step gives a different
        pool with no other sign, so resume compares every field."""
        return {
            "seed": self.seed,
            "size": self.K,
            "rate": self.rate,
            "n_hosts": self.H,
            "index_fingerprint": self.ix.fingerprint,
            "weighted": self.w is not None,
            "perm_algo": "sha256-permutation-v1",
        }

    @staticmethod
    def assert_same(saved: dict, now: dict):
        """Resume guard. Raises — a warning would not be read."""
        diff = {k: (saved.get(k), now.get(k)) for k in set(saved) | set(now) if saved.get(k) != now.get(k)}
        if diff:
            lines = "\n".join(f"    {k}: saved={a!r}  now={b!r}" for k, (a, b) in sorted(diff.items()))
            raise ValueError(
                "pool state differs from the checkpoint, so the same step would read different "
                "data; refusing to resume:\n" + lines + "\n  (n_hosts changed = different slice; "
                "index_fingerprint changed = the dataset was rebuilt or the split moved. To start "
                "a fresh run instead, change the run name.)"
            )
