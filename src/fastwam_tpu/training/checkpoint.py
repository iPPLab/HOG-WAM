"""Preemption-resilient Orbax checkpointing on GCS.

Ported from
    codebases/TinyWAM-v4/tinywam/train.py::Checkpointer (and the multihost-resumable
    system merged into TinyWAM main) — atomic versioned full/step_N directories, a
    consensus restore across hosts, NaN and save-health guards. Fault-injection tested on
    v4-16.

Only the DISTRIBUTED (collective) write path is ported. TinyWAM also has a chief-gathers-
everything path that stages through /dev/shm and rsyncs; it exists there for the HSDP
layout where one host's chips already hold a whole parameter. This harness shards
parameters over the entire slice (`training/sharding.py`), so no host can gather a whole
array and that path is not merely slow, it is invalid. Every host writes its own shards
straight to gs://, which is also what keeps host RAM out of the picture — the chief-gather
version peaks around 170 GB for a 6 B model.

Invariants carried over from prior campaigns (do not rediscover them)
    * A full save writes a NEW immutable directory and only then a commit_success marker;
      "finalized" means that marker exists. Never overwrite in place. A preemption at any
      instant therefore leaves either the previous finalized step intact or a marker-less
      partial that restore skips.
    * run_name must not contain a timestamp, so the gs:// path is stable across restarts.
      An estimation run that writes into the real run's path corrupts it.
    * Without tensorflow installed, etils.epath's gs:// mkdir is a silent no-op and orbax
      then raises FileNotFoundError -> patch mkdir; do NOT pre-create the directory (that
      trips orbax's "Destination already exists" precheck). gcsfs is a hard dependency.
    * A collective restore where one host is absent core-halts the whole slice, so every
      decision is agreed across hosts before anyone acts.
    * Never resume from a bare gs://.../latest: after a preemption it can point at a stale
      step and the run silently redoes work. There is no `latest` here at all.
    * A save that "succeeded" is one whose data landed AND whose marker was written; the
      failure counter exists because a persistently failing save on a long run is
      invisible otherwise.
"""

import hashlib
import json
import threading

import jax
import numpy as np

__all__ = [
    "CheckpointError",
    "Checkpointer",
    "step_number",
    "finalized_steps",
    "broadcast_int",
    "allgather_min",
    "patch_epath_gcs_mkdir",
]

_STEP_WIDTH = 9
_MARKER = "commit_success.txt"


class CheckpointError(RuntimeError):
    """A checkpoint that cannot be trusted, saved, or restored."""


def _log(*parts) -> None:
    print("[ckpt]", *parts, flush=True)


# --------------------------------------------------------------------------- #
# gs:// plumbing
# --------------------------------------------------------------------------- #
_PATCHED = False


def patch_epath_gcs_mkdir() -> None:
    """Make `etils.epath`'s gs:// mkdir actually create something, so orbax can write.

    Orbax's `_create_tmp_directory` calls `mkdir()` and then IMMEDIATELY requires the path
    to exist. With tensorflow installed epath routes gs:// through `tf.io.gfile`, whose
    makedirs writes a placeholder object, so the prefix exists. Without tensorflow (this
    venv) the fsspec backend's mkdir is a silent no-op — a GCS "directory" is nothing but a
    key prefix — and every orbax save to gs:// dies with `FileNotFoundError`, single
    process as much as multihost (2026-07-30, three-layer investigation).

    Patch mkdir rather than pre-creating the directory: orbax's "destination must not
    exist" check runs BEFORE mkdir, so a directory created up front trips that instead.
    """
    global _PATCHED
    if _PATCHED:
        return
    from etils import epath

    cls = type(epath.Path("gs://bucket/key"))
    original = cls.mkdir

    def mkdir(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if str(self).startswith("gs://") and not self.exists():
            (self / ".orbax_dir").write_text("")  # one real object => the prefix now exists

    cls.mkdir = mkdir
    _PATCHED = True
    _log("patched epath gs:// mkdir (orbax needs the prefix to exist right after mkdir)")


def _filesystem(root: str):
    if root.startswith("gs://"):
        import gcsfs

        return gcsfs.GCSFileSystem()
    import fsspec

    return fsspec.filesystem("file")


def _fs_path(path: str) -> str:
    return path[len("gs://") :] if path.startswith("gs://") else path


def step_number(path: str) -> int | None:
    """A canonical `step_<digits>` directory name -> int, else None.

    Strictly rejects orbax's in-progress `step_N.orbax-checkpoint-tmp-*`, which a bare
    `glob('step_*')` would happily return as a checkpoint.
    """
    name = str(path).rstrip("/").rsplit("/", 1)[-1]
    if not name.startswith("step_"):
        return None
    digits = name[len("step_") :]
    return int(digits) if digits.isdigit() else None


def finalized_steps(fs, full_root: str, *, strict: bool = False) -> list[tuple[int, str]]:
    """(step, dir) for every `step_*` under `full_root` carrying the marker, NEWEST first.

    A missing root is `[]` — the genuine-first-launch signal, never an error. `strict`
    propagates a transient list/exists failure instead of swallowing it to `[]`: on the
    resume path a swallowed blip would read as "no checkpoints" and silently restart a
    run from scratch over durable progress, and on multihost it would diverge one host
    from the rest. Garbage collection uses `strict=False` — a gc that raises is worse
    than a gc that skips a round.
    """
    root = _fs_path(full_root).rstrip("/")
    try:
        entries = fs.ls(root, detail=False)
    except FileNotFoundError:
        return []
    except Exception:
        if strict:
            raise
        return []
    out = []
    for entry in entries:
        step = step_number(entry)
        if step is None:
            continue
        directory = f"{root}/{str(entry).rstrip('/').rsplit('/', 1)[-1]}"
        try:
            committed = fs.exists(f"{directory}/{_MARKER}")
        except Exception:
            if strict:
                raise
            committed = False
        if committed:
            out.append((step, directory))
    return sorted(out, reverse=True)


def _all_step_dirs(fs, full_root: str) -> list[tuple[int, str]]:
    """Every `step_*` under `full_root`, finalized or torn, NEWEST first (gc input)."""
    root = _fs_path(full_root).rstrip("/")
    try:
        entries = fs.ls(root, detail=False)
    except Exception:
        return []
    out = [
        (step_number(entry), f"{root}/{str(entry).rstrip('/').rsplit('/', 1)[-1]}")
        for entry in entries
        if step_number(entry) is not None
    ]
    return sorted(out, reverse=True)


# --------------------------------------------------------------------------- #
# collectives
# --------------------------------------------------------------------------- #
def broadcast_int(value: int) -> int:
    """Broadcast an int from the chief to all hosts. Every host must call it equally often."""
    if jax.process_count() == 1:
        return int(value)
    from jax.experimental import multihost_utils

    return int(multihost_utils.broadcast_one_to_all(np.asarray(int(value), dtype=np.int64)))


def allgather_min(value: int) -> int:
    """min over all hosts (== logical AND for 0/1 flags). Collective; all hosts call it."""
    if jax.process_count() == 1:
        return int(value)
    from jax.experimental import multihost_utils

    return int(np.min(multihost_utils.process_allgather(np.asarray(int(value), dtype=np.int32))))


def _sync(name: str) -> None:
    if jax.process_count() == 1:
        return
    from jax.experimental import multihost_utils

    multihost_utils.sync_global_devices(name)


# --------------------------------------------------------------------------- #
# the checkpointer
# --------------------------------------------------------------------------- #
class Checkpointer:
    """Versioned resumable checkpoints under `<gcs_root>/full/step_NNNNNNN`.

    save(step, params, opt_state)  collective, BLOCKING
    save_params_bf16(step, params) collective, BLOCKING — a usable model, not resumable
    restore(step, target)          collective

    Blocking is deliberate. TinyWAM's staged path can background its upload because the
    bytes already sit in host RAM; here the write IS the upload, and backgrounding a
    collective would let the training loop race ahead into the next collective on some
    hosts and not others.
    """

    def __init__(self, gcs_root: str, *, keep_last: int = 2):
        import orbax.checkpoint as ocp

        if not gcs_root:
            raise CheckpointError("Checkpointer needs a root; pass the run's gs:// directory")
        self.gcs_root = gcs_root.rstrip("/")
        self.full_root = f"{self.gcs_root}/full"
        self.keep_last = max(1, int(keep_last))
        if self.gcs_root.startswith("gs://"):
            patch_epath_gcs_mkdir()
        # ONE persistent StandardCheckpointer, with orbax's DEFAULT multiprocessing options:
        # those defaults are exactly what makes every process write its own
        # `ocdbt.process_<i>` shard set. Scoping it to the chief would turn it back into a
        # single-host save. A fresh instance per save races the previous one's atexit thread
        # pool ("cannot schedule new futures after shutdown", seen on a pod).
        self._ckptr = ocp.StandardCheckpointer()
        self._fs = _filesystem(self.gcs_root)
        self._fs_lock = threading.Lock()
        self.save_fail_count = 0
        self.last_save_error: str | None = None
        _log(f"root={self.full_root} keep_last={self.keep_last} (collective writes, every host)")

    # ---------------- paths
    def step_dir(self, step: int) -> str:
        return f"{self.full_root}/step_{int(step):0{_STEP_WIDTH}d}"

    def bf16_dir(self, step: int) -> str:
        return f"{self.gcs_root}/params_bf16_step{int(step):07d}"

    # ---------------- save
    def save(self, step: int, params, opt_state, *, metrics=None) -> bool:
        """Full resumable state and its loss records. True iff EVERY host committed."""
        item = {
            "params": params,
            "opt_state": opt_state,
            "step": np.asarray(int(step), dtype=np.int64),
        }
        directory = self.step_dir(step)
        if not self._publish(item, directory, step, "full", metrics=metrics):
            return False
        self.save_fail_count = 0
        self.last_save_error = None
        _log(f"full step {step} committed by all hosts -> {directory}")
        if jax.process_index() == 0:
            try:
                self._gc()
            except Exception as exc:  # noqa: BLE001 - a gc failure is not a save failure
                _log(f"gc after step {step} failed ({exc}); the checkpoint is safe")
        return True

    def save_params_bf16(self, step: int, params) -> bool:
        """bf16 parameter snapshot: a usable model, kept, not resumable.

        The cast runs on device on every host in lockstep. That is safe precisely because
        all hosts participate — a chief-only `astype` is an eager collective issued by one
        process, which halts the cores.

        HOG training checkpoints intentionally contain ``mot.hog`` in the resumable full
        state, but deployment must be classic FastWAM.  The auxiliary subtree is therefore
        omitted from this inference snapshot.  The removal-safe attention mask guarantees
        that this is a structural export, not an approximation to the trained Action path.
        """
        export = params
        if isinstance(params, dict) and isinstance(params.get("mot"), dict):
            mot = params["mot"]
            if "hog" in mot:
                export = {**params, "mot": {key: value for key, value in mot.items() if key != "hog"}}
        item = jax.tree.map(lambda x: x.astype(jax.numpy.bfloat16), export)
        directory = self.bf16_dir(step)
        if self._publish(item, directory, step, "bf16"):
            _log(f"bf16 snapshot step {step} -> {directory}")
            return True
        return False

    def _publish(self, item, directory: str, step: int, label: str, *, metrics=None) -> bool:
        """Write `item` collectively, then publish the marker LAST.

        Must be called by every host: the orbax save is a collective. Ordering is data
        first, `commit_success.txt` last, so a torn directory is never certified. The
        success decision is agreed with an allgather — a chief-only verdict would diverge
        the pod at the next collective.
        """
        is_chief = jax.process_index() == 0
        # A previous attempt killed mid-write leaves a directory orbax refuses to overwrite.
        # Clear it marker-first on the chief, then everyone proceeds together.
        if is_chief:
            try:
                with self._fs_lock:
                    if self._fs.exists(_fs_path(directory)):
                        self._remove_marker_first(directory)
            except Exception as exc:  # noqa: BLE001 - reported; the save below will raise if fatal
                _log(f"WARNING: pre-clear of {directory} failed ({exc}); the save may raise")
        _sync(f"preclear_{label}_{step}")

        ok = 1
        try:
            self._ckptr.save(directory, item)
            self._ckptr.wait_until_finished()
        except Exception as exc:  # noqa: BLE001 - every host reports; the vote decides
            ok = 0
            _log(f"distributed {label} save of step {step} FAILED here ({type(exc).__name__}: {exc})")
        if not allgather_min(ok):
            if is_chief and label == "full":
                self._record_failure(f"{label} save of step {step} failed on at least one host", directory)
            return False

        commit_ok = 1
        commit_error = ""
        if is_chief:
            try:
                marker_extra = {}
                if label == "full" and metrics is not None:
                    marker_extra["metrics"] = self._write_metrics(step, metrics)
                self._write_marker(directory, step, marker_extra)
            except Exception as exc:  # noqa: BLE001
                commit_ok = 0
                commit_error = f"{type(exc).__name__}: {exc}"
        if not allgather_min(commit_ok):
            if is_chief and label == "full":
                self._record_failure(
                    f"{label} metrics/marker commit step {step} failed ({commit_error})", directory
                )
            return False
        _sync(f"marker_{label}_{step}")
        return True

    def _write_metrics(self, step: int, records) -> dict:
        """Write the chief's loss records before the checkpoint commit marker."""
        payload = b"".join(
            (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
            for record in records
        )
        path = f"{self.gcs_root}/logs/metrics/checkpoint_step_{int(step):0{_STEP_WIDTH}d}.jsonl"
        fs_path = _fs_path(path)
        with self._fs_lock:
            self._fs.makedirs(fs_path.rsplit("/", 1)[0], exist_ok=True)
            self._fs.pipe_file(fs_path, payload)
        return {
            "path": path,
            "records": payload.count(b"\n"),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _write_marker(self, directory: str, step: int, extra: dict | None = None) -> None:
        """The atomic commit point: its own object, written strictly after the data.

        Orbax refuses to restore a directory without it ("incomplete checkpoint") and it
        is the same marker `finalized_steps` scans for, so one write satisfies both.
        """
        marker = {"step": int(step), **(extra or {})}
        with self._fs_lock:
            self._fs.pipe_file(
                f"{_fs_path(directory)}/{_MARKER}", json.dumps(marker, sort_keys=True).encode()
            )

    def _remove_marker_first(self, directory: str) -> None:
        """Delete marker first, so a mid-delete kill leaves an UNfinalized directory that
        restore already ignores — never a marker over half-deleted data."""
        path = _fs_path(directory)
        try:
            self._fs.rm(f"{path}/{_MARKER}")
        except Exception:  # noqa: BLE001 - absent marker is the normal case
            pass
        self._fs.rm(path, recursive=True)

    def _gc(self) -> None:
        """Keep the newest `keep_last` FINALIZED steps and reclaim every step directory
        strictly below the oldest retained one.

        Computing the floor from FINALIZED steps means old finalized directories beyond the
        window go AND abandoned marker-less uploads below the window go — without that
        second half, torn directories leak forever. Torn directories ABOVE the floor are
        left alone (the newest may be an in-flight attempt) and get swept once the floor
        advances. Nothing finalized yet means no safe floor, so skip.
        """
        with self._fs_lock:
            finalized = finalized_steps(self._fs, self.full_root)
            if not finalized:
                return
            floor = finalized[: self.keep_last][-1][0]
            for step, directory in _all_step_dirs(self._fs, self.full_root):
                if step < floor:
                    try:
                        self._remove_marker_first(directory)
                    except Exception as exc:  # noqa: BLE001
                        _log(f"gc/prune of {directory} failed ({exc})")

    def _record_failure(self, message: str, directory: str) -> None:
        """A durable save did not finalize.

        One transient is harmless — the directory has no marker, restore skips it, the next
        cadence retries. A PERSISTENT failure silently loses all progress, so the count is
        surfaced to the loop, which aborts rather than keep burning chips for nothing.
        """
        self.save_fail_count += 1
        self.last_save_error = message
        _log(
            f"WARNING: {message} -> {directory} left UNFINALIZED "
            f"(consecutive durable-save failures: {self.save_fail_count})"
        )

    # ---------------- restore
    def restore(self, directory: str, target):
        """Collective restore of one step directory into `target`'s shapes/shardings."""
        return self._ckptr.restore(directory, target=target)

    def finalized(self, *, strict: bool = False) -> list[tuple[int, str]]:
        return finalized_steps(self._fs, self.full_root, strict=strict)


# --------------------------------------------------------------------------- #
# resume
# --------------------------------------------------------------------------- #
def resume(
    full_root: str,
    build_target,
    *,
    apply_restored,
    allow_fresh: bool,
) -> int:
    """Restore the newest finalized checkpoint under `full_root`; return the NEXT step.

    Steps are 1-indexed: checkpoint N holds N updates, so training continues at N+1. A
    fresh start returns 1.

    The newest-to-oldest fallback runs as a CONSENSUS. Only the chief lists (strictly, so
    a transient read error becomes a coordinated raise rather than a silent `[]` that
    would fresh-start this host while the others restore); it broadcasts one decision per
    round (>=0 restore that step, -1 fresh, -2 raise); every host restores the same step
    and votes on success; any failure advances everyone together. Collective counts stay
    equal on every host, which is the property that keeps the pod from hanging.

    `build_target` is called at most once, on the first restore round, and is expected to
    free the freshly-built device arrays it describes — otherwise the from-scratch state
    and the restored state sit in HBM simultaneously.

    `allow_fresh=False` (an explicit resume_from that resolves to nothing) raises instead
    of quietly starting over: that is the difference between "first launch" and "the
    scaffold is pointing at the wrong path and just threw away two days".
    """
    is_chief = jax.process_index() == 0
    chief_steps: list[int] = []
    chief_error: Exception | None = None
    if is_chief:
        try:
            fs = _filesystem(full_root)
            chief_steps = [step for step, _ in finalized_steps(fs, full_root, strict=True)]
        except Exception as exc:  # noqa: BLE001 - encoded into the broadcast, never raised alone
            chief_error = exc

    target = None
    index = 0
    while True:
        if is_chief:
            if chief_error is not None:
                decision = -2
            elif index < len(chief_steps):
                decision = int(chief_steps[index])
            elif index == 0:
                decision = -1 if allow_fresh else -2
            else:
                decision = -2
        else:
            decision = 0
        decision = broadcast_int(decision)

        if decision == -1:
            _log(f"no finalized checkpoint under {full_root} -> fresh start (step 1)")
            return 1
        if decision == -2:
            raise CheckpointError(
                "resume: no usable checkpoint where one was expected under "
                f"{full_root} (listing error: {chief_error}). Refusing to silently fresh-start "
                "or diverge hosts."
            )

        if target is None:
            target = build_target()
        directory = f"{_fs_path(full_root).rstrip('/')}/step_{decision:0{_STEP_WIDTH}d}"
        if full_root.startswith("gs://"):
            directory = f"gs://{directory}"

        ok = 1
        restored = None
        try:
            import orbax.checkpoint as ocp

            restored = ocp.StandardCheckpointer().restore(directory, target=target)
            if int(restored["step"]) != decision:
                ok = 0
                _log(
                    f"step {decision}: restored step field {int(restored['step'])} != agreed "
                    "-> voting failure"
                )
        except Exception as exc:  # noqa: BLE001 - the vote carries the verdict
            ok = 0
            _log(f"restore of step {decision} FAILED here ({type(exc).__name__}: {exc})")

        if allgather_min(ok):
            apply_restored(restored)
            _log(f"RESUMED from {directory} (all hosts, lockstep): continuing at step {decision + 1}")
            return decision + 1

        # Some host failed -> all advance. A host that SUCCEEDED must free its restored
        # arrays now, or the next round's restore doubles HBM.
        if ok and restored is not None:
            for leaf in jax.tree.leaves(restored):
                try:
                    leaf.delete()
                except Exception:  # noqa: BLE001
                    pass
            del restored
        if is_chief:
            index += 1
