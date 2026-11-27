"""How the action rows split into independent tau copies. One definition, four readers.

A training step normally supervises the action tower at ONE flow time per sample: the
objective is an integral over tau and each step draws B points of it. Copies buy more of
those points for almost nothing. The action block is 32 of the 392 joint tokens and the
joint mask's video->action half is all False, so appending M-1 more copies of the same
chunk leaves the 5 B video expert's work untouched and only widens the narrow tower.

The packed action stream is copy-major, with no anchor rows (our proprio token rides in
the text context, not in the action stream):

    [ copy_0 (L0 rows) | copy_1 (L1) | ... | copy_{M-1} (L_{M-1}) ]    Sa = sum(L_i)

Copy i holds the FIRST L_i ground-truth steps -- a prefix of the same future from the
same observation -- carrying its own tau and its own noise draw. Unequal lengths are
therefore a horizon mix: `(32, 16, 8, 4)` trains "predict the next 32/16/8/4 steps",
which is the shape inference takes when only part of a chunk is executed.

Four modules must agree on this split exactly, which is why the arithmetic lives here
and not in any of them:

    config.py               resolve_copy_lengths: the yaml knobs -> the tuple
    models/mot.py           the block-diagonal attention mask
    models/action_dit.py    the RoPE table (positions restart per copy)
    models/fastwam.py       packing the targets and averaging the loss

Disagreement between any two of them is silent: a copy masked at one offset and rope'd at
another has entirely legal shapes, raises nothing, and trains a different model than the
one on the page.

Two properties this layout was chosen for, both learned upstream (TinyWAM
`feat/multi-tau-action`):

* **Copies cannot see each other.** A copy that could read a sibling would be reading the
  same chunk at another noise level -- something inference never provides. Blocks on the
  diagonal, nothing off it.
* **Variable lengths live on the copy axis, not the sample axis.** Per-sample random
  horizons would make the mask a runtime value; per-copy lengths keep it a compile-time
  constant, and a short copy's missing rows are structurally absent rather than present
  and ignored. Upstream retired a per-sample `dynamic_horizon` knob over exactly this: its
  out-of-horizon rows were pinned to pure noise but stayed *visible* to the valid rows,
  so the only thing telling the model to ignore them was a learned response to sigma.
"""

__all__ = ["CopyLengthError", "normalize_copy_lengths", "resolve_copy_lengths"]


class CopyLengthError(ValueError):
    """A copy layout that cannot describe a correct packed action stream."""


def normalize_copy_lengths(copy_lengths, total_rows: int, what: str = "action rows") -> tuple[int, ...]:
    """Validate a per-copy row count against the stream it has to fill.

    `None` means one copy spanning everything, i.e. the single-chunk default that every
    pre-copies caller wants -- so a reader that has no opinion passes None and gets the
    old behaviour with no branch of its own.

    Args:
      copy_lengths: an iterable of per-copy row counts, or None.
      total_rows: the packed action stream's length, which the counts must sum to.
      what: what `total_rows` is, for the error message.
    """
    if copy_lengths is None:
        return (int(total_rows),)
    lengths = tuple(int(length) for length in copy_lengths)
    if not lengths:
        raise CopyLengthError("copy_lengths must not be empty (None = one full-length copy)")
    if any(length < 1 for length in lengths):
        raise CopyLengthError(f"every copy length must be >= 1, got {lengths}")
    if sum(lengths) != int(total_rows):
        raise CopyLengthError(
            f"copy_lengths {lengths} sum to {sum(lengths)}, but {what} is {total_rows}"
        )
    return lengths


def resolve_copy_lengths(tau_copies: int, copy_lengths, action_horizon: int) -> tuple[int, ...]:
    """The yaml knobs -> per-copy row counts. The one place the two spellings meet.

    `tau_copies = M` is shorthand for M copies of the FULL horizon; `copy_lengths` gives
    the lengths explicitly and decides the count itself. They may not disagree.

    This is a free function rather than three lines inside the config builder because
    upstream shipped those three lines and they were wrong: `tau_copies=4` was translated
    through the `None` branch above as "one copy of 4*32 rows" and produced `(128,)`
    instead of `(32, 32, 32, 32)`. Every one of that project's copy tests passed, because
    all of them handed the model layer an explicit tuple and none went through the
    translation. It surfaced on a live TPU slice. A row-count guard turned it into an
    immediate crash; without one the model would have trained on a single 128-row chunk
    with RoPE running 0..127 -- correct shapes, no error, and not the experiment anybody
    asked for.
    """
    horizon = int(action_horizon)
    if horizon < 1:
        raise CopyLengthError(f"action_horizon must be >= 1, got {horizon}")

    if copy_lengths is None:
        count = int(tau_copies)
        if count < 1:
            raise CopyLengthError(f"tau_copies must be >= 1, got {count}")
        return (horizon,) * count

    lengths = tuple(int(length) for length in copy_lengths)
    # Validate the list on its own terms -- it defines its own total -- then check it
    # against the data separately, so the two failures read differently.
    lengths = normalize_copy_lengths(lengths, sum(lengths), "their own sum")
    if int(tau_copies) not in (1, len(lengths)):
        raise CopyLengthError(
            f"copy_lengths has {len(lengths)} copies but tau_copies={int(tau_copies)}: set "
            f"tau_copies to 1 (the list defines the count) or to the same number"
        )
    if max(lengths) > horizon:
        raise CopyLengthError(
            f"copy_lengths {lengths} asks for a copy longer than the {horizon} action steps "
            f"the data provides; a copy is a PREFIX of the chunk, never an extension of it"
        )
    return lengths
