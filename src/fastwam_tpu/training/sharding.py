"""Mesh construction and parameter/activation sharding.

Ported from
    codebases/TinyWAM-v4/tinywam/sharding.py — make_mesh / install / batch_spec /
        shard_batch / shard_grads / largest_axis_spec / manual_axes. Our own,
        battle-tested across the v4-32 and v5e campaigns.
    reference/maxdiffusion — the named-axis discipline and the ici_*/dcn_* config
        spelling (report_maxdiffusion.md §4.2).

Facts that constrain the design
    * 6.02 B trainable params. fp32 master + Adam moments ~= 16 B/param ~= 96 GB of
      state, so full-slice FSDP is not optional. v4-32 = 16 chips x 32 GB.
    * num_heads = 24, so head-parallel shards 4 or 8 ways, never 16 (spec §9.2 item 8).
      Nothing here shards by head: ZeRO picks each leaf's LARGEST divisible axis, which
      for every 3072/4096/14336-wide weight is a feature axis, not the head axis.
    * Both experts share the attention width 3072 but not hidden_dim (3072 vs 1024);
      largest_axis_spec is per-leaf, so it needs no per-expert rule.
    * FSDP over the full slice was the v5e unlock (11.4G -> 4.0G) and only works with
      scan_layers on: with 30 layers unrolled XLA hoists every layer's all-gather to the
      top of the graph and holds all 30 gathered copies at once. make_mesh cannot see
      the model, so train.py enforces that pairing.

"The host boundary is not a bandwidth boundary" is load-bearing for putting all 16 chips
of a v4-32 in ONE fsdp axis: every chip in a slice is ICI-connected and bypasses the host
network; DCN only separates SLICES. That is why dcn_*_parallelism must be 1 here, and why
anything else raises rather than silently building a mesh across a network that is not
there.

ZeRO placement is TWO decisions, and the first v4-32 run got the first one wrong
    (1) WHICH axis of each parameter carries 'fsdp' -- `axis_policy`, below.
    (2) Whether the tensors flowing through the layers are PINNED batch-sharded --
        `activations`, applied by the MoT at its layer boundaries.

    "largest" picks a Linear kernel's CONTRACTING axis whenever it is the widest one: the
    FFN down projection [30, 14336, 3072] shards the 14336 that the matmul sums over, and
    [30, 3072, 3072] (attention out, both cross-attention halves) ties and takes the first,
    which is also contracting. With nothing pinning the activations, GSPMD resolves those
    dots by resharding the ACTIVATION onto the contracting axis instead of gathering the
    weight: the v4-32 trace showed `all-gather [32,360,3072] -> [512,360,3072]` (747 ms/step)
    feeding `all-reduce bf16[512,360,3072]` (872 ms/step) -- 1.13 GB per tensor, at the
    GLOBAL batch, to avoid gathering an 18 MB weight.

Measured by compiling the real 6.02 B step on an 8-device CPU mesh at micro-batch 512, in
GB of collective results per microstep (a scan-body collective counted 30x, once per layer):

    arm                          in-scan >0.5 GB   ZeRO weight   one-off    total
    pins off  (the first run)            305.8          49.1       11.9     366.8
    pins on                                0.0          47.3        2.9      50.2
    pins off, HSDP data=2 fsdp=4        1291.0          45.1        6.2    1342.2
    pins on,  HSDP data=2 fsdp=4           0.0          47.3        1.6      49.0

Measured on the real v4-32 slice (2026-08-04, micro-batch 512, 16 chips, all four arms
resumed from the same step-1000 checkpoint so the loss is directly comparable):

    arm                              s/step   smp/s     MFU     HBM
    pins off (the first run)          12.94    39.6    9.8%   8.7 G
    pins on                            4.72   108.4   26.9%   8.2 G     <- 2.74x
    pins on,  HSDP data=2 fsdp=8       4.77   107.4   26.7%  12.7 G
    pins on,  HSDP data=4 fsdp=4        OOM (19.5 G resident before the loop)
    pins off, HSDP data=2 fsdp=8        OOM (45.15 G of HLO temporaries vs 30.75 G)

The last two rows are the sharpest statement of what the pins do, because rows three and
five are the SAME mesh and the same config differing only in the switch: XLA asks for
45.15 G of temporaries without them and runs in 12.7 G with them. Cold compile also
roughly halved (182.6 s to the first step, against 346.9 s).

Step 1005 came out at loss 0.3461 / v 0.3282 / a 0.0179 / |g| 0.212 both with the pins and
without them -- every digit the log prints, grad norm included.

Two things fall out of that table and neither was obvious beforehand:

    * The hidden state is not the expensive tensor -- `t_mod` is. The per-token adaLN
      modulation is [B, Sv, 6, hidden] and `modulation()` upcasts it, so it is 6x the hidden
      state AND fp32 while everything around it is bf16: 13.6 GB against 1.1 GB at this
      microbatch. It rides into the scan as a closure constant, so leaving it unpinned paid
      to reshard it on every one of the 30 iterations. That is most of the 305.8 GB.
    * HSDP is not a lever here, and the slice confirmed it: 4.77 s/step against 4.72 for
      pure FSDP, for 4.5 G/chip more resident state. Its whole argument was shrinking the
      ZeRO ring, and once the pins are in, the ring is ~47 GB of unavoidable weight gather
      that a shorter ring shrinks by about a seventh. At data=4 it does not fit at all --
      18.1 G/chip of parameters and Adam moments leaves 12.19 G, and the in-loop VAE encode
      alone wants a contiguous 15.98 G. The 1342 GB in the table's third row is what an
      unpinned `t_mod` costs when there is a `data` axis to gather it over, not an argument
      against HSDP itself.

Both halves are switches so the combinations are measurable rather than argued about, and
both default OFF so this module's behaviour is unchanged until a config asks for it
(`tpu.param_axis_policy`, `tpu.shard_activations`).
"""

import contextlib

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from ..config import AXIS_POLICIES

__all__ = [
    "ShardingError",
    "AXIS_POLICIES",
    "make_mesh",
    "describe_mesh",
    "install",
    "uninstall",
    "current",
    "enabled",
    "activations_pinned",
    "axis_policy",
    "batch_axes",
    "batch_spec",
    "shard_batch",
    "shard_activation",
    "largest_axis_spec",
    "shard_grads",
    "shard_params",
    "param_shardings",
    "replicate",
    "global_from_host_local",
    "zero_degree",
    "manual_axes",
]

_MESH: Mesh | None = None
_BATCH: tuple[str, ...] = ()
_ENABLED: bool = True
_MANUAL: frozenset[str] = frozenset()
_ACTIVATIONS: bool = False
_AXIS_POLICY: str = "largest"


class ShardingError(ValueError):
    """A mesh/sharding request that cannot be satisfied on this slice."""


# --------------------------------------------------------------------------- #
# mesh
# --------------------------------------------------------------------------- #
def make_mesh(config, devices=None) -> Mesh:
    """Build the device mesh from `cfg.tpu.{mesh_axes, ici_*, dcn_*}`.

    Exactly one parallelism value may be -1 and it fills the remainder; anything that
    does not multiply out to the device count raises.

    Device ORDER is `(process_index, device.id)`, not `jax.devices()`' default. That is
    not cosmetic: it makes each fsdp group a run of ICI-adjacent chips belonging to whole
    hosts, which keeps the ZeRO all-gather ring short and makes a host's shards
    contiguous if anything ever needs to gather them. jax.make_mesh's default order does
    NOT guarantee this on every topology — on v6e-32 it put the ZeRO axis across hosts
    and a host-side gather silently read half a parameter.
    """
    devices = list(jax.devices()) if devices is None else list(devices)
    n_dev = len(devices)
    axes = tuple(config.tpu.mesh_axes)
    if sorted(axes) != ["data", "fsdp"]:
        raise ShardingError(
            f"tpu.mesh_axes must be exactly ['data', 'fsdp'] (in either order), got {list(axes)}"
        )

    for name in ("dcn_data_parallelism", "dcn_fsdp_parallelism"):
        value = int(getattr(config.tpu, name))
        if value != 1:
            raise ShardingError(
                f"tpu.{name}={value}: this harness targets ONE slice (v4-32 = 4 hosts, all "
                "ICI-connected). A DCN axis > 1 means multiple slices, which nothing here has "
                "run on — set it to 1 rather than build a mesh across a network that is not there."
            )

    requested = {
        "data": int(config.tpu.ici_data_parallelism),
        "fsdp": int(config.tpu.ici_fsdp_parallelism),
    }
    for name, size in requested.items():
        if size == 0 or size < -1:
            raise ShardingError(f"tpu.ici_{name}_parallelism must be >= 1 or -1, got {size}")
    fill = [name for name, size in requested.items() if size == -1]
    if len(fill) > 1:
        raise ShardingError(f"at most one of ici_data/ici_fsdp_parallelism may be -1, got {requested}")
    if fill:
        known = 1
        for name, size in requested.items():
            if name not in fill:
                known *= size
        if known <= 0 or n_dev % known:
            raise ShardingError(
                f"cannot fill ici_{fill[0]}_parallelism=-1: {n_dev} devices is not divisible by "
                f"the other axes' product {known}"
            )
        requested[fill[0]] = n_dev // known

    shape = tuple(requested[name] for name in axes)
    product = int(np.prod(shape))
    if product != n_dev:
        raise ShardingError(
            f"mesh {dict(zip(axes, shape))} multiplies out to {product} but this slice has "
            f"{n_dev} devices"
        )

    ordered = sorted(devices, key=lambda d: (d.process_index, d.id))
    return Mesh(np.array(ordered, dtype=object).reshape(shape), axes)


def describe_mesh(mesh: Mesh) -> str:
    """One line, printed at startup: what was actually built, not what was asked for."""
    sizes = {name: int(mesh.shape[name]) for name in mesh.axis_names}
    hosts = sorted({device.process_index for device in mesh.devices.flat})
    return (
        f"mesh {sizes} over {mesh.size} chips on {len(hosts)} host(s); "
        f"batch axes {batch_axes(mesh)}; ZeRO degree {zero_degree(mesh)}; "
        # The EFFECTIVE pin state, not the requested one: with `sharding_constraints=false`
        # the pins are the identity however `shard_activations` was set, and a banner that
        # said "on" there would be the one line everybody trusts and nobody rechecks.
        f"axis_policy={_AXIS_POLICY}; activation pins "
        f"{'on' if activations_pinned() else 'off'}"
        f"{' (asked for, but constraints are off)' if _ACTIVATIONS and not activations_pinned() else ''}"
    )


def zero_degree(mesh: Mesh) -> int:
    """How many ways parameters are ZeRO-sharded: the size of the 'fsdp' axis."""
    return int(mesh.shape["fsdp"])


def batch_axes(mesh: Mesh) -> tuple[str, ...]:
    """Mesh axes the batch dimension is sharded over — ALL of them.

    A sample lives on exactly one chip, so both the data axis and the fsdp axis split the
    batch; the fsdp axis additionally splits the parameters. Under pure FSDP the data axis
    has size 1 and contributes nothing, but naming it keeps the spec valid for a mixed 2-D
    mesh without a branch.
    """
    return tuple(mesh.axis_names)


# --------------------------------------------------------------------------- #
# the process-wide singleton
# --------------------------------------------------------------------------- #
def install(
    mesh: Mesh,
    constraints: bool = True,
    *,
    activations: bool = False,
    axis_policy: str = "largest",
) -> None:
    """Register the trainer's mesh as the process-wide singleton.

    `constraints=False` keeps the mesh available but turns every `shard_batch` /
    `shard_grads` into the identity — the escape hatch back to letting GSPMD decide
    everything, for isolating a suspected constraint bug.

    `activations=True` additionally turns on `shard_activation`, the in-model pins that
    keep hidden states batch-sharded between layers. `axis_policy` chooses which axis of
    a parameter carries 'fsdp'. Both are described in the module docstring; both are
    process-wide rather than threaded through every call site because `shard_grads`,
    `param_shardings` and the optimizer-state placement in `train.py` all have to make the
    SAME choice — a per-call argument is a way for one of the three to disagree, and a
    parameter whose gradient is pinned to a different axis than the parameter itself is a
    silent reshard on every step.
    """
    global _MESH, _BATCH, _ENABLED, _ACTIVATIONS, _AXIS_POLICY
    if axis_policy not in AXIS_POLICIES:
        raise ShardingError(f"axis_policy must be one of {AXIS_POLICIES}, got {axis_policy!r}")
    _MESH = mesh
    _BATCH = batch_axes(mesh)
    _ENABLED = bool(constraints)
    _ACTIVATIONS = bool(activations)
    _AXIS_POLICY = axis_policy


def uninstall() -> None:
    """Reset to the not-installed state (tests that iterate over meshes)."""
    global _MESH, _BATCH, _ENABLED, _ACTIVATIONS, _AXIS_POLICY
    _MESH, _BATCH, _ENABLED = None, (), True
    _ACTIVATIONS, _AXIS_POLICY = False, "largest"


def current() -> Mesh | None:
    return _MESH


def enabled() -> bool:
    """True iff constraints are active (mesh installed, >1 device, switch on)."""
    return _ENABLED and _MESH is not None and _MESH.size > 1


def activations_pinned() -> bool:
    """True iff `shard_activation` is anything but the identity."""
    return _ACTIVATIONS and enabled()


def axis_policy() -> str:
    """The installed ZeRO axis policy, for logging and tests."""
    return _AXIS_POLICY


@contextlib.contextmanager
def manual_axes(axes):
    """Mark mesh axes as MANUAL (owned by an enclosing shard_map) for the duration.

    Nothing here uses a shard_map: the fused accumulation deliberately stays in GSPMD
    auto so remat composes. TinyWAM's deferred-reduce variant, which does use one, is a
    known dead end (`project_v6e_gradaccum_defer`). The hook exists so a future
    experiment does not have to rewrite the constraint helpers.
    """
    global _MANUAL
    prev = _MANUAL
    _MANUAL = frozenset(axes)
    try:
        yield
    finally:
        _MANUAL = prev


def _active_batch_axes() -> tuple[str, ...]:
    return tuple(a for a in _BATCH if a not in _MANUAL)


# --------------------------------------------------------------------------- #
# constraints
# --------------------------------------------------------------------------- #
def batch_spec(ndim: int) -> P:
    """Batch-on-axis-0 PartitionSpec for a rank-`ndim` activation."""
    return P(_active_batch_axes(), *([None] * (ndim - 1)))


def shard_batch(x: jax.Array) -> jax.Array:
    """Pin an activation's batch (axis 0) sharding; identity when not enabled."""
    if not enabled():
        return x
    return jax.lax.with_sharding_constraint(x, NamedSharding(_MESH, batch_spec(x.ndim)))


def shard_activation(x: jax.Array) -> jax.Array:
    """Pin a hidden state's batch sharding INSIDE the model; identity unless `activations`.

    Separate from `shard_batch` only in its switch: `shard_batch` pins what the trainer
    hands the model (latents, the microbatch fields) and is part of the memory contract,
    while this one pins what flows BETWEEN layers and exists to take a choice away from
    GSPMD. Keeping them independent is what lets the throughput matrix isolate which of
    the two the win came from.

    Why a pin changes anything: a Linear whose kernel is sharded on its contracting axis
    can be lowered either as "gather the weight, keep the batch split" or as "gather the
    activation to the global batch, compute partial sums, all-reduce". The second is
    catastrophically more traffic here (1.13 GB vs 18 MB per tensor on v4-32) and is what
    XLA picked. Requiring the layer's output to stay batch-sharded makes the second
    lowering unable to satisfy the constraint without an extra reshard, so it loses.
    """
    if not _ACTIVATIONS:
        return x
    return shard_batch(x)


def largest_axis_spec(x, n_shards: int, *, policy: str | None = None) -> P:
    """ZeRO spec: which axis of this leaf carries 'fsdp'. `policy` defaults to the installed one.

    "largest" (the original): the largest `n_shards`-divisible axis, ties to the first.
    "last": the LAST divisible axis, walking from the end.

    Either way a leaf is split `n_shards` ways and holds the same bytes per chip — the axis
    choice is not a memory decision, it is a decision about which matmul lowering XLA is
    tempted by. For every Linear and Conv in this model the last axis is the OUTPUT
    feature axis, so "last" is exactly "never shard an axis a matmul contracts over":
    [30,14336,3072] and [30,3072,3072] move off the contracting 14336/3072 and onto the
    output 3072, while [30,3072,14336] (FFN up) does not move at all because its widest
    axis was already the output one. The walk falls back toward axis 0 when the last axis
    is not divisible, which is what keeps the tiny non-Linear leaves (the action head's
    [1024, action_dim], say) sharded rather than replicated.

    Scalars and leaves with no divisible axis stay replicated; at 6 B params those are the
    norm scales and biases, well under 0.1% of the tree.
    """
    policy = _AXIS_POLICY if policy is None else policy
    if policy not in AXIS_POLICIES:
        raise ShardingError(f"axis_policy must be one of {AXIS_POLICIES}, got {policy!r}")
    ndim = len(x.shape)
    if ndim == 0:
        return P()
    if policy == "last":
        axis = next((i for i in reversed(range(ndim)) if x.shape[i] % n_shards == 0), None)
        if axis is None:
            return P()
    else:
        axis = max(range(ndim), key=lambda i: x.shape[i])
        if x.shape[axis] % n_shards != 0:
            return P()
    return P(*[("fsdp" if i == axis else None) for i in range(ndim)])


def shard_grads(grads, n_shards: int | None):
    """Pin every gradient leaf to its parameter placement.

    Without this the backward produces full-size partial sums and lowers the cross-chip
    reduction as an all-reduce into a fully REPLICATED gradient tree; at 6.02 B params
    that is 24 GB/chip in fp32 and does not fit. Constraining each leaf to the ZeRO
    layout makes XLA emit a reduce-scatter instead, so the gradient tree is 1.5 GB/chip
    at 16-way. This is the single constraint that decides whether the run fits.

    `n_shards=None` means parameters are replicated (debug only) -> replicate grads too.
    """
    if not enabled():
        return grads
    spec = (lambda g: P()) if n_shards is None else (lambda g: largest_axis_spec(g, n_shards))
    return jax.tree.map(
        lambda g: jax.lax.with_sharding_constraint(g, NamedSharding(_MESH, spec(g))), grads
    )


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def param_shardings(tree, mesh: Mesh, *, shard: bool = True):
    """The NamedSharding for every leaf of a parameter/optimizer tree."""
    n_shards = zero_degree(mesh)
    spec = (lambda x: largest_axis_spec(x, n_shards)) if shard else (lambda x: P())
    return jax.tree.map(lambda x: NamedSharding(mesh, spec(x)), tree)


def shard_params(tree, mesh: Mesh, *, shard: bool = True):
    """Move a host-resident (numpy) parameter tree onto the mesh in its ZeRO layout.

    `jax.make_array_from_callback` rather than `device_put`: on multihost, device_put
    rejects a global-mesh target ("not addressable devices"), and the callback form lets
    every host materialise only its own shards out of the full host-side array. Every
    host built the same full tree on CPU, so the shards agree by construction.
    """
    shardings = param_shardings(tree, mesh, shard=shard)

    def place(x, sharding):
        host = np.asarray(x)
        return jax.make_array_from_callback(host.shape, sharding, lambda index, _h=host: _h[index])

    return jax.tree.map(place, tree, shardings)


def replicate(tree, mesh: Mesh):
    """Put a host-resident tree on every chip (the frozen VAE weights)."""
    return shard_params(tree, mesh, shard=False)


def global_from_host_local(array, mesh: Mesh):
    """Host-local numpy batch -> global jax.Array sharded over every mesh axis.

    Each host contributes its own slice of the global batch; the concatenation order is
    process order, which is the order `make_mesh` laid the devices out in.
    """
    sharding = NamedSharding(mesh, batch_spec(array.ndim))
    global_shape = (jax.process_count() * array.shape[0], *array.shape[1:])
    return jax.make_array_from_process_local_data(sharding, array, global_shape)
