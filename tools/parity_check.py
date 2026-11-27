"""Module-level parity: compare our JAX modules against a torch golden.

M1 implements this; it is the tool every later gate leans on.

Pattern from
    reference/maxdiffusion/src/maxdiffusion/tests/wan/wan_common_module_parity_test.py
        JAX_PLATFORMS=cpu so torch and jax run on the same device at the same
        precision; copy weights module-by-module; assert_allclose per module.
    codebases/TinyWAM-v4/tools/convert_wan.py — the "fixed tiny fp32 input, compare
        against the reference, then read back from gs:// and compare again" loop.

Scope limit (fastwam_tpu_plan.md decision 6): we do NOT run official FastWAM PyTorch
code. Torch goldens may only come from third-party components — diffusers' Wan2.2
modules and maxdiffusion's existing parity tests. Everything FastWAM-specific
(MoT, ActionDiT, the masks, the loss) is validated instead by:
    * structural assertions after loading the official checkpoint (exact key set,
      exact shapes, nothing missing, nothing extra), and
    * end-to-end behaviour: the closed-loop score against the published numbers, and
      the reference's own internal equivalence check (cached action path vs joint
      path within 1e-2).

Tolerances (spec §10): fp32 ~1e-5, bf16 ~1e-2. Report max abs and max rel error per
module; do not average them away.

Priority order, highest risk first
    1. RoPE tables + apply_rope, elementwise. The interleaved-pair convention and the
       fp64 -> fp32 table build are the top-ranked project risk.
    2. sinusoidal_embedding_1d, including the [cos; sin] ordering.
    3. RMSNorm (fp32 normalize, cast back, then multiply by the weight).
    4. DiTBlock and Head.
    5. The whole video tower under the per-token-timestep path.
"""

__all__ = ["main"]


def main(argv: list[str] | None = None) -> None:
    raise NotImplementedError("M1: parity harness")
