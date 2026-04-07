# 2026-04-06: Bounded vs Unbounded `x_s` Analysis

## Summary

Added a run-specific mathematical analysis explaining why the bounded-with-`u_ref` first-step target produces an `x_s` close to zero while the unbounded frozen-`dhat` target produces a large `x_s`.

New report files:

- `report/bounded_vs_unbounded_xs_analysis.tex`
- `report/bounded_vs_unbounded_xs_analysis.pdf`

## Main Findings

The report uses the actual identified model and the latest two saved runs:

- bounded with `u_ref`:
  - `Data/debug_exports/mpc_first_step_replacement_bounded_frozen_dhat/20260406_232910`
- unbounded:
  - `Data/debug_exports/mpc_first_step_replacement_unbounded_frozen_dhat/20260406_232953`

The main conclusions are:

1. The steady-state implementation is algebraically consistent.
   - `x_s = (I-A)^{-1} B u_s` holds to machine precision in both runs.
   - The run-to-run difference also satisfies
     - `x_s_unbounded - x_s_bounded = (I-A)^{-1} B (u_s_unbounded - u_s_bounded)`
     - to machine precision.

2. The large discrepancy in `x_s` is not evidence of a coding bug.
   - The state map `W = (I-A)^{-1}B` has moderate singular values.
   - The real amplification is on the output side through `G^{-1}`, where
     - `G = C(I-A)^{-1}B`
     - is small, so exact output tracking requires very large `u_s`.

3. In the bounded-with-`u_ref` run, the small `x_s` is driven mainly by the regularization term, not by active box clipping.
   - For the saved bounded run, the target never actually sat on the bounds.
   - With `u_ref_weight = 1.0`, the `u_ref` penalty is orders of magnitude stronger than the tracking curvature `G^T G`.
   - At step 0, the bounded target matches the unconstrained regularized solution almost exactly, before box constraints matter.

## Validation

- Recomputed the actual matrices `A`, `B`, `C`, `W=(I-A)^{-1}B`, and `G=CW`
- Compared the latest bounded and unbounded run bundles numerically
- Compiled the LaTeX report with `pdflatex`

## Notes

- No controller code was changed in this update.
- The report is intended to answer whether the observed `x_s` behavior is structural or a bug; the conclusion is that it is structural and tied to objective scaling.
