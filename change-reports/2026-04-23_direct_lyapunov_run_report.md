# Direct Lyapunov Run Report

## Summary

Added a full report for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` covering:

- the direct frozen-output-disturbance Lyapunov MPC architecture
- the target-solver mathematics
- the hard and soft Lyapunov MPC formulations
- the April 23, 2026 saved run from `Data/debug_exports/direct_lyapunov_mpc_unbounded_hard/20260423_181204`
- copied figures from that run under `report/figures/direct_lyapunov_mpc_frozen_output_disturbance/`
- comparison against `LyapunovSafetyFilterMPCTargetSelectorTermAblation.ipynb`
- literature context for offset-free MPC, tracking MPC, Lyapunov MPC, and predictive safety filters

The key run interpretation is that the frozen target solve succeeds exactly, but the `unbounded` target is outside the admissible input box at all 1600 steps. The `hard` direct MPC solve is therefore infeasible at every step and the rollout holds the previous input.

## Files

- `report/direct_lyapunov_mpc_frozen_output_disturbance.tex`
- `report/direct_lyapunov_mpc_frozen_output_disturbance_run_report.md`
- `report/figures/direct_lyapunov_mpc_frozen_output_disturbance/*.png`
- `report/README.md`

## Validation

- Checked the LaTeX source has balanced top-level document, table, figure, itemize, enumerate, abstract, and bibliography environments.
- Checked the LaTeX source is ASCII-only.
- Confirmed the copied report figures are present.
- Attempted to rebuild `report/direct_lyapunov_mpc_frozen_output_disturbance.pdf`, but `pdflatex` is not installed in this sandbox.

## Notes

The browsable Markdown report is the easiest current entrypoint because it renders the figures directly in the IDE. The LaTeX source is ready for PDF rebuild in an environment with `pdflatex`, `xelatex`, `lualatex`, or another compatible TeX engine installed.
