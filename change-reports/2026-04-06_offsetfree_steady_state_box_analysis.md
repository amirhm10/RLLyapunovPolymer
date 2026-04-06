# 2026-04-06 Offset-Free Steady-State Box Analysis

## Why

The existing offset-free steady-state sidecar only analyzed the unbounded frozen-`dhat` steady-state target. A second analysis layer was needed to check whether that exact target lies inside the MPC input box and, when it does not, compute an analysis-only box-constrained least-squares target for comparison.

## What Changed

- Extended `analysis/steady_state_debug_analysis.py` with a bounded analysis branch while preserving the original unbounded outputs.
- Added new helper functions for:
  - exact unbounded compatibility wrapping
  - box-membership checks
  - box-constrained least-squares fallback solves
  - per-step bounded sidecar aggregation across the rollout
- The bounded branch now uses `system_data["b_min"]` and `system_data["b_max"]` as the steady-state input box in scaled-deviation coordinates.
- Added bounded diagnostics to the saved bundle, including:
  - exact-within-bounds flags
  - exact violation magnitudes
  - bounded steady-state targets
  - bound activity masks
  - box solve modes and summary tables
- Added new bounded-analysis plots and new CSV exports:
  - `box_overall_summary.csv`
  - `box_per_input_activity.csv`
  - `box_event_table.csv`
- Updated `MPCOffsetFree_SteadyStateDebug.ipynb` to enable the bounded branch and display the new summary tables.
- Added documentation:
  - `report/steady_state_box_analysis_parameters.md`
  - `report/steady_state_box_analysis.tex`

## Validation

- `python -m py_compile analysis\steady_state_debug_analysis.py analysis\__init__.py`
- JSON parse check for `MPCOffsetFree_SteadyStateDebug.ipynb`

## Notes

- The existing unbounded analysis outputs were kept intact.
- The controller path remains unchanged.
- The bounded branch prefers `scipy.optimize.lsq_linear`; if SciPy is unavailable, the bounded solve is marked as failed instead of crashing the sidecar.
- The LaTeX source was added, but PDF compilation was not run in this environment.
