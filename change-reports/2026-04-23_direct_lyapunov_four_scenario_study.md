# Direct Lyapunov Four-Scenario Study

## Summary

Implemented the four-scenario direct frozen-output-disturbance Lyapunov MPC study workflow.

The notebook now runs:

- `unbounded_hard`
- `bounded_hard`
- `unbounded_soft`
- `bounded_soft`

Each case uses identical plant, disturbance, setpoint, horizon, weighting, and failure-policy settings. Only `target_mode` and `lyapunov_mode` vary. Hard infeasibility is logged as a diagnostic result rather than treated as a notebook failure.

## Main Changes

- Added direct-study helpers in `Lyapunov/direct_lyapunov_mpc.py` for:
  - physical-output RMSE
  - comparison-table records
  - comparison CSV/PKL/JSON artifacts
  - comparison plots for reward, RMSE, solver/contraction rates, slack, target diagnostics, and cross-case output/input overlays
- Extended `save_direct_lyapunov_debug_artifacts(...)` with:
  - `step_table.pkl`
  - optional `timestamp_subdir=False` so notebooks can save directly into a case folder
- Rewrote `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` around:
  - one visible study configuration
  - shared plant/controller setup
  - reusable `run_case(...)`
  - a default four-case loop
  - study-level comparison artifact saving
  - report-figure copy support
- Updated the Markdown and LaTeX reports to describe the four-scenario workflow and preserve the previous `unbounded_hard` run as prior diagnostic context.

## Validation

- `python -m py_compile Lyapunov\direct_lyapunov_mpc.py`
- Notebook JSON parse check for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
- Report image-link check for the currently tracked Markdown report

## Not Run

- Full notebook execution was not run in this sandbox because only the default Python interpreter is visible, and the project dependencies are not installed there.
- `python -m Lyapunov.direct_lyapunov_smoke_tests` was attempted and stopped at `ModuleNotFoundError: No module named 'numpy'`.
- `nbformat.validate(...)` was not run because `nbformat` is not installed in the default Python environment.
- PDF rebuild was not run because no TeX engine is available on PATH.

## Notes

Runtime four-scenario exports remain under ignored `Data/debug_exports/`. Report-ready figures are copied by the notebook into `report/figures/direct_lyapunov_mpc_frozen_output_disturbance/` when the full plotted study is run.
