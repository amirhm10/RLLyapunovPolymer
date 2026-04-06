# 2026-04-06 Offset-Free Steady-State Debug Analysis

## Why

The offset-free MPC notebook did not have a clean way to inspect the frozen-`dhat` steady-state targets implied by the legacy augmented model during a normal closed-loop disturbance run. The goal was to add that analysis beside the controller, without changing the existing MPC rollout code or feeding any new target back into control.

## What Changed

- Added `analysis/steady_state_debug_analysis.py` as a standalone sidecar module for:
  - building the exact stacked steady-state system
  - computing reduced-form diagnostics
  - solving the per-step frozen-`dhat` steady-state target with exact or least-squares fallback
  - reconstructing per-step closed-loop diagnostics from a completed OF-MPC rollout
  - exporting CSV tables, a pickle bundle, a Markdown summary, and optional plots
- Added `analysis/__init__.py` so the new helper module imports cleanly from notebooks.
- Added `MPCOffsetFree_SteadyStateDebug.ipynb` as a disturbance-only wrapper notebook that:
  - keeps the normal `run_mpc(...)` disturbance simulation
  - builds the offline rollout dictionary after the run
  - calls the sidecar analysis
  - writes artifacts under `Data/mpc_offsetfree_steady_state_debug/<timestamp>/`
  - displays compact structure and per-step summaries in the notebook
- Added a small synthetic smoke-check helper inside the analysis module with one exact case and one least-squares fallback case.

## Validation

- `python -m py_compile analysis\steady_state_debug_analysis.py analysis\__init__.py`
- JSON parse check for `MPCOffsetFree_SteadyStateDebug.ipynb`

## Notes

- `MPCOffsetFree.ipynb`, `Simulation/mpc_run.py`, and `Simulation/mpc.py` were left unchanged.
- The sidecar solve uses the unaugmented `A`, `B`, and `C` matrices from `system_data`.
- The local default `python` still does not have the scientific stack installed, so the synthetic smoke checks and notebook execution were added but not run in this environment.
