# Debug-first export cleanup and safety-aware plots

Date: 2026-05-18

## Summary

Updated the direct Lyapunov MPC and RL safety-gate export paths so active debugging remains the default, while result folders and artifacts are easier to inspect.

## Changes

- Added an `export_profile` option to both exporters.
  - `debug` is the default and keeps rich diagnostic arrays.
  - `compact` writes a smaller paper/analysis-oriented subset.
- Replaced normal hashed safety/direct fallback folders with sanitized method-name folders such as `mixed` or `bounded_hard`.
- Stopped pretrained and cold-start RL notebooks from saving `trained_agent_*.pkl` by default.
  - Added `SAVE_TRAINED_AGENT = False`.
  - The old save path remains available behind this flag.
- Slimmed default `step_table.csv` output by removing large JSON vector/detail columns while keeping scalar diagnostics for tracking, target selection, fallback, safety status, and Lyapunov behavior.
- Added `safety_active_flags` to safety-gate `arrays.npz`.
- Added reward moving-average and episode-average reward plots.
- Redesigned every-10-episode safety-gate snapshots to include:
  - output tracking with safety-active shading
  - candidate versus executed input traces
  - fallback input traces when available
  - a dedicated safety active/fallback status strip
  - detailed gate/fallback diagnostics
- Added a direct MPC reward summary plot for consistency.

## Validation

- `python -m py_compile Lyapunov/safety_debug.py Lyapunov/direct_lyapunov_mpc.py`
- Notebook JSON validation for:
  - `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
  - `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

## Notes

The debug profile intentionally remains the default because the current experiments are still being diagnosed. The compact profile is available for later paper-facing exports without changing the debugging workflow.
