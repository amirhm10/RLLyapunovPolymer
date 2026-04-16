# Direct Lyapunov Plotting Integration

## Summary

This follow-up change reworked the direct frozen-output-disturbance Lyapunov plotting path to reuse the repository's existing CSTR MPC plotting helper instead of keeping the direct exporter fully standalone.

The direct exporter now saves:

- the canonical shared CSTR output/input figures from `Plotting_fns/mpc_plot_fns.py`
- the richer direct Lyapunov diagnostic figures already specific to the new controller

## Main Changes

### Shared MPC plotter enrichment

Updated `Plotting_fns/mpc_plot_fns.py` so `plot_mpc_results_cstr(...)` now supports optional overlays and direct-save usage:

- optional steady target output overlay
- optional stage tracking target overlay
- optional steady input target overlay
- optional physical input bounds
- optional save into an already-created directory without adding another timestamp subdirectory
- shared plot-style colors via `utils.plot_style`

The existing notebook callers remain compatible because the original core arguments and default behavior are preserved.

### Direct exporter integration

Updated `Lyapunov/direct_lyapunov_mpc.py` so the direct plotting path now:

- calls the shared `plot_mpc_results_cstr(...)` helper when bundle scaling metadata is available
- saves the canonical figure set directly into the direct run plot directory
- keeps the direct-specific Lyapunov and target diagnostics alongside those shared plots

### Enriched direct target diagnostics

The direct step log and bundle now also carry bounded-target activity information:

- bounded active lower-count trace
- bounded active upper-count trace
- exact active lower/upper counts in the step table
- input-bound overlays in the direct input plots

That information is now plotted in the direct target diagnostics panel and exported into the bundle summary/CSV flow.

## Validation

Executed with `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe`.

- `python -m py_compile Plotting_fns\mpc_plot_fns.py Lyapunov\direct_lyapunov_mpc.py`
- `python -m Lyapunov.direct_lyapunov_smoke_tests`
- notebook JSON validation for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
- direct export smoke run confirming these files are emitted together:
  - `fig_mpc_outputs_full.png`
  - `fig_mpc_outputs_last20.png`
  - `fig_mpc_inputs_full.png`
  - `fig_mpc_inputs_last20.png`
  - `01_outputs_vs_targets.png`
  - `02_inputs_vs_targets.png`
  - `03_state_target_error.png`
  - `04_lyapunov_diagnostics.png`
  - `05_target_diagnostics.png`
  - `06_tail_window_summary.png`

## Notes

- The direct export smoke run still shows the pre-existing observer-gain warning from `Simulation/mpc.py`, but plotting/export completed successfully.
- This change intentionally leaves the unrelated `.agents/` worktree and the user-deleted planning scratch files untouched.
