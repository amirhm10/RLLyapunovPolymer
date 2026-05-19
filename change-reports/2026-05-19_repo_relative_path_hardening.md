# Repository-Relative Path Hardening

## Summary

This update removes runtime dependence on the current working directory for common `Data/`, `results/`, and debug-export paths.
It also replaces stale absolute notebook checkpoint paths that still pointed at the old OneDrive project location.

## What Changed

- Added `utils/path_helpers.py` with repository-root helpers.
- Updated Python modules that previously defaulted to `os.getcwd()` so they now resolve paths from the repository root:
  - `utils/td3_helpers.py`
  - `Simulation/sys_ids.py`
  - `Plotting_fns/mpc_plot_fns.py`
  - `Plotting_fns/rl_plots.py`
  - `Lyapunov/direct_lyapunov_mpc.py`
  - `Lyapunov/safety_debug.py`
  - `analysis/steady_state_debug_analysis.py`
- Updated notebook source cells that built repo paths from `os.getcwd()` to use the repository-root helper instead.
- Replaced old absolute agent checkpoint references in legacy RL notebooks with `Data/agent_2507171027.pkl`.

## Validation

- `py_compile ok 8` on the touched Python modules.
- `notebook json ok 23` on the touched notebooks.
- Confirmed no remaining `os.getcwd()` calls in active Python modules outside archived snapshot files.
- Confirmed no remaining old OneDrive absolute paths in notebook source cells.

## Notes

- Existing saved notebook outputs and historical result bundles may still display old absolute paths inside their saved text or JSON payloads. That does not affect new runs after this hardening update, but it can still appear in historical outputs.
- A clean Git commit was not created during this task because the worktree already contained many unrelated in-progress changes before the path hardening started.
