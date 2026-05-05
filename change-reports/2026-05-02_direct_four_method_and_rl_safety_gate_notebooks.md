# Direct Four-Method And RL Safety-Gate Notebooks

## Summary

This change implements the four-method direct Lyapunov disturbance study and the matching RL safety-gate notebooks requested on 2026-05-02.

The main backend addition is a new `projection_backend="direct_accept_or_fallback"` path in [`Simulation/run_rl_lyapunov.py`](../Simulation/run_rl_lyapunov.py). It uses the exact direct bounded frozen-output-disturbance target family:

- accept the RL candidate only if it already satisfies the direct Lyapunov contraction test
- otherwise solve the direct Lyapunov MPC fallback
- if the direct target or fallback solve fails, keep the current direct semantics and hold the previous input
- push the executed action, not the raw RL proposal, into replay

The notebook layer now has three new experiment entrypoints:

- [`DirectLyapunovMPC_FourMethodDisturbance.ipynb`](../DirectLyapunovMPC_FourMethodDisturbance.ipynb)
- [`DirectLyapunovSafetyGateRL_Pretrained.ipynb`](../DirectLyapunovSafetyGateRL_Pretrained.ipynb)
- [`DirectLyapunovSafetyGateRL_ColdStart.ipynb`](../DirectLyapunovSafetyGateRL_ColdStart.ipynb)

All three use:

- two setpoints
- `set_points_len = 400`
- `n_tests = 200`
- Rawlings output-disturbance augmentation
- the same four direct target variants

## Main Code Changes

### Shared study definition

Added [`utils/direct_lyapunov_study.py`](../utils/direct_lyapunov_study.py) with:

- the two-setpoint disturbance study constants
- the shared `TEST_CYCLE` helper
- the four direct target method definitions

This removes notebook-level duplication for the case list and base schedule.

### Direct Lyapunov backend reuse

Refactored [`Lyapunov/direct_lyapunov_mpc.py`](../Lyapunov/direct_lyapunov_mpc.py) so the direct rollout logic is reusable outside the original notebook:

- `direct_lyapunov_evaluation_ingredients(...)`
- `prepare_direct_output_disturbance_step(...)`
- `solve_direct_tracking_from_target(...)`

`run_direct_output_disturbance_lyapunov_mpc(...)` now uses the shared helper path instead of duplicating the step-level target and fallback solve logic inline.

The direct debug/comparison exporters also now write `figure_manifest.json`.

### RL direct safety gate

Extended [`Simulation/run_rl_lyapunov.py`](../Simulation/run_rl_lyapunov.py):

- added `projection_backend="direct_accept_or_fallback"`
- added `direct_target_mode`, `direct_target_config`, and `direct_tracking_use_target_output`
- added `disturbance_after_step` so the RL notebooks can match the direct disturbance timing
- added default weight fallback logic so the runner works with both the legacy `MpcSolver` and the direct Lyapunov solver

The direct gate path records the requested correction modes:

- `accepted_candidate`
- `fallback_mpc_verified`
- `fallback_mpc_unverified`
- `target_fail_hold_prev`
- `solver_fail_hold_prev`

### Safety/debug exports

Extended [`Lyapunov/safety_debug.py`](../Lyapunov/safety_debug.py) so RL direct-gate runs automatically save:

- direct target diagnostics such as `target_cond_M`, `target_cond_G`, `target_residual_total_norm`
- anchor/smoothness activation fields and mismatch magnitudes
- executed-action gap metrics `||u_exec - u_rl||_inf`
- `episode_table.csv` and `episode_table.pkl`
- `figure_manifest.json`
- cross-method comparison tables and plots through `save_safety_filter_comparison_artifacts(...)`

## Notebook Outputs

The new RL notebooks now save, per method:

- debug bundle
- summary JSON/CSV
- step table CSV/PKL
- episode table CSV/PKL
- NPZ arrays
- paper plots
- final trained agent checkpoint

The cross-method study root also saves:

- `comparison_table.csv`
- `comparison_table.pkl`
- `comparison_summary.json`
- `figure_manifest.json`
- comparison plots for reward, RMSE, correction modes, gap distribution, episode fallback counts, and last-episode overlays

## Validation

### Compiler and notebook checks

- `python -m py_compile ...` still fails in this repo because Windows/OneDrive blocks the final `__pycache__` rename
- completed in-memory `compile(...)` checks for:
  - `utils/direct_lyapunov_study.py`
  - `Lyapunov/direct_lyapunov_mpc.py`
  - `Simulation/run_rl_lyapunov.py`
  - `Lyapunov/safety_debug.py`
- validated all three new notebooks with `nbformat.validate(...)`
- compiled all code cells in the three new notebooks with `compile(...)`

### Smoke runs

Used the existing `rl-env` kernel interpreter at:

- `C:\\Users\\HAMEDI\\miniconda3\\envs\\rl-env\\python.exe`

Smoke checks completed:

1. RL direct gate accept path
   - direct gate produced `accepted_candidate`
   - replay stored the executed action, with max difference about `3.27e-10`

2. RL direct gate fallback path
   - aggressive candidate produced `fallback_mpc_verified`

3. Combined direct method
   - saved bundle summary showed both activations present:
   - `target_u_ref_active_steps = 1`
   - `target_x_ref_active_steps = 1`

## Notes

- The smoke runs still emit the existing observer pole-placement warning from [`Simulation/mpc.py`](../Simulation/mpc.py); this was already present and was not changed here.
- I did not run the full 200-episode notebooks end-to-end as part of low-cost validation.
