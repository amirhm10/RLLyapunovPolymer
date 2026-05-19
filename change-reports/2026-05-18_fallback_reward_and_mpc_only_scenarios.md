# Fallback-Aware Reward And MPC-Only Scenarios

## Summary

Implemented a fallback mismatch penalty in the active RL reward path and added an `mpc_only` comparison scenario across the pretrained RL, cold-start RL, and direct no-RL studies.

## What Changed

- `TD3Agent/reward_functions.py`
  - Extended `make_reward_fn_relative_QR(...)` with `gamma_fallback` and optional `R_fallback_diag`.
  - Uses the existing input-channel `R_diag` by default.
  - Supports component logging through `return_components=True`.

- `Simulation/run_rl_lyapunov.py`
  - Logs `reward_base`, `fallback_penalty`, `reward_augmented`, and weighted correction gap.
  - Adds `offset_free_mpc` as a BC teacher/source.
  - Adds `mpc_only` / `mpc_only_diagnostic` mode: BC can clone normal offset-free MPC, then execution continues without Lyapunov replacement while direct target and Lyapunov checks are recorded as diagnostics.

- `Lyapunov/safety_debug.py`
  - Saves reward component arrays and step-table fields.
  - Adds diagnostic-only unsafe/unstable flags and actual intervention flags.
  - Keeps `mpc_only_diagnostic_bypass` out of safety-active plots because no action is replaced.

- `Lyapunov/direct_lyapunov_mpc.py`
  - Adds `run_offset_free_mpc_with_direct_diagnostics(...)`.
  - Runs normal offset-free MPC while saving direct target-selector and Lyapunov diagnostic fields.
  - Adds diagnostic rates to direct summaries and comparison records.

- Notebooks
  - `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
  - `DirectLyapunovSafetyGateRL_ColdStart.ipynb`
  - `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
  - All now include `mpc_only`.
  - RL notebooks set `gamma_fallback = 0.25` and use normal MPC as the `mpc_only` BC teacher.

## Validation

- Passed:
  - `python -m py_compile TD3Agent/reward_functions.py Simulation/run_rl_lyapunov.py Lyapunov/safety_debug.py Lyapunov/direct_lyapunov_mpc.py`
  - Notebook JSON validation for all three updated notebooks.
  - Reward smoke checks:
    - zero correction gap preserves the old reward
    - accepted/no-gap candidate has zero penalty
    - larger correction gap gives larger penalty with `R_diag`
  - Runner config smoke check for `mpc_only` and `offset_free_mpc`.

## Notes

- The fallback penalty is active only when the reward call marks a safety correction as active.
- `mpc_only` keeps actual intervention at zero and stores diagnostic unsafe/unstable rates separately.
- Debug export remains the default.
