# Direct Target Quality Gate And RL Guard

Date: 2026-05-18

## Summary

Implemented the direct target-quality and RL guard plan for the latest direct notebook diagnosis. The change keeps notebook-facing function names and return tuple shapes stable while adding opt-in controls for poor-target bypass, lexicographic bounded targets, disturbance-model selection, RL maintenance rewards, a performance guard, and residual RL.

## Code Changes

- Added `target_quality` handling in `Lyapunov/direct_lyapunov_mpc.py`.
- Added `solve_strategy="lexicographic"` support for bounded steady-state targets in `analysis/steady_state_debug_analysis.py` and exposed it through `Lyapunov/frozen_output_disturbance_target.py`.
- Added `disturbance_model_mode` with the existing output-disturbance path as default and generic augmented selector routing for `state_via_B` and `mixed`.
- Added optional RL maintenance reward terms in `TD3Agent/reward_functions.py`.
- Added optional direct-gate performance guard and residual-RL controls in `Simulation/run_rl_lyapunov.py`.
- Extended direct and RL debug exports with target-quality, rate, performance-guard, and residual-RL fields.

## Diagnosis Preserved

The latest saved runs show that `mpc_only` wins because it keeps tracking the raw setpoint. The Lyapunov gate can instead enforce contraction around a poor target under the disturbed plant.

Key numbers:

- No-RL: Lyap RMSE mean 0.436 vs `mpc_only` 0.357; Lyap reward -5.70 vs `mpc_only` -3.88.
- No-RL tail: `mpc_only` final physical error `[0.004, -0.020]`; Lyap final physical error `[0.125, -0.598]`.
- Cold-start RL: safe-gate RMSE 0.265 vs `mpc_only` 0.239; reward -3.209 vs -2.225.
- Pretrained RL: safe-gate RMSE 0.255 vs `mpc_only` 0.245; reward -3.036 vs -2.249.

## Validation

- `python -m py_compile` passed for touched modules using a temporary bytecode directory because the normal OneDrive pycache path denied writes.
- A synthetic target-selector test confirmed lexicographic priority:

| Strategy | u target | output residual |
|---|---:|---:|
| legacy_ls | 0.009901 | 0.990099 |
| lexicographic | 0.200000 | 0.800000 |

## Follow-Up

Rerun the no-RL direct comparison with `solve_strategy="lexicographic"` and `target_quality.enabled=True`, then rerun a short RL smoke test with the performance guard enabled before committing to another full training run.
