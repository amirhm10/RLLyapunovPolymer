# Direct `x_s` Smoothing Extension

## Summary

This change extends the direct frozen-output-disturbance Lyapunov MPC path with
a new bounded-target regularization term that penalizes motion of the steady
target state across time.

The direct notebook default is now a three-scenario nominal single-setpoint
study:

1. `bounded_hard`
2. `bounded_hard_u_prev_0p1`
3. `bounded_hard_xs_prev_0p1`

The controller still tracks the raw setpoint $y_{\mathrm{sp}}$ in the online
MPC objective. The new term only modifies the bounded steady-target solve.

## Technical Changes

- Extended `analysis/steady_state_debug_analysis.py` so
  `solve_bounded_steady_state_least_squares(...)` now supports:
  - `x_ref`
  - `x_ref_weight`
- Added the new bounded-target objective component
  $$
  \lambda_x \|x_s - x_{s,\mathrm{prev}}\|_2^2
  $$
  in both the reduced and full least-squares forms.
- Extended `Lyapunov/frozen_output_disturbance_target.py` with:
  - config key `x_ref_weight`
  - optional `x_ref` argument
  - debug fields for `x_ref`, `x_ref_weight`, activity, penalty, and mismatch
- Extended `Lyapunov/direct_lyapunov_mpc.py` so the rollout:
  - carries the previous successful `x_s`
  - passes it into the target solver as `x_ref`
  - logs and exports `x_s`-smoothing diagnostics
- Updated `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` to the new
  three-scenario focused setup and cleared outputs.
- Extended
  `report/direct_lyapunov_bounded_single_setpoint_settling_report_2026-04-30.md`
  with a method addendum explaining the new objective and the new next-rerun
  configuration.

## Control Interpretation

The new regularization term smooths the Lyapunov center, not the applied input.
That makes it structurally different from the previous-input penalty:

- `u_prev` regularization suppresses movement of the steady input.
- `x_s` smoothing suppresses movement of the internal steady target state.

In the current implementation, the new term is active only when the direct path
uses the bounded least-squares target fallback. It is inactive on the first step
and inactive when the exact bounded steady target is already feasible.

## Validation

- `python -m py_compile` on the touched Python modules
- `Lyapunov/direct_lyapunov_smoke_tests.py`
- `nbformat.validate` on `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`

## Notes

This change does not include a new notebook rerun. The current settling report
still analyzes the earlier saved two-case bundle and now documents the new
three-scenario method extension for the next comparison.
