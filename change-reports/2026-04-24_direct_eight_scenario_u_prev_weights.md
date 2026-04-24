# Direct Eight-Scenario U-Prev Weight Study

## Summary

Expanded `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` from the six-scenario
direct Lyapunov study to an eight-scenario study. The bounded previous-input
target regularization now runs at two strengths:

- `lambda_prev = 0.1`
- `lambda_prev = 1.0`

## Changes

- Kept the original four baseline cases unchanged:
  - `unbounded_hard`
  - `bounded_hard`
  - `unbounded_soft`
  - `bounded_soft`
- Kept the existing previous-input regularized cases at `lambda_prev = 0.1`:
  - `bounded_hard_u_prev`
  - `bounded_soft_u_prev`
- Added stronger previous-input regularized cases at `lambda_prev = 1.0`:
  - `bounded_hard_u_prev_1p0`
  - `bounded_soft_u_prev_1p0`
- Added a visible notebook switch:
  - `target_u_prev_regularization_weight_strong = 1.0`
- Updated the study name and export root to
  `direct_lyapunov_mpc_eight_scenario`.

## Validation

- Ran a notebook JSON/AST check confirming:
  - exactly eight scenarios,
  - the original four scenarios have `u_ref_weight = 0.0`,
  - the two existing `u_prev` scenarios have `u_ref_weight = 0.1`,
  - the two new `u_prev_1p0` scenarios have `u_ref_weight = 1.0`,
  - the study name is `direct_lyapunov_mpc_eight_scenario`.
