# Direct Ten-Scenario U-Prev Weight Study

## Summary

Expanded `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` from the
eight-scenario direct Lyapunov study to a ten-scenario study by adding a third
previous-input target regularization strength:

- `lambda_prev = 10.0`

## Changes

- Kept the original four baseline cases unchanged:
  - `unbounded_hard`
  - `bounded_hard`
  - `unbounded_soft`
  - `bounded_soft`
- Kept previous-input regularized bounded cases at:
  - `lambda_prev = 0.1`
  - `lambda_prev = 1.0`
- Added stronger previous-input regularized cases at `lambda_prev = 10.0`:
  - `bounded_hard_u_prev_10p0`
  - `bounded_soft_u_prev_10p0`
- Added a visible notebook switch:
  - `target_u_prev_regularization_weight_very_strong = 10.0`
- Updated the study name and export root to
  `direct_lyapunov_mpc_ten_scenario`.

## Validation

- Ran a notebook JSON/AST check confirming:
  - exactly ten scenarios,
  - the original four scenarios have `u_ref_weight = 0.0`,
  - the `u_prev` scenarios have weights `0.1`, `1.0`, and `10.0`,
  - the study name is `direct_lyapunov_mpc_ten_scenario`.
