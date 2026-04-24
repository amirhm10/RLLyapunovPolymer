# Direct Eight-Scenario Supervisor Report

## Summary

Rewrote the direct Lyapunov MPC supervisor report around the latest nominal
eight-scenario export:

`Data/debug_exports/direct_lyapunov_mpc_eight_scenario/20260423_231816`

The refreshed report now covers the original four cases plus the four
previous-input regularized bounded cases at `lambda_prev = 0.1` and
`lambda_prev = 1.0`.

## Changes

- Replaced the prior four-scenario supervisor narrative with an eight-scenario
  report.
- Added the target-selector mathematics for:
  - exact unbounded target selection,
  - bounded least-squares projection,
  - bounded previous-input regularized projection,
  - hard and soft first-step Lyapunov constraints.
- Updated the recommendation:
  - primary nominal supervisor: `bounded_soft_u_prev_1p0`,
  - strict no-slack backup: `bounded_hard_u_prev_1p0`.
- Added nominal-mode audit results from the export, including unchanged
  `Qi`, `Qs`, and `hA` values across all cases.
- Updated result tables for reward, solver success, contraction rates, slack,
  output RMSE, target residuals, bounded target activity, and solver/status
  counts.
- Pointed the report to the latest eight-scenario comparison figures and the
  full-run plots for the recommended soft/hard `lambda_prev = 1.0` cases.

## Validation

- Read `comparison_summary.json` and `comparison_table.csv` from the latest
  eight-scenario export.
- Read per-case `summary.json` and `step_table.csv` files for the recommended
  `lambda_prev = 1.0` cases.
- Checked every figure link in the report resolves to an existing file under
  `report/figures/direct_lyapunov_mpc_frozen_output_disturbance/`.
