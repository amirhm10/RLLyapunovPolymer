# Direct Ten-Scenario Supervisor Report

## Summary

Rewrote the direct Lyapunov MPC supervisor report around the latest nominal
ten-scenario export:

`Data/debug_exports/direct_lyapunov_mpc_ten_scenario/20260423_234338`

The refreshed report now covers the original four cases plus all six
previous-input regularized bounded cases at `lambda_prev = 0.1`, `1.0`, and
`10.0`.

## Changes

- Replaced the previous supervisor narrative with a ten-scenario report.
- Added the `lambda_prev = 10.0` hard and soft bounded cases to the study
  matrix, performance table, target-selector table, solver table, and
  recommendation.
- Updated the main recommendation:
  - primary nominal supervisor: `bounded_soft_u_prev_1p0`;
  - strongest soft comparison: `bounded_soft_u_prev_10p0`;
  - hard no-slack tracking candidate: `bounded_hard_u_prev_10p0`;
  - hard no-slack reliability candidate: `bounded_hard_u_prev_1p0`.
- Added the interpretation that stronger previous-input regularization is not
  monotonically better: `lambda_prev = 10.0` is competitive, but the best soft
  nominal result remains `lambda_prev = 1.0`.
- Pointed the report to the latest ten-scenario comparison figures and the
  full-run plots for the recommended `1.0` soft case and the new `10.0`
  soft/hard comparisons.

## Validation

- Read `comparison_summary.json` and `comparison_table.csv` from the latest
  ten-scenario export.
- Checked that the export contains exactly ten cases.
- Checked that every figure link in
  `report/direct_lyapunov_mpc_four_scenario_supervisor_report.md` resolves to
  an existing file under
  `report/figures/direct_lyapunov_mpc_frozen_output_disturbance/`.
- Checked the report for stale wording from the earlier scenario counts.
