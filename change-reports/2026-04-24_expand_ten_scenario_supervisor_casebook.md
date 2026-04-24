# Expand Ten-Scenario Supervisor Casebook

## Summary

Expanded `report/direct_lyapunov_mpc_four_scenario_supervisor_report.md` so it
again reads like a full supervisor report rather than a short recommendation
summary.

## Changes

- Added a dedicated comparison-plot section covering:
  - output overlay,
  - input overlay,
  - reward comparison,
  - output RMSE comparison,
  - solver/contraction comparison,
  - Lyapunov slack comparison,
  - target residual and bounded-target activity.
- Added an individual report section for all ten scenarios:
  - `unbounded_hard`,
  - `bounded_hard`,
  - `unbounded_soft`,
  - `bounded_soft`,
  - `bounded_hard_u_prev`,
  - `bounded_soft_u_prev`,
  - `bounded_hard_u_prev_1p0`,
  - `bounded_soft_u_prev_1p0`,
  - `bounded_hard_u_prev_10p0`,
  - `bounded_soft_u_prev_10p0`.
- For each scenario, added a compact metric table plus output, input,
  Lyapunov-diagnostics, and target-diagnostics figures.
- Added the newly referenced per-case `u_prev` diagnostic figures to the
  report figure set so the markdown is self-contained.

## Validation

- Checked every image link in the supervisor report resolves to an existing
  file.
- Reused metrics from the latest ten-scenario `comparison_table.csv`.
- Ran `git diff --check` on the rewritten supervisor report.
