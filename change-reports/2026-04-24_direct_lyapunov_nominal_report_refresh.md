# Direct Lyapunov Nominal Report Refresh

## Summary

Rewrote the direct Lyapunov four-scenario supervisor report around the latest
nominal-mode export:

`Data/debug_exports/direct_lyapunov_mpc_four_scenario/20260423_221822`

The refreshed report now uses the nominal run metrics and references the
proper-scale report figures copied from that run.

## Changes

- Replaced the previous supervisor-report narrative with a nominal-mode report.
- Updated the headline conclusion:
  - `bounded_hard` is the best strict Lyapunov baseline by nominal reward.
  - `bounded_soft` is the robust fallback baseline with rare, small slack.
  - unbounded modes remain diagnostic controls because their steady targets are
    input-inadmissible.
- Updated all main result tables for reward, physical output RMSE, solver
  success, contraction rates, slack, target admissibility, and target gaps.
- Pointed case-study figures to the refreshed full-horizon nominal output/input
  plots and refreshed Lyapunov/target diagnostics.
- Updated `lyapunov_delta_summary.csv` to match the new nominal step tables.

## Validation

- Read the latest `comparison_table.csv` and per-case `summary.json` files.
- Recomputed target-gap and Lyapunov-delta summary values from the latest
  per-case `step_table.csv` files.
- Checked the report links against the report figure directory.
