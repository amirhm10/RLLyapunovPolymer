# Direct Lyapunov Long-Setpoint Settling Analysis

## Summary

Added a standalone report for the latest ten-scenario direct Lyapunov MPC run
with 1500-step setpoint segments:

`Data/debug_exports/direct_lyapunov_mpc_ten_scenario/20260424_105932`

The report answers whether simply holding the setpoint longer makes the
controller settle, using the latest 6000-step export, comparison tables,
per-segment settling diagnostics, copied plots, Lyapunov diagnostics, and
literature-supported interpretation.

## Files

- `report/direct_lyapunov_long_setpoint_settling_analysis.md`
- `report/figures/direct_lyapunov_long_setpoint_settling/`

## Main Finding

The long run partially supports the settling hypothesis: the current best
case, `bounded_soft_u_prev_1p0`, eventually enters the practical settling band
in all four 1500-step segments. However, longer time does not guarantee
settling for every tuning. The `lambda_prev = 10.0` cases expose severe
long-run target residual and solver failure modes that were hidden by the
shorter 400-step run.

## Validation

- Checked that all Markdown image links in the new report resolve.
- Ran `git diff --check` on the new Markdown report path.
