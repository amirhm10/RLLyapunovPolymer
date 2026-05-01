# 2026-05-01 Direct Three-Method Rho Sensitivity Report

## Why This Change Was Needed

The previous report at:

- `report/direct_lyapunov_bounded_single_setpoint_settling_report_2026-04-30.md`

was written around an older two-case settling story. The latest direct nominal
study changed the scientific question:

- the controller family now has three methods instead of two
- the key finding is no longer just eventual settling
- the important new variable is the Lyapunov contraction factor `rho_lyap`

Without a rewrite, the report would understate the main result and mix an older
single-run interpretation with newer multi-run artifacts.

## What Changed

Rewrote the report into a four-run contraction-factor sensitivity study built on
the latest three-method nominal single-setpoint bundles.

Updated report:

- `report/direct_lyapunov_bounded_single_setpoint_settling_report_2026-04-30.md`

Added a reproducible analysis script:

- `analysis/direct_lyapunov_rho_sensitivity_report.py`

Generated new report assets under:

- `report/figures/2026-05-01_direct_three_method_rho_sensitivity/`

Generated files:

- `rho_run_mapping.csv`
- `rho_sweep_summary.csv`
- `method_metrics_by_rho.svg`
- `bounded_hard_contraction_ratio_by_rho.svg`
- `bounded_hard_u_prev_0p1_contraction_ratio_by_rho.svg`
- `bounded_hard_xs_prev_0p1_contraction_ratio_by_rho.svg`

## Scientific Content Added

The new report now:

- identifies the four unique `rho_lyap` runs directly from the saved step-table
  relation `V_bound / V_k`
- records that `20260501_000956` is a duplicate export of the `rho = 0.98` run
- gives individual result tables for:
  - `bounded_hard`
  - `bounded_hard_u_prev_0p1`
  - `bounded_hard_xs_prev_0p1`
- adds a dedicated Lyapunov contraction plot for each method across the rho
  sweep
- adds a comparison section and a summary section that make the contraction
  factor the main reported finding

## Validation

- `python analysis/direct_lyapunov_rho_sensitivity_report.py`
- `python -m py_compile analysis/direct_lyapunov_rho_sensitivity_report.py`

The generator completed successfully and produced the expected SVG and CSV
artifacts in the report figure directory.
