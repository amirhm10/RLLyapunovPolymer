# 2026-05-01 Direct Latest Run Numerics Report

## Why This Change Was Made

The user requested a focused analysis of the latest direct Lyapunov two-setpoint run after observing that:

- the plain bounded direct method still oscillates
- the input-anchor and `x_s`-smoothing variants behave much better
- the remaining question is whether the bad plain-case behavior is caused mainly by:
  - target-selection drift
  - ill conditioning
  - very small Lyapunov values near the setpoint
  - or a numerical contraction-check issue

The goal of this change was to answer that question quantitatively and write a reusable report rather than relying only on visual inspection of the saved run plots.

## What Was Added

Added analysis script:

- `analysis/direct_lyapunov_latest_run_numerics_report.py`

This script:

- finds the latest run under `Data/debug_exports/direct_lyapunov_mpc_bounded_three_scenario_two_setpoint_nominal/`
- reads the saved CSV and JSON artifacts using only the Python standard library
- computes per-case target-movement, failure-cluster, conditioning, and small-`V` diagnostics
- generates new SVG figures
- writes a markdown report summarizing the diagnosis

Added report:

- `report/direct_lyapunov_latest_two_setpoint_numerical_analysis_2026-05-01.md`

Added generated artifacts:

- `report/figures/2026-05-01_direct_latest_run_numerics/target_step_changes_by_case.svg`
- `report/figures/2026-05-01_direct_latest_run_numerics/bounded_hard_target_drift_cluster_748_798.svg`
- `report/figures/2026-05-01_direct_latest_run_numerics/bounded_hard_small_v_tail.svg`
- `report/figures/2026-05-01_direct_latest_run_numerics/case_diagnostics.csv`

## Main Technical Finding

The generated report concludes that the latest saved bad `bounded_hard` behavior is dominated by target-selection drift inside the bounded least-squares target stage, not by raw target-matrix conditioning and not by the tiny-`V` Lyapunov regime.

The report also states explicitly that:

- increasing `rho_lyap` from `0.98` to `0.995` makes the Lyapunov bound looser, not tighter
- the smallest saved `V_k` values in this run are still well above the `eps_lyap / (1 - rho)` scale where the additive Lyapunov floor would dominate
- all saved failures in the plain case occur during the bounded least-squares target stage
- the anchor variants improve behavior by stabilizing the selected steady target, especially `u_s`

## Validation

- Ran `python analysis/direct_lyapunov_latest_run_numerics_report.py`
- Confirmed the markdown report and SVG/CSV artifacts were generated successfully
- Ran an in-memory `compile(...)` check for `analysis/direct_lyapunov_latest_run_numerics_report.py`

## Notes

- `python -m py_compile analysis/direct_lyapunov_latest_run_numerics_report.py` hit the existing Windows `__pycache__` rename permission problem in this repository, so bytecode-write validation was replaced with an in-memory syntax check
- This change analyzes the latest saved run only; it does not rerun the notebook
