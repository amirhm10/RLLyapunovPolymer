# Direct Bounded Single-Setpoint Settling Report

## Summary

Added a new direct Lyapunov analysis report for the focused nominal single-setpoint run:

- `report/direct_lyapunov_bounded_single_setpoint_settling_report_2026-04-30.md`

The report compares:

- `bounded_hard`
- `bounded_hard_u_prev_1p0`

with emphasis on the late-settling behavior observed after roughly step 1000.

## Main Additions

- Reconstructed the direct target-selection and tracking interpretation for the
  frozen output-disturbance formulation.
- Quantified full-episode and windowed performance for both cases.
- Computed sustained-settling indices using practical and tight physical-unit
  bands.
- Explained why `bounded_hard` settles earlier while `bounded_hard_u_prev_1p0`
  still wins on aggregate reward and RMSE.
- Connected late settling to the transition from bounded least-squares targets
  to sustained exact bounded targets.

## Figures Added

Created a new figure set under:

- `report/figures/2026-04-30_direct_bounded_single_setpoint_settling/`

New generated figures:

- `late_settling_zoom_outputs.png`
- `settling_stage_timeline.png`

Copied comparison figures from the run bundle:

- `comparison_outputs_overlay.png`
- `comparison_output_rmse.png`
- `comparison_reward_mean.png`
- `comparison_target_residual_bounded_activity.png`

## Data Basis

The report is based on:

- `Data/debug_exports/direct_lyapunov_mpc_bounded_hard_single_setpoint_nominal/20260430_232523`

## Validation

- Verified the report text locally.
- Generated figures using the user's `rl-env` Python environment.
