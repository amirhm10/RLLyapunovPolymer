# GART No-U-Mid Scenario Report Update

Date: 2026-06-14

## Summary

Updated the GART-LMPC scenario analysis report after the full no-`u_mid` runs completed.

Updated report:

- `report/gart_lmpc_post_patch_disturbance_run_2026-06-14.md`

Updated figures and metrics:

- `report/figures/gart_scenario_analysis_2026-06-14/closed_loop_rmse_by_scenario.png`
- `report/figures/gart_scenario_analysis_2026-06-14/target_mismatch_by_scenario.png`
- `report/figures/gart_scenario_analysis_2026-06-14/hold_rate_by_scenario.png`
- `report/figures/gart_scenario_analysis_2026-06-14/scenario_metrics.json`

## New Run Analyzed

Latest full closed-loop run:

- `results/GARTLMPC/20260614_141830`

Cases:

- `gart_target_raw_no_dx_headroom_0p01_dy2_no_umid`
- `gart_target_raw_no_dx_headroom_0p01_dy4_no_umid`

## Main Finding

Removing the `u_mid` tie-breaker did not materially change the closed-loop metrics relative to the previous no-`x_s/y_s` smoothing run:

- solver success remained `1.0`
- hard contraction remained `1.0`
- hold rate remained `0.0`
- acceptable target rate remained about `0.769`
- mean target mismatch remained about `0.561`
- output RMSE remained about `0.370104`

The no-`u_mid` version is still the recommended forward path because it is the cleanest target selector for pretraining and RL exploration: the only remaining stage-2 regularization is smoothing `u_s` toward the actually applied input `u_{t-1}`.

## Updated Recommendation

Move forward with:

```text
gart_target_raw_no_dx_headroom_0p01_dy2_no_umid
```

Keep dy4 as sensitivity only, and keep mixed objective disabled until target-quality logs remain strong under RL exploration.

