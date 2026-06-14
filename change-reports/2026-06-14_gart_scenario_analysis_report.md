# GART Scenario Analysis Report

Date: 2026-06-14

## Summary

Rewrote the GART-LMPC analysis report around the full set of completed scenarios instead of a single post-patch disturbance run.

Updated report:

- `report/gart_lmpc_post_patch_disturbance_run_2026-06-14.md`

Generated supporting figures under:

- `report/figures/gart_scenario_analysis_2026-06-14/`

## Case Studies Covered

- Mixed hard/soft objective failure in long nominal and disturbed runs.
- Conservative raw GART target selector after the correctness patch.
- Relaxed raw GART with `dx_s` rate disabled, 1% input headroom, and previous-applied-input smoothing.
- Relaxed raw GART with `x_s/y_s` smoothing removed.

## Main Conclusion

The method to carry forward is:

```text
gart_target_raw_no_dx_headroom_0p01_dy2
```

with:

- `input_headroom_frac = 0.01`
- `dx_s_max = None`
- `dy_rate_scale = 2.0`
- `W_u_smooth_diag = [1.0, 1.0]`
- `W_x_smooth_diag = 0`
- `W_y_smooth_diag = 0`
- stage-2 input smoothing against previous applied input `u_{t-1}`
- raw tracking objective, `eta_y = eta_u = 0`
- hard Lyapunov contraction

Mixed objective remains rejected for the next phase because it tracks the certified target as a performance target, which degraded tracking when `y_s` was imperfect or stale.

## Validation

This was a report/data analysis update. Metrics were read from:

- `results/GARTLMPC/20260613_235051/summary.json`
- `results/GARTLMPC/20260614_003718/summary.json`
- `results/GARTLMPC/20260614_124954/summary.json`
- `results/GARTLMPC/20260614_133147/summary.json`
- `results/GARTLMPC/20260614_134444/summary.json`
- corresponding `steps.csv` and `config.json` files for target-quality and configuration checks

