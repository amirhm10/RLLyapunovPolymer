# GART Absolute Output-Target Rate Comparison

## Summary

This patch adds a direct absolute output target-motion bound for the GART target selector and changes the active root runner comparison to:

- adaptive certified disturbance projection;
- previous fixed symmetric certified disturbance projection.

Both active cases use the same raw GART MPC objective and the same target selector shape:

$$
\|y_s(k)-y_s(k-1)\|_\infty \le 0.1,\qquad
\|x_s(k)-x_s(k-1)\|_\infty \le 0.05.
$$

## Motivation

The previous disturbed runs showed that the adaptive certified-disturbance projection could reduce some disturbance-induced motion but did not fully remove occasional setpoint-near jumps. The next clean experiment is to control the proof-relevant output target motion directly with `dy_s_max_abs=0.1`, then compare adaptive projection against the older fixed symmetric disturbance certificate.

The asymmetric fixed-rate case `[1.0, 0.5]` is still available for manual reference, but it is no longer part of the active comparison because the requested experiment is symmetric.

## Code Changes

- Added `dy_s_max_abs` / `dy_s_max_override` support in `make_gart_target_config(...)`.
- Added shared override cases:
  - `GART_DX_ABS_0P05_SYMMETRIC_DYABS0P1_OVERRIDES`
  - `GART_DX_ABS_0P05_ADAPTIVE_DYABS0P1_OVERRIDES`
- Updated the root runner to enable both `dyabs0p1` cases by default.
- Updated the experiment-script fallback case to use the adaptive `dyabs0p1` configuration.
- Added a unit test covering scalar and vector `dy_s_max_abs` overrides.
- Updated the design notes with the current output target-motion constraint.

## Expected Experiment

Run the root runner in the current disturbance closed-loop configuration. It will generate two active closed-loop result folders:

- `gart_target_raw_dxabs0p05_adaptive0p25_min0p10_dyabs0p1_no_umid`
- `gart_target_raw_dxabs0p05_symmetric_dyabs0p1_no_umid`

The important comparison metrics are RMSE, max tracking error, reward, target hold rate, `dx_s_inf`, `dc_rate_inf`, `d_raw_gap_inf`, adaptive scale statistics, and the new effective target-motion behavior through `y_s-y_sp` and `dy_s`.
