# GART State-Target Rate Bound Lowered To 0.025

## Summary

This patch adds `dx_s_max_abs=0.025` GART target-selector cases and makes them the active root runner comparison.

The active cases now use:

$$
\|x_s(k)-x_s(k-1)\|_\infty \le 0.025,\qquad
\|y_s(k)-y_s(k-1)\|_\infty \le 1.0.
$$

## Motivation

The next hypothesis is that some tail-window jumps under `eps=1e-4` may come from movement of the Lyapunov center `x_s` after the plant is already near the setpoint. Lowering `dx_s_max_abs` from `0.05` to `0.025` tests whether a slower moving state target can reduce those jumps without forcing the target selector into the excessive hold behavior observed with `dy_s_max_abs=0.1`.

## Runner Cases

The root runner now enables:

- `gart_target_raw_dxabs0p025_adaptive0p25_min0p10_dyabs1_no_umid`
- `gart_target_raw_dxabs0p025_symmetric_dyabs1_no_umid`

The previous `dx_s_max_abs=0.05` and `dy_s_max_abs=1.0` cases remain available in the experiment module for manual comparison.

## Evaluation Notes

Compare against the previous `dx_s_max_abs=0.05`, `dy_s_max_abs=1.0` run using:

- RMSE and max output error;
- target hold rate and `target_stage_counts`;
- `dx_s_inf_mean`, `dx_s_inf_max`, and `dx_s_max_active_rate`;
- `target_rate_inf_mean` and `target_rate_inf_max`;
- tail-window input jumps and contraction margins.
