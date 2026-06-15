# GART Output-Target Rate Bound Raised To 1.0

## Summary

This patch keeps the previous `dy_s_max_abs=0.1` cases available for reference and adds new active `dy_s_max_abs=1.0` cases for the next disturbed closed-loop run.

The active root runner comparison is now:

- adaptive certified disturbance projection with `dy_s_max_abs=1.0`;
- fixed symmetric certified disturbance projection with `dy_s_max_abs=1.0`.

Both use:

$$
\|x_s(k)-x_s(k-1)\|_\infty \le 0.05,\qquad
\|y_s(k)-y_s(k-1)\|_\infty \le 1.0.
$$

## Motivation

The `dy_s_max_abs=0.1` disturbed run made the target selector too sticky. In the latest full run, the adaptive case held the previous target for most steps and the symmetric fixed case almost never moved the target. That produced worse tracking than the previous `dy_rate_scale=2.0` family.

This test raises the direct output target-motion bound to `1.0` to check whether the fixed absolute bound can regain target mobility without fully returning to the looser quantile-scaled `dy2` behavior.

## Runner Cases

The root runner now enables:

- `gart_target_raw_dxabs0p05_adaptive0p25_min0p10_dyabs1_no_umid`
- `gart_target_raw_dxabs0p05_symmetric_dyabs1_no_umid`

The older `dyabs0p1` constants remain in the experiment module for manual comparison.
