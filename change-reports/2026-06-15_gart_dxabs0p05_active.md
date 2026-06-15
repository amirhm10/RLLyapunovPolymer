# GART Active Target-Rate Bound Set To 0.05

## Summary

The active GART-LMPC raw runner now uses:

$$
\|x_s(k)-x_s(k-1)\|_\infty \le 0.05
$$

instead of the previous `dx_s_max_abs = 0.025`.

The rest of the active setting remains:

- raw GART objective;
- hard Lyapunov contraction;
- `rho = 0.98`;
- `eps = 1.0e-3`;
- `dy_s_max_abs = 1.0`;
- fixed symmetric certified disturbance update with `d_rate_scale = 1.0`;
- no adaptive disturbance certificate;
- no $x_s/y_s$ smoothing;
- no input-midpoint tie-breaker;
- `input_headroom_frac = 0.01`.

## Motivation

The latest disturbed run with `dx_s_max_abs = 0.025` showed that episodes 2 and 3 were poor mainly during the second setpoint of each cycle, especially around the high-to-low transitions:

$$
[4.5, 324]\rightarrow[3.4, 321].
$$

At those switches, $y_s$ lagged far behind the raw setpoint while the certified disturbance was still moving. The target-motion bound was active, so the smaller `0.025` value was likely too slow for the feasible steady-state target to follow the setpoint change under the changing disturbance profile.

Using `dx_s_max_abs = 0.05` keeps a finite moving-target bound for recursive-feasibility/proof bookkeeping, but gives the target selector more room to move during the problematic high-to-low transition.

## Files Changed

- `GARTLyapunovMPC.py`
  - Active root-runner case changed to `gart_target_raw_dxabs0p05_symmetric_dyabs1_no_umid`.
  - Target-only override changed to the same 0.05 symmetric target configuration.
- `experiments/run_gart_target_selector_study.py`
  - Default closed-loop fallback changed to the 0.05 symmetric target configuration.
  - Target-only ablation label/override changed from 0.025 to 0.05.
- `report/gart_lmpc_design_notes.md`
  - Current target-motion bound updated from 0.025 to 0.05.

## Next Validation

Rerun the disturbed 5-episode case and compare against `results/GARTLMPC/20260615_011635`, focusing on:

- episode 2 and 3 reward/RMSE;
- windows `1200-1500` and `2000-2450`;
- $y_s-y_{sp}$ during the high-to-low transition;
- `dx_s_inf` versus `dx_s_max_inf`;
- tail-window input movement.
