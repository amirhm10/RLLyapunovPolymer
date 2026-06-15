# GART Fixed Symmetric Certificate With Relaxed Contraction

## Summary

This change makes the fixed symmetric certified-disturbance path the active GART-LMPC runner configuration and restores the main contraction setting to:

$$
\rho = 0.98,\qquad \epsilon = 10^{-3}.
$$

The adaptive certified-disturbance projection remains in the implementation for reproducibility of earlier diagnostic runs, but it is no longer used by the root runner or the default closed-loop fallback case.

## Motivation

The latest disturbed runs showed that the adaptive disturbance certificate could still produce mixed behavior. In particular, it sometimes lagged the observer correction and then allowed late target/input movement near the end of the run. The fixed symmetric certificate was more predictable and avoided the large adaptive tail jump in the recent comparisons.

The recent $\epsilon=10^{-4}$ runs also made the contraction test too sensitive near the setpoint. When $V$ is already small, very small target, observer, or numerical changes can dominate the contraction residual. Returning to $\epsilon=10^{-3}$ gives the controller more practical-stability room while keeping the contraction certificate explicit. Lowering $\rho$ to `0.98` keeps the contraction target meaningful without making the near-setpoint certificate as brittle as the aggressive tolerance trial.

## Active Runner Path

The root runner now enables only:

- `gart_target_raw_dxabs0p025_symmetric_dyabs1_no_umid`

with:

- raw GART MPC objective;
- hard Lyapunov contraction;
- `dx_s_max_abs = 0.025`;
- `dy_s_max_abs = 1.0`;
- `d_rate_scale = 1.0`;
- `input_headroom_frac = 0.01`;
- no $x_s/y_s$ smoothing;
- no input-midpoint tie-breaker.

The old governed-reference, mixed-objective, no-`dx_s`, and asymmetric disturbance-rate variants remain disabled manual comparison cases.

## Files Changed

- `GARTLyapunovMPC.py`
  - Removed the adaptive case from the active root-runner case list.
  - Switched target-only overrides to the fixed symmetric certificate.
- `experiments/run_gart_target_selector_study.py`
  - Set `RHO_LYAP = 0.98`.
  - Set `LYAP_EPS = 1.0e-3`.
  - Removed the adaptive `dx_s=0.025` target ablation from the default matrix.
  - Switched the default closed-loop case to fixed symmetric disturbance certification.
- `utils/gart_defaults.py`
  - Updated shared GART defaults to `rho = 0.98` and `eps = 1.0e-3`.
- `report/gart_lmpc_design_notes.md`
  - Updated the method description so the current proof path is the fixed symmetric certificate, with adaptive projection documented as disabled after the latest comparisons.

## Validation

Run:

```powershell
C:\Users\hamed\miniconda3\envs\rlenv\python.exe -m py_compile Lyapunov/gart_target.py utils/gart_defaults.py experiments/run_gart_target_selector_study.py GARTLyapunovMPC.py
```

Then rerun the disturbed GART case and compare it to the latest adaptive and symmetric result folders on:

- RMSE and reward;
- tail-window input movement;
- $y-y_{sp}$ and $y_s-y_{sp}$;
- `d_hat-d_c` gap;
- contraction margin distribution.
