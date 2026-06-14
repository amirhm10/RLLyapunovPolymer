# GART Zero X/Y Target Smoothing

Date: 2026-06-14

## Summary

Removed target-selector smoothing toward previous steady-state `x_s` and `y_s` from the relaxed GART runner cases by setting those weights to zero through explicit config overrides.

The online target selector now keeps the physically meaningful input smoothing term against the previous applied input, while allowing the steady-state target state and output to move more freely toward the governed reference.

## Code Changes

- Added target smoothing override support in `utils/gart_defaults.py`:
  - direct vector/scalar overrides for `W_u_smooth_diag`, `W_x_smooth_diag`, `W_y_smooth_diag`, and `W_u_mid_diag`
  - disable flags:
    - `disable_u_smoothing`
    - `disable_x_smoothing`
    - `disable_y_smoothing`
    - `disable_u_mid_tiebreak`
- Updated relaxed GART target overrides to include:
  - `disable_x_smoothing = True`
  - `disable_y_smoothing = True`
- Kept `W_u_smooth_diag = [1, 1]`, which now smooths stage-2 target input against the previous applied input in closed-loop runs.
- Added target-ablation rows for no-`x_s/y_s` smoothing variants:
  - `T5_no_dx_rate_headroom_0p01_dy2_no_xy_smooth`
  - `T6_no_dx_rate_headroom_0p01_dy4_no_xy_smooth`
- Updated root runner labels to make the no-`x_s/y_s` smoothing experiment explicit.

## Rationale

Smoothing the target selector toward previous `x_s` and `y_s` can make the certified target artificially sticky, especially under random or aggressive setpoint schedules. The applied input is the actual actuator state, so smoothing `u_s` toward `u_{t-1}` is the more meaningful online regularization.

The relaxed target selector now tests:

$$
\min\; \|W_{\mathrm{mid}}(u_s-u_{\mathrm{mid}})\|_2^2
+ \|W_u(u_s-u_{t-1})\|_2^2
$$

inside the stage-1 primary target shell, without the previous:

$$
\|W_x(x_s-x_{s,\mathrm{prev}})\|_2^2
+ \|W_y(y_s-y_{s,\mathrm{prev}})\|_2^2 .
$$

## Validation

Passed:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe -m py_compile utils/gart_defaults.py experiments/run_gart_target_selector_study.py GARTLyapunovMPC.py Lyapunov/gart_target.py
```

Passed tiny closed-loop smoke:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe GARTLyapunovMPC.py --closed-loop --no-target-only --mode nominal --n-tests 1 --set-points-len 5 --timestamp codex_smoke_zero_xy_smooth --max-target-evals 40 --max-closed-loop-steps 40 --max-solver-calls 40 --max-wall-clock-seconds 180
```

The smoke run completed both relaxed GART cases and did not run old governed-reference or mixed cases.

Saved config verification from `results/GARTLMPC/codex_smoke_zero_xy_smooth/gart_target_raw_no_dx_headroom_0p01_dy2/config.json`:

- `W_x_smooth_diag = [0, 0, 0, 0, 0, 0, 0]`
- `W_y_smooth_diag = [0, 0]`
- `W_u_smooth_diag = [1, 1]`
- `input_headroom_frac = 0.01`
- `dx_s_max = None`

