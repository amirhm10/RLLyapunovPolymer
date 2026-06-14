# GART Zero U-Mid Tie-Break

Date: 2026-06-14

## Summary

Disabled the stage-2 `u_mid` tie-breaker in the relaxed GART runner so the next experiment can test whether midpoint input regularization is still influencing target quality.

## Code Changes

- Added `disable_u_mid_tiebreak = True` to the relaxed GART target override set.
- Renamed the default relaxed GART runner cases to make the new experiment explicit:
  - `gart_target_raw_no_dx_headroom_0p01_dy2_no_umid`
  - `gart_target_raw_no_dx_headroom_0p01_dy4_no_umid`
- Updated the diagnostic probe-log-only case and replay-source case names to include `no_umid`.
- Updated target-ablation labels for the no-`x_s/y_s` smoothing plus no-`u_mid` variants.

## Interpretation

This sets:

$$
W_{\mathrm{mid}} = 0,
$$

so the stage-2 tie-breaker no longer contains:

$$
\|W_{\mathrm{mid}}(u_s-u_{\mathrm{mid}})\|_2^2 .
$$

The input smoothing term remains:

$$
\|W_u(u_s-u_{t-1})\|_2^2 ,
$$

and the closed-loop still uses the raw MPC objective with hard Lyapunov contraction.

## Validation

Passed:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe -m py_compile experiments/run_gart_target_selector_study.py GARTLyapunovMPC.py utils/gart_defaults.py Lyapunov/gart_target.py
```

Passed tiny closed-loop smoke:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe GARTLyapunovMPC.py --closed-loop --no-target-only --mode nominal --n-tests 1 --set-points-len 5 --timestamp codex_smoke_no_umid --max-target-evals 40 --max-closed-loop-steps 40 --max-solver-calls 40 --max-wall-clock-seconds 180
```

Saved config verification from `results/GARTLMPC/codex_smoke_no_umid/gart_target_raw_no_dx_headroom_0p01_dy2_no_umid/config.json`:

- `W_u_mid_diag = [0.0, 0.0]`
- `W_u_smooth_diag = [1.0, 1.0]`
- `W_x_smooth_diag = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
- `W_y_smooth_diag = [0.0, 0.0]`
- `input_headroom_frac = 0.01`
- `dx_s_max = None`

