# GART dx-rate Scenario Matrix

## Summary

Restored the main GART practical-contraction offset to `eps=1e-3` and reintroduced finite `dx_s_max` target-state rate bounds through three raw dy2 scenarios:

- `gart_target_raw_dx5_headroom_0p01_dy2_no_umid`
- `gart_target_raw_dx10_headroom_0p01_dy2_no_umid`
- `gart_target_raw_dx20_headroom_0p01_dy2_no_umid`

The previous no-`dx_s` raw case remains in the root runner as a disabled manual comparison.

## Motivation

The `eps=1e-4` run remained feasible but degraded tracking and target quality:

- reward mean worsened from about `-4.158` to `-4.425`
- output RMSE increased from about `0.370` to `0.382`
- good target rate dropped from about `45.8%` to `33.3%`

More importantly, post-run analysis showed large target-state jumps even when the raw setpoint was constant. The moving-target proof concern is:

$$
x_{k+1}-x_s(k+1)
= [x_{k+1}-x_s(k)] - [x_s(k+1)-x_s(k)].
$$

Therefore a finite bound on $\Delta x_s$ is useful for recursive-feasibility and moving-target practical-stability arguments.

## Implementation

The base target configuration still uses:

- no `x_s/y_s` smoothing
- no `u_mid` tie-breaker
- `input_headroom_frac = 0.01`
- `dy_rate_scale = 2.0`
- raw MPC objective

The three active cases differ only in:

```python
dx_rate_scale = 5.0
dx_rate_scale = 10.0
dx_rate_scale = 20.0
```

The runner resource guards were increased back to three-case full-run values:

- `MAX_TARGET_EVALS = 15000`
- `MAX_CLOSED_LOOP_STEPS = 15000`
- `MAX_SOLVER_CALLS = 15000`
- `MAX_WALL_CLOCK_SECONDS = 21600.0`

## Added Diagnostics

Closed-loop step logs and direct-style bundles now include:

- `dx_s_inf`: $\lVert x_s(k)-x_s(k-1)\rVert_\infty$
- `dc_rate_inf`: $\lVert d_c(k)-d_c(k-1)\rVert_\infty$
- `dx_s_max_active`: whether a finite `dx_s_max` was configured
- `dx_s_max_inf`: $\lVert dx_{\max}\rVert_\infty$

Direct-style summaries and comparison records now include mean/max `dx_s_inf`, mean/max `dc_rate_inf`, and `dx_s_max_active_rate`.

## Validation

Passed:

```powershell
python -m py_compile GARTLyapunovMPC.py experiments\run_gart_target_selector_study.py Lyapunov\gart_lmpc.py Lyapunov\gart_target.py Lyapunov\direct_lyapunov_mpc.py utils\gart_defaults.py
```

Passed smoke run:

```powershell
python GARTLyapunovMPC.py --closed-loop --no-target-only --mode nominal --n-tests 1 --set-points-len 5 --timestamp codex_smoke_dx_scales --max-target-evals 300 --max-closed-loop-steps 300 --max-solver-calls 300 --max-wall-clock-seconds 240
```

Smoke output confirmed all three active cases ran and produced the new diagnostics:

| Case | Solver | Hard Contract | `dx_s_max_active` | Output RMSE |
|---|---:|---:|---:|---:|
| dx5 | 1.000 | 1.000 | 1.000 | 1.268 |
| dx10 | 1.000 | 1.000 | 1.000 | 1.268 |
| dx20 | 1.000 | 1.000 | 1.000 | 1.268 |

The smoke run also showed that the absolute `dx_s_max` values can be loose in the short nominal case. The full disturbance run should therefore be interpreted using both performance metrics and the new `dx_s_inf` / `dx_s_max_inf` diagnostics.
