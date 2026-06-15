# Try GART Certified Disturbance Rate Scale 0.5

## Summary

Added a `d_rate_scale` override to the GART target configuration and selected the next raw GART trial:

`gart_target_raw_dxabs0p05_drate0p5_headroom_0p01_dy2_no_umid`

## Selected Settings

- raw MPC objective
- hard Lyapunov contraction
- `eps = 1e-4`
- absolute `dx_s_max = 0.05`
- `d_rate_scale = 0.5`
- `dy_rate_scale = 2`
- `input_headroom_frac = 0.01`
- no `u_mid` tie-breaker
- no $x_s$ smoothing
- no $y_s$ smoothing

## Rationale

The `dx_s_max = 0.05` run reduced target-state and target-input jumps, but the remaining spikes appeared to be driven by certified disturbance motion:

$$
y_s = Cx_s + d^c.
$$

Scaling the certified disturbance rate limit by `0.5` tests whether smoother $d^c$ motion reduces the remaining $y_s$ and output-error spikes without creating excessive target lag.

## Expected Diagnostic Comparison

Compare this run against:

`results/GARTLMPC/20260614_213132`

Key metrics:

- reward mean
- output RMSE
- max output error
- target mismatch
- max $\Delta d^c$
- max $\Delta y_s$
- max $\Delta u_s$
- governor active rate
- hard contraction rate
