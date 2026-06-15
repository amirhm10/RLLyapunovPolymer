# Try GART Asymmetric Certified Disturbance Rate Scale

## Summary

Changed the selected GART disturbance-rate trial from a global `d_rate_scale = 0.5` to a channel-specific scale:

$$
d_{\text{rate scale}} = [1.0,\;0.5].
$$

The active runner case is now:

`gart_target_raw_dxabs0p05_drate1_0p5_headroom_0p01_dy2_no_umid`

## Selected Settings

- raw MPC objective
- hard Lyapunov contraction
- `eps = 1e-4`
- absolute `dx_s_max = 0.05`
- `d_rate_scale = [1.0, 0.5]`
- `dy_rate_scale = 2`
- `input_headroom_frac = 0.01`
- no `u_mid` tie-breaker
- no $x_s$ smoothing
- no $y_s$ smoothing

## Rationale

The global `d_rate_scale = 0.5` run reduced the largest certified-disturbance and target-output jumps, but worsened average tracking and target mismatch. The degradation was stronger in the first output channel, suggesting that scaling both disturbance channels was too conservative.

This asymmetric trial keeps the first disturbance channel adaptive while still damping the second disturbance channel, where the largest jumps were observed.

Expected certified disturbance rate limits:

$$
\Delta d^c_{\max} = [0.15564781,\;0.34462059].
$$

## Compare Against

- `results/GARTLMPC/20260614_213132`: `dx_s_max = 0.05`, original disturbance rate
- `results/GARTLMPC/20260614_214921`: `dx_s_max = 0.05`, global `d_rate_scale = 0.5`

Key metrics:

- reward mean and output RMSE
- mean and max output error
- target mismatch by channel
- max $\Delta d^c$
- max $\Delta y_s$
- governor active rate
- hard contraction rate
