# Try GART dx_s Bound At Absolute 0.05

## Summary

Changed the selected GART raw controller from absolute `dx_s_max = 0.1` to absolute `dx_s_max = 0.05` for the next disturbance run.

The selected target-state rate constraint is now:

$$
|x_{s,i}(k)-x_{s,i}(k-1)| \le 0.05,\quad i=1,\ldots,7.
$$

## Rationale

The `dx_s_max = 0.1` run improved tracking relative to the loose dx5 run with `eps = 1e-4`, and the bound was active. The remaining target/input jumps motivate testing a tighter target-state motion bound.

## Selected Runner Case

The active root-runner case is now:

`gart_target_raw_dxabs0p05_headroom_0p01_dy2_no_umid`

Other selected settings remain unchanged:

- raw MPC objective
- hard Lyapunov contraction
- `eps = 1e-4`
- `dy_rate_scale = 2`
- `input_headroom_frac = 0.01`
- no `u_mid` tie-breaker
- no $x_s$ smoothing
- no $y_s$ smoothing

## What To Compare After The Run

Compare against `results/GARTLMPC/20260614_211609`:

- reward mean
- output RMSE
- target mismatch
- governor active rate
- hard contraction rate
- max $\Delta u_s$
- `dx_s_inf / dx_s_max`

If `0.05` improves max $\Delta u_s$ without increasing target lag or governor activity, it is a better RL-exploration candidate. If tracking worsens or governor activity rises, keep `0.1` as the main setting.
