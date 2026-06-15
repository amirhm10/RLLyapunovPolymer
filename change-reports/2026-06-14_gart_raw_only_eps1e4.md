# GART Raw-Only Default And Epsilon 1e-4

## Summary

Switched the editable GART-LMPC runner back to the raw objective path and disabled the active mixed case by default. This matches the latest result interpretation: raw GART keeps tracking aligned with the actual setpoint while using the accepted GART target for Lyapunov contraction.

Also tightened the Lyapunov practical-contraction offset from `1e-3` to `1e-4` across the GART target selector, GART MPC config, and experiment runner.

## Controller Defaults

The active closed-loop cases are now:

- `gart_target_raw_no_dx_headroom_0p01_dy2_no_umid`
- `gart_target_raw_no_dx_headroom_0p01_dy4_no_umid`

The mixed cases remain present in `GARTLyapunovMPC.py`, but they are disabled for manual sensitivity runs.

## Mathematical Meaning Of dy2 And dy4

The `dy_rate_scale` setting multiplies the admissible per-step target-output movement bound:

$$
\left| y_s(k) - y_s(k-1) \right| \le d y_{\max}.
$$

The base value of $d y_{\max}$ is discovered from existing case values, using the output-motion quantile when available and a minimum fraction of the output width otherwise. In code:

$$
d y_{\max} = \gamma_y d y_{\max,\mathrm{base}},
$$

where:

- `dy2` means $\gamma_y = 2.0$.
- `dy4` means $\gamma_y = 4.0$.

For the current Polymer setup, the saved smoke config for dy2 had:

$$
d y_{\max}^{\mathrm{dy2}} = [0.992543,\;0.767718].
$$

So dy4 is approximately:

$$
d y_{\max}^{\mathrm{dy4}} \approx [1.985087,\;1.535435].
$$

This is a target-selector governor/rate limit. It does not change the raw MPC tracking objective. It only changes how fast the admissible steady target can move from one accepted target to the next.

## Epsilon Change

The Lyapunov contraction condition is:

$$
V(x_{k+1}-x_s) \le \rho V(x_k-x_s) + \epsilon.
$$

Changed:

$$
\epsilon = 10^{-3} \rightarrow 10^{-4}.
$$

This makes the practical contraction certificate stricter. The raw controller still tracks the raw setpoint objective:

$$
\sum_{i=1}^{N_p}\left\|y_i-y_{sp}\right\|_{Q_{\mathrm{raw}}}^2
+ \sum_{i=0}^{N_c-1}\left\|\Delta u_i\right\|_{R_{\Delta u}}^2,
$$

while the Lyapunov contraction and terminal ingredients remain centered on the accepted GART target.

## Validation

Passed:

```powershell
python -m py_compile GARTLyapunovMPC.py experiments\run_gart_target_selector_study.py Lyapunov\gart_lmpc.py Lyapunov\gart_target.py utils\gart_defaults.py
```

Passed nominal smoke run:

```powershell
python GARTLyapunovMPC.py --closed-loop --no-target-only --mode nominal --n-tests 1 --set-points-len 5 --timestamp codex_smoke_raw_eps1e4 --max-target-evals 200 --max-closed-loop-steps 200 --max-solver-calls 200 --max-wall-clock-seconds 240
```

Smoke result:

| Case | Reward Mean | Output RMSE | Solver | Hard Contract |
|---|---:|---:|---:|---:|
| raw dy2 no-u-mid | -40.799 | 1.285 | 1.000 | 1.000 |
| raw dy4 no-u-mid | -40.799 | 1.285 | 1.000 | 1.000 |

The smoke run confirms the stricter `epsilon=1e-4` configuration is immediately feasible for the short nominal check. A full disturbance run is still needed to compare against the earlier `epsilon=1e-3` long-run metrics.
