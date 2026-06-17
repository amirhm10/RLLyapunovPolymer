# Slow GART-LMPC Disturbance Target

Date: 2026-06-17

## Objective

Test whether the large late-run GART-LMPC input jumps are driven more by fast certified-disturbance target motion than by steady-input target motion alone.

## Change

`GARTLyapunovMPC.py` now uses the following local GART-LMPC target-governor overrides:

```python
DU_S_MAX_ABS = [0.2, 0.2]
DY_S_MAX_ABS = 0.25
D_RATE_SCALE = 0.25
ALPHA_D = 0.05
```

The previous `DU_S_MAX_ABS = [0.05, 0.05]` setting was relaxed to avoid excessive lag in the governed steady input. The disturbance-rate scale and disturbance adaptation gain were reduced to slow the certified disturbance target.

## Method Interpretation

This experiment keeps the governed target motion bounded while separating two mechanisms:

$$
\|u_{s,k} - u_{s,k-1}\|_\infty \le 0.2
$$

and the certified disturbance update is slowed by both a smaller admissible rate and a smaller adaptation gain.

## Expected Diagnostic Signal

The next GART-LMPC run should be checked for:

- lower `dc_rate_inf_max`
- lower `disturbance_target_error_inf_mean`
- fewer `hold_previous_current_disturbance_rejected` events
- lower maximum applied `|Delta u|`
- no large increase in output tracking error from target lag

## Risk

If the disturbance target is slowed too much, the controller may reject valid disturbance correction and accumulate steady target mismatch. If that happens, the next adjustment should be to relax `D_RATE_SCALE` upward before changing the Lyapunov contraction parameters.
