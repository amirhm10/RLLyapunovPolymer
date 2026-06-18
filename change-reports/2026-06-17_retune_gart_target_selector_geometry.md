# Retune GART Target Selector Geometry

Date: 2026-06-17

## Objective

Reduce the tendency of the GART steady target input `u_s` to sit on the tightened input bounds by changing the target-selector geometry, while keeping the GART-LMPC objective weights unchanged.

## Change

The GART-LMPC runner and exploration-probe runner now use:

```python
INPUT_HEADROOM_FRAC = 0.03
PRIMARY_TOL_REL = 1.0e-4
W_U_SMOOTH_DIAG = [2.0, 2.0]
TARGET_WY_DIAG = [1.0, 1.0]
```

These values are passed only through the GART target-selector overrides.

## Method Interpretation

The target selector solves a steady-state target projection problem:

$$
\min_{x_s,u_s} \|W_y(y_s-r)\|^2
$$

subject to steady-state equality, input bounds, rate limits, and the GART contraction probe. Setting `TARGET_WY_DIAG = [1, 1]` changes the target projection metric so that the first output is not weighted five times more heavily than the second output.

The MPC objective remains separate:

$$
\sum_i \|y_i-y_{\mathrm{sp}}\|_{Q_{\mathrm{raw}}}^2 + \|u_i-u_{i-1}\|_{R_{\Delta u}}^2
$$

with the existing MPC weights unchanged. This separation is intentional because target selection is a steady-state feasibility/projection problem, while the MPC objective is the dynamic tracking problem.

## Why This Is Proof-Friendly

The change does not relax the first-step Lyapunov contraction constraint. It changes which governed steady target is selected before the contraction check. The selected target must still pass the same GART feasibility and contraction-probe logic.

## Expected Diagnostic Signal

The next run should be checked for:

- lower rate of `u_s` within `1e-3` of the tightened input bounds
- smaller `|u_s - u_prev_applied|_inf`
- no loss of `target_accepted_rate`
- no loss of `hard_contraction_rate`
- improved or unchanged `output_rmse_to_ys`
- no increase in large applied `delta_u` events

## Risk

Changing `Wy` to `[1, 1]` may trade some first-output target accuracy for a less extreme steady target. If output tracking worsens while `u_s` moves away from the bounds, then the projection metric is too weak on the first output or the raw setpoint remains outside the reachable target set under the certified disturbance.
