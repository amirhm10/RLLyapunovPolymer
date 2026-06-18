# Scale GART RL Observation Inputs

## Objective

Fix the GART RL observation used by online TD3 so the added input-related features are numerically consistent with the standard RL state scaling.

## Change

The GART observation now contains:

```text
x_aug, d_cert, y_sp, u_prev, r_cmd, y_s, u_s
```

with no raw contraction-margin feature.

The manipulated-input blocks now use the same min-max-to-`[-1, 1]` scaling as the standard TD3 observation:

```python
scale_pm1(u_prev, u_min, u_max)
scale_pm1(u_s, u_min, u_max)
```

The output-reference blocks continue to use the RL setpoint scaler:

```python
scale_pm1(y_sp, y_sp_min, y_sp_max)
scale_pm1(r_cmd, y_sp_min, y_sp_max)
scale_pm1(y_s, y_sp_min, y_sp_max)
```

## Dimension Update

Removing the raw margin changes the GART observation dimension from:

```text
n_aug + 4*n_y + 2*n_u + 1
```

to:

```text
n_aug + 4*n_y + 2*n_u
```

For the current polymer case, this changes the GART TD3 state dimension from `22` to `21`.

## Reason

The latest cold-start safety run showed that `u_prev` and `u_s` were entering the actor in scaled-deviation coordinates with ranges near `[-10, 10]`, while the standard TD3 state scales the same input quantity to approximately `[-1, 1]`. The raw contraction margin was also in unbounded Lyapunov units, so it was removed for now.
