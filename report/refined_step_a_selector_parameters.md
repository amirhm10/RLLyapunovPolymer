# Refined Step A Selector Parameters

## Overview
The refined Step A selector solves one steady-state target problem in the repository's scaled deviation coordinates. The decision variables are `x_s` and `u_s`, with `r_s = H y_s` when a controlled-output map `H` is provided and `r_s = y_s` otherwise. The disturbance target is fixed:

`d_s = d_hat_k`

The selector objective is

`||r_s - y_sp||^2_{Q_r} + ||u_s - u_applied,k||^2_{R_u,ref} + ||u_s - u_s_prev||^2_{R_delta_u,sel} + ||x_s - x_s_prev||^2_{Q_delta_x} + ||x_s - xhat_k||^2_{Q_x,ref}`.

## Exposed Parameters

### `Qr_diag`
- Meaning: diagonal weight on the output-reference term `r_s - y_sp`.
- Default: inherited from the selector `Q_out` input.
- Increase it when output-target quality is not dominant enough.
- Decrease it if the selector is overfitting the requested output target and ignoring operating-region consistency.

### `alpha_u_ref`
- Meaning: multiplier used to build the applied-input anchor matrix.
- Default: `0.5`.
- Matrix construction: `R_u_ref = alpha_u_ref * R_delta_u_mpc`, unless `R_u_ref_diag` is provided directly.
- Increasing it pulls `u_s` closer to the currently applied input.
- Decreasing it allows `u_s` to move away from the current operating region more aggressively.

### `alpha_du_sel`
- Meaning: multiplier used to build the previous-input smoothing matrix.
- Default: `0.5`.
- Matrix construction: `R_delta_u_sel = alpha_du_sel * R_delta_u_mpc`, unless `R_delta_u_sel_diag` is provided directly.
- Increasing it makes `u_s` evolve more smoothly over time.
- Decreasing it allows faster target motion after a setpoint change.

### `alpha_dx_sel`
- Meaning: multiplier used to build the previous-state smoothing matrix.
- Default: `0.05`.
- Matrix construction: `Q_delta_x = alpha_dx_sel * Q_x_base`, unless `Q_delta_x_diag` is provided directly.
- Increasing it suppresses sudden motion in `x_s`.
- Decreasing it allows the state target to move more freely after large changes.

### `alpha_x_ref`
- Meaning: multiplier used to build the weak `xhat_k` anchor matrix.
- Default: `0.01`.
- Matrix construction: `Q_x_ref = alpha_x_ref * Q_x_base`, unless `Q_x_ref_diag` is provided directly.
- Increasing it pulls `x_s` more strongly toward the current observer state.
- Decreasing it reduces transient contamination of the steady target.

### `x_weight_base`
- Meaning: selects how the state-side weights are constructed.
- Supported values:
  - `"CtQC"`: `Q_x_base = C_r^T Q_r C_r`, where `C_r = C` if `H` is absent and `C_r = H C` otherwise.
  - `"identity"`: `Q_x_base = I`.
- Default: `"CtQC"`.
- `"CtQC"` ties the state weights to output sensitivity.
- `"identity"` is useful if the output map is poorly scaled or nearly singular for selector tuning.

### `use_output_bounds_in_selector`
- Meaning: whether output bounds, if supplied, are enforced in the selector.
- Default: `True`.
- Increase safety/feasibility discipline by keeping it enabled.
- Disable only if the output bounds path is known to be causing numerical problems in a specific experiment.

### `u_tight`
- Meaning: input-tightening vector used inside the selector.
- Default: zero.
- Tightened bounds become `u_min + u_tight <= u_s <= u_max - u_tight`.

### `y_tight`
- Meaning: output-tightening vector used inside the selector.
- Default: zero.
- Tightened bounds become `y_min + y_tight <= y_s <= y_max - y_tight`.

### `solver_pref`
- Meaning: preferred CVXPY solver sequence for the selector.
- Default: repository default CVXPY solver list.
- Use this when one solver is more reliable for a given experiment.

## Direct Override Matrices
The implementation also supports direct diagonal overrides:
- `R_u_ref_diag`
- `R_delta_u_sel_diag`
- `Q_delta_x_diag`
- `Q_x_ref_diag`

If one of these is provided, it overrides the corresponding alpha-based construction.

## Default Weight Hierarchy
The intended qualitative ordering is:

`Q_r >> R_u_ref ~ R_delta_u_sel > Q_delta_x > Q_x_ref`

Interpretation:
- output target quality remains dominant,
- input operating-region consistency and temporal input smoothness come next,
- state smoothing is weaker,
- the direct anchor to `xhat_k` is weakest.

## Recommended Tuning Order
1. Tune `Qr_diag` first so the selector returns acceptable `r_s` / `y_s`.
2. Tune `alpha_u_ref` so `u_s` sits near the actual operating region after setpoint changes.
3. Tune `alpha_du_sel` to suppress chattering in `u_s`.
4. Tune `alpha_dx_sel` to smooth `x_s` without freezing it.
5. Tune `alpha_x_ref` last, because this anchor is intentionally weak.

## Notes on Scaled Deviation Coordinates
- All selector quantities are expected to be in the same scaled deviation coordinates used by the controller.
- `xhat_k`, `dhat_k`, `u_applied_k`, `x_s`, `u_s`, `y_s`, and `y_sp` must all be in those same coordinates.
- Many selector bugs in this repository come from mixing physical units, min-max scaled values, and steady-state deviations.

## When to Change the `xhat` Anchor
- Increase `alpha_x_ref` when `x_s` stays unrealistically close to zero or to a stale previous target after a real operating-point change.
- Decrease `alpha_x_ref` when `x_s` becomes too transient-sensitive and starts following observer noise or short-lived excursions.
- If the observer state is known to be noisy, prefer keeping `alpha_x_ref` small and rely more on `alpha_u_ref` plus the previous-target smoothing terms.
