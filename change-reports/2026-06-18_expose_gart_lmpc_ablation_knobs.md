# Expose GART-LMPC Ablation Knobs

Date: 2026-06-18

## Objective

Make the GART-LMPC target-selector ablation settings visible and easy to change from the root runner.

## Change

`GARTLyapunovMPC.py` now exposes the target-rate, disturbance-update, and target-selector geometry settings as top-level constants. It also adds:

```python
ABLATION_CASE = "candidate"
```

with available cases:

```python
"candidate"
"A_wy"
"B_headroom"
"C_u_smooth"
"D_primary_tol"
```

The candidate settings remain:

```python
CANDIDATE_INPUT_HEADROOM_FRAC = 0.03
CANDIDATE_PRIMARY_TOL_REL = 1.0e-4
CANDIDATE_W_U_SMOOTH_DIAG = [2.0, 2.0]
CANDIDATE_TARGET_WY_DIAG = [1.0, 1.0]
```

The ablation cases change one suspected cause at a time:

```python
"A_wy": {"Wy_diag": [5.0, 1.0]}
"B_headroom": {"input_headroom_frac": 0.01}
"C_u_smooth": {"W_u_smooth_diag": [1.0, 1.0]}
"D_primary_tol": {"primary_tol_rel": 1.0e-6}
```

## Method Interpretation

The ablation cases test which part of the target-selector geometry caused the reduction in input-bound target solutions:

$$
\min_{x_s,u_s} \|W_y(y_s-r)\|^2
$$

subject to the steady-state relation, input bounds, rate limits, and GART contraction probe. The MPC objective remains unchanged.

## How To Use

Change one line in `GARTLyapunovMPC.py` before running:

```python
ABLATION_CASE = "A_wy"
```

The selected case and full ablation menu are saved into the run summary.

## Metrics To Compare

For each case, compare:

- `u_s` near-bound rate
- `mean |u_s - u_prev_applied|_inf`
- `target_reference_error_inf_mean`
- `output_rmse_raw_ysp`
- `hard_contraction_rate`
- `executed_contraction_violation_max`
