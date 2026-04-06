# Steady-State Box Analysis Parameters

## Purpose

This note documents the parameters used by the bounded steady-state sidecar in `analysis/steady_state_debug_analysis.py`. The bounded branch is analysis-only. It does not change the offset-free MPC controller or feed any target back into the closed loop.

## Core Switches

- `enabled`
  - Master switch for the steady-state sidecar.
  - Default: `True`
  - Set to `False` only when the notebook should skip all sidecar diagnostics.

- `enable_box_analysis`
  - Enables the box-constrained least-squares expansion on top of the existing unbounded solve.
  - Default: `True`
  - When `False`, the module keeps only the original unbounded steady-state analysis outputs.

## Exact-Solve Diagnostics

- `solver_mode`
  - Controls the unbounded steady-state solve policy.
  - Default: `"auto"`
  - Supported values: `"auto"`, `"stacked_exact"`, `"stacked_lstsq"`, `"reduced_exact"`, `"reduced_lstsq"`.
  - `"auto"` is recommended because it preserves the exact solve when reliable and falls back cleanly when needed.

- `cond_warn_threshold`
  - Numerical conditioning threshold used to decide whether exact linear solves are reliable enough to trust directly.
  - Default: `1e8`
  - Increase only if the model is known to remain numerically stable at larger condition numbers.

- `residual_warn_threshold`
  - Residual threshold used to classify the unbounded solve as exact or unsolved.
  - Default: `1e-8`
  - Larger values are more permissive; smaller values are stricter.

- `rank_tol`
  - Optional rank tolerance passed into matrix-rank checks and least-squares routines.
  - Default: `None`
  - Leave as `None` unless a specific rank-sensitivity problem has been identified.

## Box-Constrained Branch

- `box_bound_tol`
  - Tolerance used for box-membership checks and active-bound detection.
  - Default: `1e-9`
  - Increase slightly if floating-point noise causes false bound-violation flags.

- `box_use_reduced_first`
  - If `True`, the bounded branch first solves the reduced box-constrained least-squares problem
    \[
    \min_{u_s} \| G u_s - r_k \|_2^2
    \]
    before falling back to the full least-squares solve in `[x_s; u_s]`.
  - Default: `True`
  - Recommended setting for clarity and speed when `(I-A)` is invertible.

## Event Analysis Parameters

- `box_event_window_radius`
  - Number of time steps shown on each side of an event anchor in the event-focused table and zoomed plots.
  - Default: `5`
  - Increase for broader context around setpoint changes or disturbance-estimate jumps.

- `box_dhat_event_threshold`
  - Infinity-norm threshold on `dhat_k - dhat_{k-1}` used to mark a disturbance-estimate jump event.
  - Default: `5e-2`
  - This value is interpreted in the same scaled-deviation coordinates used by the observer and sidecar analysis.

- `box_max_event_plots`
  - Maximum number of event-window figures written to the output directory.
  - Default: `6`
  - Lower this if event plots become too numerous.

## Output and Plotting

- `save_csv`
  - Writes CSV summaries including `step_table.csv`, `box_overall_summary.csv`, `box_per_input_activity.csv`, and `box_event_table.csv`.
  - Default: `True`

- `save_plots`
  - Writes all standard unbounded plots plus the bounded plot group.
  - Default: `True`

- `sample_table_stride`
  - Downsampling stride used for the sampled markdown and notebook summary table.
  - Default: `10`
  - Larger values make the summary lighter; smaller values expose more time steps.

- `case_name`
  - Label written into the bundle and summary markdown.
  - Default: `"disturbance"`
  - For the current notebook this should remain the disturbance run label.

## Selected Run Shape

The current `MPCOffsetFree_SteadyStateDebug.ipynb` notebook uses:

- the normal disturbance closed-loop OF-MPC run
- the same `run_mpc(...)` controller path as before
- the unaugmented `A`, `B`, and `C` matrices for the steady-state analysis
- `system_data["b_min"]` and `system_data["b_max"]` as the steady-state input box in scaled-deviation coordinates

## Solver Notes

- The bounded branch prefers `scipy.optimize.lsq_linear`.
- Reduced form is used first when `G = C (I-A)^{-1} B` is available and numerically usable.
- If reduced form is unavailable, the module solves the full box-constrained least-squares problem in `[x_s; u_s]`.
- If SciPy is unavailable, bounded analysis is marked as failed instead of crashing the sidecar.
