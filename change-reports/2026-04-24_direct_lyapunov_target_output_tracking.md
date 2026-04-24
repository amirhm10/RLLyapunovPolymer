# Direct Lyapunov Target-Output Tracking

## Summary

Updated the direct frozen-output-disturbance Lyapunov MPC notebook so the
controller tracks the constraint-aware modified reference `y_s` rather than the
raw requested reference `y_sp`.

## Reference Semantics

- Raw requested reference: `y_sp`
- Constraint-aware modified reference: `y_s`
- Controller stage target: `y_s`
- External performance metrics: both `y_s - y_sp` and `y - y_sp`

## Changes

- Set `use_target_output_for_tracking = True` in
  `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`.
- Added a notebook markdown note documenting the distinction between raw
  requested reference, modified target, controller target, and external
  performance metrics.
- Extended `Lyapunov/direct_lyapunov_mpc.py` exports with:
  - per-step `y_s_minus_y_sp`
  - per-step `y_target_minus_y_sp`
  - per-step `y_minus_y_sp`
  - per-step `y_minus_y_target`
- Added bundle arrays, summary fields, comparison table columns, and a
  comparison plot for the new reference-error metrics.

## Validation

- `python -m py_compile Lyapunov/direct_lyapunov_mpc.py`
- `nbformat.validate` on `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`

