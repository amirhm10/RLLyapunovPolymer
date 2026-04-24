# Restore Direct Lyapunov Raw-Setpoint Tracking

## Summary

Restored the direct frozen-output-disturbance Lyapunov MPC notebook default so
the MPC stage objective tracks the raw requested setpoint `y_sp` instead of the
constraint-aware modified target output `y_s`.

## Changes

- Set `use_target_output_for_tracking = False` in
  `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`.
- Updated the notebook reference-semantics note to state that `y_sp` tracking
  is the default and `y_s` tracking is only a diagnostic variant.
- Added `report/direct_lyapunov_target_output_tracking_analysis.md` analyzing
  the latest `use_target_output_for_tracking=True` run.

## Finding

The target-output run made the controller track `y_s` very accurately, but
`y_s` moved far from `y_sp`. The result was high feasibility and low
`y-y_s` error, but poor external `y-y_sp` performance. The default should
therefore remain raw-setpoint tracking while the report keeps both
`y_s-y_sp` and `y-y_sp` as diagnostics.

## Validation

- `nbformat.validate` on `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
- `git diff --check`

