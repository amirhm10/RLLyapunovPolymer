# Direct Setpoint Segment Length 1500

## Summary

Updated `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` so the direct
Lyapunov ten-scenario study uses longer constant-setpoint windows.

## Changes

- Changed:
  - `set_points_len = 400`
- To:
  - `set_points_len = 1500`

This makes each setpoint segment long enough to test whether the current best
regularized controller eventually settles after the initial transient.

## Validation

- Parsed the notebook JSON successfully.
- Confirmed `set_points_len = 1500` is present.
- Confirmed `set_points_len = 400` is no longer present in the direct
  Lyapunov notebook.
- Did not execute the notebook because this change makes the ten-scenario
  rollout substantially longer.
