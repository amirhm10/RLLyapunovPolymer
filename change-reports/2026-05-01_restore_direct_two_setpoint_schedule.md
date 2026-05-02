# 2026-05-01 Restore Direct Two-Setpoint Schedule

## Why This Change Was Made

The direct Lyapunov notebook had been narrowed to a single-setpoint nominal study while the controller behavior was still being stabilized.

Now that the direct formulation is working again, the user requested a return to the earlier schedule structure:

- two setpoints
- the earlier episode length
- the same direct notebook entrypoint

The goal of this change is to restore the longer two-step reference schedule without undoing the current direct-controller architecture or the current bounded three-case scenario study.

## Notebook Changes

Updated `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` so that:

- `n_tests = 2`
- `set_points_len = 1500`
- `setpoint_y_phys = np.array([[4.5, 324.0], [3.4, 321.0]])`
- `study_name` now saves under:
  - `direct_lyapunov_mpc_bounded_three_scenario_two_setpoint_nominal`

The notebook markdown cells were also updated so the visible description matches the restored two-setpoint schedule and export path.

## Output Handling

- Notebook outputs remained cleared
- Execution counts remained reset

## Validation

- JSON load check passed for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
- `nbformat.validate` passed for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`

## Notes

- This change restores the notebook configuration only. It does not rerun the experiment.
- The bounded three-case nominal comparison remains in place; only the schedule length and setpoint count were restored.
