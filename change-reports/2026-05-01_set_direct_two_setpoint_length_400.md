# 2026-05-01 Set Direct Two-Setpoint Length to 400

## Why This Change Was Made

The direct Lyapunov notebook had just been restored to the earlier two-setpoint schedule, but the segment length was set back to `1500`.

The user clarified that the intended segment length is now `400`, not `1500`.

This change keeps the restored two-setpoint schedule while shortening each setpoint segment to the current requested episode length.

## Notebook Changes

Updated `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` so that:

- `set_points_len = 400`

All other restored two-setpoint settings remain unchanged, including:

- `n_tests = 2`
- `setpoint_y_phys = np.array([[4.5, 324.0], [3.4, 321.0]])`
- `study_name = "direct_lyapunov_mpc_bounded_three_scenario_two_setpoint_nominal"`

## Output Handling

- Notebook outputs remained cleared
- Execution counts remained reset

## Validation

- JSON load check passed for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
- `nbformat.validate` passed for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`

## Notes

- This change updates the notebook configuration only. It does not rerun the experiment.
