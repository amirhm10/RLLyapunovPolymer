# 2026-04-30 Direct Bounded-Hard Single-Setpoint Nominal Notebook Update

## Why This Change Was Made

The direct Lyapunov notebook was previously configured as a broad ten-scenario study with two setpoints, two tests, and modified-target tracking in the MPC stage objective.

This update narrows the notebook to the specific comparison requested by the user:

- nominal plant mode
- one episode
- episode length 2000
- one setpoint only, using the first setpoint
- only two cases:
  - `bounded_hard`
  - `bounded_hard_u_prev_1p0`
- MPC stage objective tracks `y_sp`, not `y_s`

## Notebook Changes

Updated `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` so that:

- `n_tests = 1`
- `set_points_len = 2000`
- `TEST_CYCLE = [False]`
- `plant_mode = "nominal"`
- `disturbance_after_step = False`
- `use_target_output_for_tracking = False`
- `setpoint_y_phys = np.array([[4.5, 324.0]])`
- `scenario_matrix` contains only:
  - `bounded_hard`
  - `bounded_hard_u_prev_1p0`
- `study_name` now saves under:
  - `direct_lyapunov_mpc_bounded_hard_single_setpoint_nominal`

The notebook markdown cells were also updated so the visible description matches the new focused configuration.

## Output Handling

- Cleared notebook outputs
- Reset code-cell execution counts to avoid stale saved results from the previous ten-scenario configuration

## Validation

- JSON load check passed for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
- `nbformat.validate` passed for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`

## Notes

- This change updates the notebook configuration only. It does not regenerate results.
- Existing reports based on earlier saved direct-study runs may no longer describe the current notebook defaults exactly until the notebook is rerun.
