# Offset-Free MPC Phase-2 Feasibility Probe

## Objective

Add a short offset-free MPC runner that tests candidate Phase-2 continuation setpoints and disturbance ramp before using that profile in the full online RL comparison.

## Changes

- Updated `OffsetFreeMPC_DisturbanceRunner.py` from a simple five-episode wrapper into an editable Phase-2 feasibility runner.
- The runner now executes two episodes with the selected candidate-3 setpoints:
  - setpoints `[[4.15, 323.0], [3.35, 323.5]]`
  - `set_points_len = 400`
  - total profile length `1600` plant steps
- The disturbance ramp starts from the planned Phase-1-final values:
  - `Qi = 102.6`
  - `Qs = 481.95`
  - `hA = 966000.0`
- The disturbance ramp ends at the planned Phase-2-final values:
  - `Qi = 113.4`
  - `Qs = 436.05`
  - `hA = 924000.0`
- Added explicit setpoint-profile support to the direct and offset-free diagnostic MPC rollout functions.
- Extended `run_offset_free_mpc_disturbance(...)` to pass explicit setpoint and disturbance profiles and save lightweight profile metadata.

## Validation

- Compiled:
  - `OffsetFreeMPC_DisturbanceRunner.py`
  - `utils/online_disturbance_runner.py`
  - `Lyapunov/direct_lyapunov_mpc.py`
- Checked the generated probe profile:
  - setpoint shape `(1600, 2)`
  - disturbance lengths `1600` for `qi`, `qs`, and `ha`
  - disturbance endpoints match the intended Phase-2 feasibility test.
- Ran the two-episode OF-MPC probe for three Phase-2 candidates:
  - original `[[4.4, 321.5], [3.3, 324.5]]`: reward mean `-39.07`, mean output RMSE `0.803`, `Qc` hit the upper bound.
  - candidate 2 `[[4.25, 322.5], [3.35, 323.5]]`: reward mean `-18.60`, mean output RMSE `0.531`, but the second episode still reached the `Qc` upper bound.
  - candidate 3 `[[4.15, 323.0], [3.35, 323.5]]`: reward mean `-12.61`, mean output RMSE `0.427`, and `Qc` remained below the upper bound in both episodes.

Candidate 3 is the current selected profile for the root feasibility runner.
