# 2026-05-11 Direct Four-Method Disturbance Settling Analysis

## What changed

Added a new analysis report:

- `report/direct_four_method_disturbance_settling_analysis_2026-05-11.md`

Added a supporting figure and embedded it directly in the report:

- `direct_four_method_disturbance_last_episode_target_vs_setpoint_2026-05-11.png`

## Why

The latest results from `DirectLyapunovMPC_FourMethodDisturbance.ipynb` still show a reach-then-oscillate pattern, and the question was whether this meant the earlier settling fix in the target-selector objective had regressed.

## Main findings

- The direct disturbance notebook is not using the same selector path that had earlier settling-focused improvements.
- The notebook still sets `use_target_output_for_tracking = False`, so the controller tracks raw setpoint while the bounded disturbance target generator may select a different admissible target.
- In the latest disturbance export, the selected target still moves significantly near the end for `bounded_hard` and `bounded_hard_xs_prev_0p1`.
- `bounded_hard_u_prev_0p1` is the most stable latest case and shows that previous-input anchoring helps much more than state-smoothness alone in this notebook.
- The latest export folder is missing the combined anchor-plus-smoothness case, so the latest four-method sweep appears incomplete.

## Validation

- Read the notebook configuration and direct Lyapunov source path.
- Read the latest disturbance and nominal debug exports.
- Computed final-episode physical-unit metrics from saved bundles.
- Generated and embedded a final-episode comparison figure.
