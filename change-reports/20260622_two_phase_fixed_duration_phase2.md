# Two-Phase Fixed-Duration Phase-2 Update

## Objective

Update the two-phase online TD3/GART experiment so Phase 1 uses 150 true learning episodes, while Phase 2 is treated as a fixed-duration robustness segment instead of another episode schedule.

## Changes

- Changed the shared two-phase profile defaults:
  - Phase 1: `150` learning episodes.
  - Phase 1 setpoint hold: `400` samples per setpoint.
  - Phase 2: `10000` continuous plant steps.
  - Reporting window: `400` samples.
  - Phase-2 held setpoint: `[[3.3, 323.0]]`.
- Updated the five two-phase method runners to expose:
  - `PHASE2_STEPS`
  - `PHASE1_SETPOINT_HOLD_STEPS`
  - `REPORTING_WINDOW_STEPS`
- Updated `RunOnlineTD3TwoPhaseStudy.py` so the old rollout API receives compatible reporting windows:
  - `rollout_n_tests = 325`
  - `rollout_set_points_len = 200`
  - total steps `150 * 800 + 10000 = 130000`
- Updated the OF-MPC Phase-2 probe to run the same `10000`-step held-setpoint robustness profile, reported as `25` windows of `400` samples.
- Added explicit reporting-window and phase-step metadata to TD3 safety and GART/direct exports:
  - `report_window`
  - `step_in_report_window`
  - `phase_step`
  - `phase1_episode`
  - `phase2_report_window`
- Extended `phase_table.csv` rows with reporting-window and learning-episode labels, so Phase 1 can still be interpreted as 150 learning episodes while the exported episode table uses 400-sample reporting windows.

## Validation

- Profile checks passed:
  - total steps `130000`
  - Phase-1 steps `120000`
  - Phase-2 steps `10000`
  - total reporting windows `325`
  - Phase-1 reporting windows `300`
  - Phase-2 reporting windows `25`
  - rollout setpoint length `200`
  - Phase-2 physical setpoint `[[3.3, 323.0]]`
  - exploration decay reaches `0.005` at the end of Phase 1 and remains fixed afterward.
- OF-MPC Phase-2 probe profile checks passed:
  - profile shape `(10000, 2)`
  - `rollout_n_tests = 25`
  - `rollout_set_points_len = 200`
  - physical setpoint metadata `[[3.3, 323.0]]`.
- `py_compile` was attempted, but bytecode writes were blocked by the Windows/OneDrive cache permissions. An in-memory Python syntax compile passed for all touched Python files.
