# GART Editable Root Runner

## Summary
Converted `GARTLyapunovMPC.py` from a thin CLI wrapper into a direct-style editable experiment runner. The root file now exposes the same kind of top-level knobs used in `DirectLyapunovMPC.py`, while keeping the implementation in `experiments/run_gart_target_selector_study.py`.

## Added Runner Controls
- `MODE`
- `N_TESTS`
- `SET_POINTS_LEN`
- `RUN_TARGET_ONLY`
- `RUN_CLOSED_LOOP`
- `TIMESTAMP`
- `CASE_SPECS` with per-case `enabled` flags
- `ALLOW_CLI_OVERRIDES`

## Experiment Module Change
`run_closed_loop(...)` now accepts an optional `case_specs` list, so the root runner can turn individual GART cases on or off without editing the experiment internals.

## Default Behavior
The root runner now defaults to closed-loop disturbance mode with target-only diagnostics disabled. This avoids the long target-only prepass during normal performance monitoring.

## Validation
- `python -m py_compile GARTLyapunovMPC.py experiments/run_gart_target_selector_study.py`
