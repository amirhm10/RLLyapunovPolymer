# GART Default Disturbance Mode

## Summary
Changed the GART runner CLI default from nominal mode to disturbance mode.

## Behavior
- `python GARTLyapunovMPC.py --closed-loop` now runs with `--mode disturb`.
- Nominal mode remains available explicitly with `--mode nominal`.
- The default setpoint block length remains the direct-study default of 400.

## Validation
- `python -m py_compile experiments/run_gart_target_selector_study.py GARTLyapunovMPC.py`
