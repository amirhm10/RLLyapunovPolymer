# GART Direct Runner Parity

## Summary
Updated the GART closed-loop runner so the experiment setup matches `DirectLyapunovMPC.py` more closely. The closed-loop comparison should differ only in the GART target selector and GART MPC objective path, while setpoint generation, scaling convention, plant rollout timing, observer update, and debug plotting use the direct Lyapunov conventions.

## Changes
- Switched the GART runner setpoint source to `DIRECT_TWO_SETPOINT_Y_PHYS`.
- Switched the GART runner test-cycle source to `direct_disturbance_test_cycle(...)`.
- Restored the CLI default `--set-points-len` to the direct-study default of 400.
- Added direct-style GART step diagnostics for `d_s`, `r_cmd`, target gaps, governor probe fields, and target residuals.
- Saved each closed-loop case through `build_direct_lyapunov_run_bundle(...)` and `save_direct_lyapunov_debug_artifacts(...)`, so plots use the same scaled-deviation-to-physical conversion as the direct runner.

## Result Interpretation
The existing `results/GARTLMPC/20260613_231610` smoke run used `set_points_len=20`, giving only 10 hours per setpoint at `delta_t=0.5`. That is too short for the usual polymer CSTR tracking comparison and makes even the old governed-reference baseline look poor. The next fair nominal check should use the restored 400-step dwell with `n_tests=5`.

## Validation
- `python -m py_compile experiments/run_gart_target_selector_study.py GARTLyapunovMPC.py`

Runtime closed-loop validation still requires the scientific Python environment with NumPy, CVXPY, SciPy, control, scikit-learn, torch, joblib, and matplotlib installed.
