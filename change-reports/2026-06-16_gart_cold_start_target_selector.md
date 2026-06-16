# GART Target Selector For Cold-Start Online TD3

## Summary

The two cold-start online TD3 runners now use the GART target selector while retaining the existing TD3 action space, reward, Direct LMPC fallback solver, and no-safety diagnostic structure.

## Changes

- Added a GART branch to `prepare_direct_output_disturbance_step(...)` that converts `GARTTargetResult` into the existing direct-style target package expected by Lyapunov candidate evaluation and Direct LMPC fallback.
- Threaded `GARTTargetState` through `run_rl_train(...)` so certified disturbance and governed-reference memory persist across online steps.
- Added preset-level target selector selection in `utils/online_disturbance_runner.py`; only `cold_start_safety_gate` and `cold_start_no_safety_gate` use `target_mode="gart"`.
- Moved the final GART target override values into `utils/gart_defaults.py` and reused them from the standalone GART study and cold-start online runner.
- Added GART stage labels to safety debug target-stage encoding.

## Scientific Interpretation

This edit changes the governed steady target used by the cold-start safety diagnostics and safety gate, not the RL policy architecture or reward. The safety-gate run now tests TD3 candidate actions against Lyapunov contraction around GART-selected `(x_s, u_s, y_s, d_s)`. The no-safety run still applies the TD3 or teacher action directly, but its would-be Lyapunov diagnostics are computed against the same GART target selector.

## Validation

Planned checks:

```powershell
python -m py_compile Simulation/run_rl_lyapunov.py Lyapunov/direct_lyapunov_mpc.py utils/online_disturbance_runner.py utils/gart_defaults.py Lyapunov/safety_debug.py experiments/run_gart_target_selector_study.py
pytest tests/test_gart_target.py
python OnlineTD3_ColdStart_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
python OnlineTD3_ColdStart_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
```

The short smoke runs should be used only as wiring checks. Full performance interpretation still requires longer disturbed cold-start runs.
