# Align Online GART Logs And Config Values

## Summary

This update aligns the cold-start online TD3 GART path with the standalone
`GARTLyapunovMPC.py` runner while keeping pretrained Direct/OF-MPC presets on
their existing settings.

## Changes

- Added shared final GART constants in `utils/gart_defaults.py`:
  - `rho = 0.98`
  - `eps = 1.0e-3`
  - `slack_penalty = 1.0e6`
  - `objective = "raw"`
  - `lyapunov_mode = "hard"`
- Added `GART_FINAL_TARGET_CONFIG_OVERRIDES`, which combines the final target
  overrides with the final GART target `rho` and `eps`.
- Updated the standalone GART study and root GART runner path to use the shared
  constants through `experiments/run_gart_target_selector_study.py`.
- Updated cold-start online TD3 GART presets so the target selector,
  GART-LMPC teacher, and GART-LMPC fallback use the standalone GART values
  `rho=0.98` and `eps=1.0e-3`.
- Preserved pretrained online Direct/OF-MPC presets on their existing Direct
  values.
- Added `controller_family`, `lyap_param_source`, and JSON-safe GART config
  source fields to online run summaries.
- Changed cold-start safety block summaries to print `GART safety gate` instead
  of a stale Direct-facing label.
- Added GART target diagnostics to runtime block prints and `step_table.csv`:
  - `target_rejection_reason`
  - `target_usable_for_lmpc`
  - `contraction_probe_margin`
  - `governor_alpha`

## Validation

Completed checks:

```powershell
python -m py_compile Simulation/run_rl_lyapunov.py utils/online_disturbance_runner.py utils/gart_defaults.py experiments/run_gart_target_selector_study.py Lyapunov/safety_debug.py
& "C:\Users\hamed\miniconda3\envs\rlenv\python.exe" -c "<standalone/online GART constant audit>"
& "C:\Users\hamed\miniconda3\envs\rlenv\python.exe" OnlineTD3_ColdStart_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
& "C:\Users\hamed\miniconda3\envs\rlenv\python.exe" OnlineTD3_ColdStart_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
```

The smoke summaries and first `step_table.csv` rows showed:

- `target_mode="gart"`
- `controller_family="gart_lmpc"`
- `rho_lyap=0.98`
- `lyap_eps=0.001`
- `gart_lmpc_objective="raw"`
- `gart_lmpc_lyapunov_mode="hard"`
- the four new GART target diagnostic columns were present
- safety-gate run kept `fallback_controller="gart_lmpc"` and
  `safety_gate_active=True`
- no-safety run kept `fallback_controller="none"` and
  `safety_gate_active=False`

`pytest tests/test_gart_target.py` was not run because `pytest` is not installed
in the default shell or the `rlenv` environment.
