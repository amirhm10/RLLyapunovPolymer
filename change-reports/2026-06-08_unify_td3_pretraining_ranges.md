# Unify TD3 Pretraining and Evaluation Ranges

Date: 2026-06-08

## Summary

Centralized the Polymer TD3 normalization defaults so pretraining, saved-agent comparison, and the active Lyapunov pretrained workflow use the same state and setpoint ranges.

## Changes

- Added `utils/polymer_td3_defaults.py` with:
  - default augmented-state TD3 scaling bounds from the previous Polymer pretrained workflow
  - default direct two-setpoint output envelope `[[4.5, 324.0], [3.4, 321.0]]`
  - default physical input bounds `[71.6, 78.0]` to `[870.0, 670.0]`
- Updated `utils.td3_helpers.load_and_prepare_system_data(...)` so the default `min_max_dict["x_min"]` and `["x_max"]` come from the centralized Polymer TD3 state bounds.
- Updated `utils/of_mpc_td3_workflow.py` so both pretraining and comparison use the centralized direct two-setpoint envelope.
- Updated `utils/direct_lyapunov_study.py` so the active Lyapunov direct setpoint schedule shares the same centralized default.
- Updated `report/of_mpc_td3_pretraining_process_2026-06-08.md` to document the single default range source.

## Rationale

The previous migrated OF-MPC TD3 workflow reconstructed state and setpoint scaling ranges from local data and local setpoint arrays. That allowed the saved Lyapunov TD3 checkpoint, the pretraining runner, the comparison runner, and the Lyapunov pretrained runner to receive different normalized features for the same physical plant condition.

This change makes the TD3 feature normalization default explicit and shared.

## Validation

Completed low-cost checks:

```powershell
python -m py_compile utils/polymer_td3_defaults.py utils/td3_helpers.py utils/of_mpc_td3_workflow.py utils/direct_lyapunov_study.py PretrainTD3OffsetFreeMPC.py ComparePretrainedTD3OffsetFreeMPC.py DirectLyapunovSafetyGateRL_Pretrained.py
```

Also checked that the runner path returns:

- `state_bounds_source = default_polymer_td3`
- `min_max_dict["x_min"]` and `["x_max"]` matching `utils/polymer_td3_defaults.py`
- direct setpoint-derived `y_sp_min = [-2.210728, -3.332520]`
- direct setpoint-derived `y_sp_max = [2.751989, 0.506069]`
