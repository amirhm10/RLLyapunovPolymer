# Restore Polymer TD3 Setpoint Scaler Envelope

Date: 2026-06-09

## Summary

Restored the exact Polymer-example convention for the old pretrained TD3 checkpoint:

- TD3 feature scaling uses the broad pretraining setpoint envelope `[[2.8, 320.0], [5.0, 326.0]]`.
- Rollout and comparison still use the direct two-setpoint scenario `[[4.5, 324.0], [3.4, 321.0]]`.
- Augmented-state scaling continues to use the previous manually defined Polymer TD3 bounds.

## Why

The latest comparison run with the direct setpoint range used as the TD3 scaler showed a visible mismatch:

- broken scaler run: `results/PretrainOFMPCComparison/20260609_000554`
- nominal TD3 mean RMSE: `1.2721`
- disturbed TD3 mean RMSE: `1.2931`

The archived Polymer-example evaluation path loaded system data with the broad pretraining envelope, then created the direct rollout setpoint schedule separately. The saved checkpoint `Data/agent_2507171027.pkl` expects that broad setpoint scaler.

## Changes

- Added separate centralized constants in `utils/polymer_td3_defaults.py`:
  - `DEFAULT_TD3_SETPOINT_SCALER_Y_PHYS`
  - `DEFAULT_DIRECT_SETPOINT_Y_PHYS`
- Updated `utils.td3_helpers.load_and_prepare_system_data(...)` to use the broad TD3 setpoint scaler by default for `min_max_dict["y_sp_min"]` and `["y_sp_max"]`.
- Added a shape check so the setpoint scaler envelope must match the output dimension.
- Kept `utils.direct_lyapunov_study.DIRECT_TWO_SETPOINT_Y_PHYS` on the direct rollout scenario.
- Updated `utils/of_mpc_td3_workflow.py` so OF-MPC TD3 pretraining uses the broad scaler envelope and comparison uses the direct scenario.
- Added scaling metadata to comparison `summary.json` bundles.
- Updated the OF-MPC TD3 process report to document the scaler/scenario distinction.

## Validation

Static validation passed:

```powershell
python -m py_compile utils/polymer_td3_defaults.py utils/td3_helpers.py utils/of_mpc_td3_workflow.py utils/direct_lyapunov_study.py PretrainTD3OffsetFreeMPC.py ComparePretrainedTD3OffsetFreeMPC.py DirectLyapunovSafetyGateRL_Pretrained.py
```

Range check confirmed:

- `state_bounds_source = default_polymer_td3`
- `setpoint_bounds_source = default_polymer_td3_scaler`
- `min_max_dict["y_sp_min"] = [-4.917664, -4.612049]`
- `min_max_dict["y_sp_max"] = [5.007769, 3.065128]`
- rollout scenario scaled deviations remain `[[2.751989, 0.506069], [-2.210728, -3.332520]]`

Full compare rerun:

```powershell
python ComparePretrainedTD3OffsetFreeMPC.py --agent-path Data/agent_2507171027.pkl --modes both --n-tests 2 --set-points-len 400
```

Result bundle:

- `results/PretrainOFMPCComparison/20260609_001124`

Key metrics:

- nominal TD3 mean RMSE: `0.3566`
- nominal OF-MPC mean RMSE: `0.3554`
- disturbed TD3 mean RMSE: `0.3587`
- disturbed OF-MPC mean RMSE: `0.3569`

The large scaler-induced mismatch is removed.
