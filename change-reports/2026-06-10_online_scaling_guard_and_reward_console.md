# Online Scaling Guard And Reward Console

## Summary

- Added a runtime TD3 scaling contract check to the online disturbance runner.
- Added `scaling_contract` and `online_td3_hparams` metadata to online TD3 run configs.
- Updated cycle-boundary console output to print:
  - actual training reward
  - `reward_no_penalty`
  - average fallback penalty
- Updated the online runner algorithm audit to explain why no-gate raw reward can look much better while still failing Direct LMPC gate diagnostics.

## Why

The online pretrained runners must preserve the distinction between:

- the broad TD3 setpoint feature scaler used during pretraining: `[[2.8, 320.0], [5.0, 326.0]]`
- the direct comparison rollout setpoints: `[[4.5, 324.0], [3.4, 321.0]]`

The code now fails loudly if that contract is broken. The console output also separates raw training reward from penalty-free control reward so safety-gate and no-gate runs are not compared using different reward definitions.

## Validation

- Passed: `python -m py_compile utils/online_disturbance_runner.py Simulation/run_rl_lyapunov.py`
- Passed: direct scaling-contract check:
  - `setpoint_bounds_source = default_polymer_td3_scaler`
  - `td3_setpoint_scaler_y_phys = [[2.8, 320.0], [5.0, 326.0]]`
  - `rollout_setpoint_y_phys = [[4.5, 324.0], [3.4, 321.0]]`
  - `y_sp_min = [-4.917664, -4.612049]`
  - `y_sp_max = [5.007769, 3.065128]`
- Passed: online agent hyperparameter check after loading the OF-MPC checkpoint:
  - online `policy_delay = 2`
  - checkpoint metadata `policy_delay = 4`
  - online `gamma = 0.99`
- Passed smoke wiring:
  - `python OnlineTD3_OFMPCPretrained_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots`
  - console printed `avg. reward`, `avg. reward_no_penalty`, and `avg. fallback penalty`
