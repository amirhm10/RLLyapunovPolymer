# Disturbance-Only Online TD3 Runner Split

Date: 2026-06-10

## Summary

Added a shared disturbance-only runner implementation and separate root launchers for the requested online TD3, Direct LMPC, and offset-free MPC experiments. The new launchers default to disturbance mode, 300 episodes, 400-step setpoint blocks, `force_final_test=False`, and no nominal-mode branches.

## Code Changes

- Added `utils/online_disturbance_runner.py` to centralize polymer setup, scaling, observer gain, TD3 creation, checkpoint resolution, Direct LMPC/OF-MPC controller construction, reward config, training phases, debug export, and comparison records.
- Added root runners:
  - `OnlineTD3_LMPCPretrained_SafetyGate.py`
  - `OnlineTD3_OFMPCPretrained_SafetyGate.py`
  - `OnlineTD3_LMPCPretrained_NoSafetyGate.py`
  - `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`
  - `OnlineTD3_ColdStart_SafetyGate.py`
  - `OnlineTD3_ColdStart_NoSafetyGate.py`
  - `DirectLyapunovMPC_DisturbanceRunner.py`
  - `OffsetFreeMPC_DisturbanceRunner.py`
- Converted `DirectLyapunovSafetyGateRL_Pretrained.py` and `DirectLyapunovSafetyGateRL_ColdStart.py` into compatibility wrappers for the new shared implementation.
- Extended `Simulation/run_rl_lyapunov.py` with an optional `teacher_mpc_obj` argument so OF-MPC teacher behavior can be used while Direct LMPC remains the safety gate.
- Added root-level artifact mirrors in each timestamped run directory for `summary.json`, `step_table.csv`, `episode_table.csv`, and `arrays.npz`, plus `record.json`, `comparison_summary.json`, `comparison_table.csv`, and `run_summary.json`.

## Behavior

- Safety-gate TD3 runners use `projection_backend="direct_accept_or_fallback"`.
- No-safety-gate TD3 runners use `projection_backend="mpc_only_diagnostic"` and keep actual intervention/fallback penalties off while logging would-be Direct LMPC safety diagnostics.
- LMPC-pretrained runners auto-load the latest checkpoint under `results/PretrainLMPC` unless overridden by `LMPC_PRETRAINED_TD3_AGENT_PATH`, `PRETRAINED_TD3_AGENT_PATH`, or `--agent-path`.
- OF-MPC-pretrained runners auto-load the latest checkpoint under `results/PretrainOFMPC` unless overridden by `OFMPC_PRETRAINED_TD3_AGENT_PATH`, `PRETRAINED_TD3_AGENT_PATH`, or `--agent-path`.
- Pretrained runners infer TD3 actor/critic hidden layers from checkpoint metadata before constructing the agent.
- Direct LMPC and OF-MPC baseline runners save single-case disturbance bundles with comparison artifacts.

## Validation

- Static validation passed:
  - `python -m py_compile utils/online_disturbance_runner.py Simulation/run_rl_lyapunov.py OnlineTD3_LMPCPretrained_SafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_LMPCPretrained_NoSafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_ColdStart_NoSafetyGate.py DirectLyapunovMPC_DisturbanceRunner.py OffsetFreeMPC_DisturbanceRunner.py DirectLyapunovSafetyGateRL_Pretrained.py DirectLyapunovSafetyGateRL_ColdStart.py`
- Smoke validation passed with `--episodes 1 --set-points-len 5 --no-save-plots` for all eight new root runners.
- Artifact audit passed for all eight smoke roots; each created `summary.json`, `step_table.csv`, `episode_table.csv`, `arrays.npz`, `record.json`, `comparison_summary.json`, `comparison_table.csv`, and `run_summary.json`.
- Config audit confirmed all eight smoke runs used `plant_mode="disturb"` and `force_final_test=False`.

## Notes

The five-step smoke runs are intentionally tiny, so the governed-reference target selector can fall back to hold-prev in those runs. That is a smoke-test artifact; the validation target here was runner wiring, checkpoint loading, diagnostic logging, and export structure.
