# Direct Lyapunov Method Step-by-Step Report

Date: 2026-05-13

## Summary

Added a new method report that reconstructs the current direct Lyapunov algorithm family mathematically and step by step across:

- `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

The new report does not focus on results. It focuses on the actual implemented method:

1. augmented plant and observer model
2. RL observation and action mapping
3. frozen output-disturbance target selection
4. Lyapunov candidate acceptance test
5. direct Lyapunov MPC fallback
6. plant step, reward, and replay update
7. the exact mismatch between admissible-target contraction and raw-setpoint tracking

## Files added

- [report/direct_lyapunov_method_step_by_step_2026-05-13.md](../report/direct_lyapunov_method_step_by_step_2026-05-13.md)

## Validation

- Cross-checked the target equations against `Lyapunov/frozen_output_disturbance_target.py`.
- Cross-checked the gate equations against `Lyapunov/lyapunov_core.py`.
- Cross-checked the fallback optimization and target-routing logic against `Lyapunov/direct_lyapunov_mpc.py`.
- Cross-checked the RL state, reward, and replay update flow against `Simulation/run_rl_lyapunov.py`, `utils/helpers.py`, and `TD3Agent/reward_functions.py`.
