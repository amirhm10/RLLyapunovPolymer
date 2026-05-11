# Phase-Aware Teacher Warm Start And Parameter Noise

Date: 2026-05-11

## Summary

Implemented the new direct safety-gate RL training schedule:

- warm start uses `direct_lyapunov_mpc` teacher data with no policy noise
- BC uses `direct_lyapunov_mpc` teacher data with no policy noise
- full RL uses parameter-noise behavior exploration
- test/eval episodes remain deterministic

The implementation is shared in the TD3 agent and RL rollout path, and both direct safety-gate notebooks now opt into the new defaults.

## Main code changes

- Added a perturbed behavior-actor path in [TD3Agent/agent.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/TD3Agent/agent.py>) for parameter-noise exploration.
- Added parameter-noise adaptation based on nominal-versus-perturbed action deviation, with the configured target and bounds.
- Added phase-specific behavior-noise settings and cycle-boundary parameter-noise resampling in [Simulation/run_rl_lyapunov.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Simulation/run_rl_lyapunov.py>).
- Added rollout diagnostics for:
  - `behavior_noise_mode`
  - `parameter_noise_active`
  - `parameter_noise_std`
  - `parameter_noise_resampled_this_step`
  - `behavior_action_pre_filter`
- Updated both direct safety-gate notebooks to use:
  - teacher warm start
  - no-noise warm start
  - no-noise BC
  - parameter-noise full RL
  - zeroed Gaussian behavior-noise defaults

## Updated notebooks

- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/DirectLyapunovSafetyGateRL_Pretrained.ipynb>)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/DirectLyapunovSafetyGateRL_ColdStart.ipynb>)

## Validation

- Compiled [TD3Agent/agent.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/TD3Agent/agent.py>) in memory.
- Compiled [Simulation/run_rl_lyapunov.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Simulation/run_rl_lyapunov.py>) in memory.
- Verified both notebooks remain valid JSON and compiled every code cell in memory.
- Ran a small agent-level behavior check confirming:
  - parameter-noise actions are stable within one cycle
  - actions change after resampling
  - the parameter-noise std adapts and remains bounded
