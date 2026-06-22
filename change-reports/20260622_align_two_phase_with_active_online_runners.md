# Align Two-Phase Runners With Active Online Runner Settings

Date: 2026-06-22

## Summary

Aligned the two-phase TD3 method runners with the archived active online runner settings.

## Changes

- Reverted behavior exploration starts:
  - pretrained TD3: `0.04 -> 0.02`
  - cold-start TD3: `0.15 -> 0.10`
- Kept target-policy smoothing unchanged:
  - pretrained: smoothing `0.02`, clip `0.04`
  - cold-start: smoothing `0.10`, clip `0.20`
- Made active-runner settings explicit in the four TD3 two-phase runners:
  - `RL_OBSERVATION_MODE = "standard"`
  - safety-gate runners use `PROJECTION_BACKEND = "direct_accept_or_fallback"`
  - no-gate runners use `PROJECTION_BACKEND = "mpc_only_diagnostic"`
  - `REWARD_FALLBACK_PENALTY_ENABLED = False`
  - `GAMMA_FALLBACK = 0.0`
  - `FALLBACK_EVENT_PENALTY = 0.0`
  - GART Lyapunov constants are exposed as `RHO_LYAP`, `LYAP_EPS`, and `LYAP_TOL`
- Extended the shared two-phase runner to pass these explicit settings into `run_online_td3_disturbance_preset(...)`.

## Validation

- Passed `py_compile` on the shared runner, all method runners, and `TD3Agent/agent.py`.
- Passed a tiny OF-MPC-pretrained safety-gate smoke run.
- Passed a tiny cold-start no-safety-gate smoke run.
- Smoke configs confirmed reward fallback penalties are disabled and `reward` equals `reward_no_penalty`.
