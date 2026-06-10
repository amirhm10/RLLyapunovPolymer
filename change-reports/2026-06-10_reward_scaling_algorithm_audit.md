# Reward, Scaling, And Online Algorithm Audit

Date: 2026-06-10

## Summary

Audited the new disturbance-only online TD3, Direct LMPC, and OF-MPC runners for controller objective weights, reward-shaping fallback penalties, safety-gate routing, min-max scaling, and online phase behavior. Added a standalone Markdown report documenting the algorithm and the main tuning knobs for warmup, behavior cloning, handoff, and exploration.

## Code Changes

- Updated `utils/online_disturbance_runner.py` so fallback reward penalties are enabled only for safety-gate runners.
- No-safety-gate runners and MPC-only baselines now store reward configs with `gamma_fallback=0` and `fallback_event_penalty=0`.
- Added explicit run-config fields:
  - `reward_fallback_penalty_enabled`
  - `reward_fallback_penalty_activation_rule`

## Report

Added:

- `report/online_disturbance_runner_algorithm_audit_2026-06-10.md`

The report documents:

- controller objective weights versus RL reward weights
- safety-gate and no-gate reward behavior
- scaled-deviation and TD3 action coordinates
- numerical input and setpoint range audit
- Direct LMPC and OF-MPC objective structure
- warmup, BC, handoff, replay storage, and exploration behavior
- practical tuning knobs for the next experiment

## Validation

- `python -m py_compile utils/online_disturbance_runner.py` passed.
- Smoke checks were run for:
  - `python OnlineTD3_ColdStart_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots`
  - `python OnlineTD3_ColdStart_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots`
- Config checks confirmed:
  - safety-gate reward config keeps fallback penalties enabled
  - no-gate reward config disables fallback penalties
  - MPC/LMPC objective weights remain `Q=[5,1]`, `R/Rdu=[1,1]`
- Smoke behavior confirmed:
  - no-gate fallback penalty sum was `0.0`
  - safety-gate fallback penalty was active when the gate changed the action
