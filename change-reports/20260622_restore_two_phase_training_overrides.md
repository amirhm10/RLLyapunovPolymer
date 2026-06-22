# Restore Two-Phase Training Phase Overrides

Date: 2026-06-22

## Summary

Restored the old online runner training-phase behavior inside the new per-method two-phase runners. The two-phase runner now preserves method-specific noisy GART-LMPC teacher critic warmup and handoff settings, then adds the new global exploration decay ending at the Phase-1 boundary.

## Why

The first two-phase implementation only passed the Phase-1 exploration decay override. That unintentionally fell back to the generic online TD3 phase defaults:

- 20 behavior-clone teacher episodes
- `critic_td_plus_actor_bc`
- actor BC active during the teacher phase

The archived online runners used `default_noisy_teacher_critic_warmup_overrides(...)`, which gives:

- 10 noisy GART-LMPC teacher episodes
- `critic_td_only`
- no actor BC in the teacher phase
- 10 handoff episodes with TD3 full updates

This difference can change the early reward behavior, so it was restored before interpreting the experiment.

## Tuning Change

Changed only behavior exploration starts:

- pretrained TD3 runners: `0.02 -> 0.04`
- cold-start TD3 runners: `0.10 -> 0.15`

Target-policy smoothing and clips were left unchanged:

- pretrained smoothing/clip: `0.02 / 0.04`
- cold-start smoothing/clip: `0.10 / 0.20`

## Files Changed

- `RunOnlineTD3TwoPhaseStudy.py`
- `RunTwoPhase_OFMPCPretrained_SafetyGate.py`
- `RunTwoPhase_OFMPCPretrained_NoSafetyGate.py`
- `RunTwoPhase_ColdStart_SafetyGate.py`
- `RunTwoPhase_ColdStart_NoSafetyGate.py`

## Validation

- Passed `py_compile` on the shared two-phase runner and all method runners.
- Passed a tiny OF-MPC-pretrained safety-gate smoke run.
- Passed a tiny cold-start no-safety-gate smoke run.
- Confirmed saved config now records `behavior_clone_teacher_episodes = 10`, `bc_update_mode = critic_td_only`, and pretrained `exploration_std_start = 0.04`.
