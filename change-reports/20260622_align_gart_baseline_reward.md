# Align GART Baseline Reward With Online TD3

Date: 2026-06-22

## Summary

Aligned the two-phase GART-LMPC-only baseline reward with the active online TD3 reward definition used for fair comparison.

## Why

`RunTwoPhase_GART_LMPC.py` already used the GART closed-loop implementation from `experiments/run_gart_target_selector_study.py`, but that implementation still used older reward shaping constants:

- `Q = [5, 1]`
- wider relative bands
- older bonus/inside-band terms

The online TD3 runners use the active comparison reward:

- `Q_reward = [12, 6]`
- `R_reward = [1, 1]`
- tighter relative bands
- quadratic bonus
- fallback penalties disabled for comparison

## Changes

- Added `QY_REWARD_DIAG = [12, 6]` and `RDU_REWARD_DIAG = [1, 1]` to the GART study module.
- Updated the GART closed-loop reward function to match `utils.online_disturbance_runner._build_reward(...)` with fallback penalties disabled.
- Updated the GART console print to include `avg. reward_no_penalty`.
- Added reward settings to the GART direct-style bundle config.

## Validation

- Passed `py_compile` for `experiments/run_gart_target_selector_study.py`, `RunOnlineTD3TwoPhaseStudy.py`, and `RunTwoPhase_GART_LMPC.py`.
- Passed a tiny two-phase GART-LMPC smoke run.
- Confirmed console output prints both `avg. reward` and `avg. reward_no_penalty`.
- Confirmed saved summary has identical `reward_mean`, `reward_base_mean`, and `reward_no_penalty_mean`.
