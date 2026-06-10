# Record Reward And Objective Conventions

Date: 2026-06-09

## Summary

Updated `AGENTS.md` so future runners and scripts preserve the intended separation between controller objectives, offline pretraining rewards, and online RL shaped rewards.

## Convention Added

- MPC and Direct LMPC optimization objectives use:
  - `Q = [5, 1]`
  - `R` / `Rdu = [1, 1]`

- Offline TD3 pretraining rewards use the one-step MPC quadratic stage cost with:
  - `Q = [5, 1]`
  - `R = [1, 1]`

- Online RL training and evaluation may use the shaped reward family with separate reward weights:
  - `Q_reward = [12, 6]`
  - `R_reward = [1, 1]`
  - fallback, event, and bonus terms as configured by the online runner

The shaped online reward parameters must not overwrite MPC, OF-MPC, LMPC, target-selector, or safety-gate objective weights.

## Validation

Documentation-only update. Reviewed `AGENTS.md` for placement under `Core Conventions`.
