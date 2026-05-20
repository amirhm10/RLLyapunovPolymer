# Latest Strict Reward RL Analysis Report

## Summary

Added a new report analyzing the latest complete cold-start and pretrained direct Lyapunov safety-gated TD3 result folders.

## Changes

- Created `report/rl_latest_strict_reward_cold_pretrain_analysis_2026-05-19.md`.
- Generated supporting figures under `report/figures/2026-05-19_latest_strict_reward_rl_analysis/`.
- Compared RL against matched MPC-only diagnostics for reward, RMSE, final-tail offset, fallback dependence, and input activity.
- Updated the fallback plots and report tables to show MPC-only "would-be fallback" counts using the diagnostic Lyapunov unsafe/contraction-failure flags, since actual fallback is zero by construction for MPC-only.
- Flagged that the latest runs use stricter Lyapunov settings than the earlier `eps=1e-3` discussion:
  cold start uses `rho=0.99`, `eps=1e-6`, while pretrained uses `rho=0.995`, `eps=0.0`.
- Noted that the active local scripts had since been edited toward a matched relaxed-gate rerun with `rho=0.99` and `eps=1e-3`, so those future reruns should be analyzed separately.

## Validation

- Metrics were computed from each case `arrays.npz`, `episode_table.csv`, `summary.json`, and `comparison_table.csv`.
- Figure files were generated from raw saved arrays and checked for existence.
- No raw result files were modified.

## Notes

Generated figures remain under the ignored `report/figures/` path and are not intended to be committed unless explicitly requested.
