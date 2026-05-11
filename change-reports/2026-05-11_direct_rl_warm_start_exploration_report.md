# Direct RL Warm-Start Exploration Report

Date: 2026-05-11

## Summary

Added a research note diagnosing the low warm-start reward behavior in the direct Lyapunov safety-gate RL notebooks and connecting the diagnosis to online RL literature.

## Added artifacts

- [report/direct_rl_warm_start_exploration_fix_ideas_2026-05-11.md](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/direct_rl_warm_start_exploration_fix_ideas_2026-05-11.md>)
- [warm_start_reward_and_sigma_2026-05-11.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/warm_start_reward_and_sigma_2026-05-11.png>)

The generated figure is also embedded directly in the report Markdown so the analysis and its visual evidence stay together.

## Main conclusions captured in the report

- The pretrained notebook has a real exploration-design problem during warm start because it uses `warmup_behavior_source="policy"` with very small Gaussian noise and no learning updates.
- Low warm-start reward is not only an exploration issue because the cold-start notebook also shows poor warmup reward while using the direct Lyapunov MPC teacher.
- The current shaped reward is numerically harsh in the early phases, and the direct notebooks are not yet using the proposed `reward_scale=0.01`.
- The strongest next experiment is a combined fix: teacher-seeded warmup, exploration reset at RL start, short-lived BC anchoring into early online RL, and reward scaling.

## Validation

- Confirmed the notebook phase configuration values with `rg`.
- Recomputed phase-average reward, fallback-count, and RMSE statistics from the saved `episode_table.csv` files.
- Generated a diagnostic figure showing episode reward and the effective sigma decay schedule.
