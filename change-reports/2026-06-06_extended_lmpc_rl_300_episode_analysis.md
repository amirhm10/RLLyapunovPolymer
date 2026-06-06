# Extended LMPC/RL 300-Episode Analysis

Date: 2026-06-06

## Summary

Added an extended report analyzing the latest 300-episode direct LMPC, cold-start safety-gate RL, and pretrained safety-gate RL runs.

## Inputs

Analyzed these result bundles:

- `results/directLyap/20260606_020549`
- `results/ColdStart/20260606_020555`
- `results/Pretrain/20260606_020559`

## Added Artifacts

- `report/extended_lmpc_rl_300_episode_analysis_2026-06-06.md`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/make_figures.py`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/metrics_table.csv`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/metrics_summary.json`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/late_episode_metrics.csv`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/late_episode_metrics.json`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/disturbance_equality_checks.json`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/fig_01_overall_performance.png`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/fig_02_safety_rates.png`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/fig_03_episode_rmse_reward.png`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/fig_04_tail_tracking.png`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/fig_05_contraction_margins.png`
- `report/figures/2026-06-06_lmpc_rl_300_episode_analysis/fig_06_disturbance_profile.png`

## Main Findings

- Direct LMPC achieved 100% hard contraction with nearly the same RMSE as the direct MPC-only diagnostic case.
- Cold-start RL learned substantially after the early phase but remained worse than its same-run no-gate diagnostic case.
- Pretrained RL improved strongly over cold-start RL but still trailed the no-gate diagnostic case and required more safety fallback.
- Disturbance and setpoint schedules match inside each latest bundle.
- The RL `mpc_only` label is scientifically risky because the export behaves as a no-gate diagnostic path inside the RL training loop, not necessarily as a pure standalone offset-free MPC rollout.

## Validation

Generated figures and metrics by running:

```powershell
python report\figures\2026-06-06_lmpc_rl_300_episode_analysis\make_figures.py
```

No controller code was changed.
