# TD3 Pretraining Pilot Analysis Report

## Summary

Added a standalone research report for the June 9, 2026 small-sample TD3 pretraining pilots. The report analyzes the OF-MPC-pretrained and LMPC-pretrained TD3 checkpoints, loss histories, label diagnostics, saved-agent settings, and comparison rollout metrics.

## Added

- `report/td3_pretraining_pilot_analysis_2026-06-10.md`
- `report/figures/td3_pretraining_pilot_analysis_2026-06-10/loss_curves.png`
- `report/figures/td3_pretraining_pilot_analysis_2026-06-10/lmpc_label_diagnostics.png`
- `report/figures/td3_pretraining_pilot_analysis_2026-06-10/mean_rmse_bars.png`
- `report/figures/td3_pretraining_pilot_analysis_2026-06-10/reward_bars.png`
- `report/figures/td3_pretraining_pilot_analysis_2026-06-10/mean_abs_du_bars.png`
- `report/figures/td3_pretraining_pilot_analysis_2026-06-10/rollout_outputs_nominal.png`
- `report/figures/td3_pretraining_pilot_analysis_2026-06-10/rollout_outputs_disturb.png`
- supporting normalized CSV summaries in the same figure directory

## Findings Captured

- Both pilot actors learned meaningful expert imitation maps from 160,000 replay labels.
- OF-MPC actor cloning reached lower final imitation loss than LMPC at the same sample count.
- LMPC label generation was feasible with a 94.88% accepted-label rate and 99.59% solve-success rate.
- Both TD3 policies remained worse than their expert baselines in tracking error, reward, and input movement.
- LMPC-pretrained TD3 showed larger transient mismatch, likely from the harder expert action map and insufficient label density rather than label-generation failure.
- The report recommends larger pretraining runs, held-out imitation validation, and online critic recalibration before full safety-gate RL conclusions.

## Validation

- Figures were generated from the saved JSON, CSV, and pickle artifacts under:
  - `results/PretrainOFMPC/20260609_222522`
  - `results/PretrainLMPC/20260609_220058`
  - `results/PretrainOFMPCComparison/20260609_233011`
  - `results/PretrainLMPCComparison/20260609_232747`
- Rollout figures were regenerated after confirming that stored `y_sp` values are scaled deviation coordinates and must be converted to physical output units for plotting against physical plant outputs.
