# TD3 Pretraining Full-Scale Analysis Refresh

## Summary

Rewrote the TD3 pretraining analysis report around the June 10, 2026 full-scale OF-MPC and Direct LMPC pretraining artifacts. The new analysis supersedes the earlier June 9 small pilot interpretation.

## Changed

- Rewrote `report/td3_pretraining_pilot_analysis_2026-06-10.md`.
  - Uses `results/PretrainOFMPC/20260610_005048`.
  - Uses `results/PretrainLMPC/20260610_005100`.
  - Uses `results/PretrainOFMPCComparison/20260610_154032`.
  - Uses `results/PretrainLMPCComparison/20260610_173925`.
  - Updates the method reconstruction, objective/reward separation, loss analysis, LMPC label feasibility, rollout metrics, rollout-trace interpretation, risks, and next experiments.
- Added `analysis/td3_pretraining_latest_analysis.py`.
  - Regenerates compact metric tables and report figures from saved JSON, CSV, and pickle artifacts.
  - Converts stored scaled-deviation setpoints back to physical output units before plotting against physical plant outputs.
  - Writes strict JSON summaries with non-applicable diagnostics represented as `null`.
- Added full-scale report figures and source tables under `report/figures/2026-06-10_td3_pretraining_full_scale/`.

## Findings Captured

- OF-MPC-pretrained TD3 is effectively baseline-level on the saved comparison rollouts.
  - Mean RMSE gap versus OF-MPC is +0.23% in nominal mode and +0.70% in disturbance mode.
  - Mean input-movement gap versus OF-MPC is +1.15% in nominal mode and -1.43% in disturbance mode.
- LMPC-pretrained TD3 is feasible but still not controller-quality as a standalone actor.
  - Mean RMSE gap versus Direct LMPC is +27.03% in nominal mode and +44.03% in disturbance mode.
  - Mean input-movement gap versus Direct LMPC is +153.81% in nominal mode and +164.41% in disturbance mode.
- Direct LMPC label generation accepted 2.1M labels from 2.117M attempted candidates, with 99.18% acceptance and 99.56% solve success.
- The LMPC-pretrained actor's main visible failure mode is oscillatory input behavior after downward setpoint transitions, not failed LMPC label generation.

## Validation

Generated figures and compact tables with:

```powershell
python analysis/td3_pretraining_latest_analysis.py
```

Static validation:

```powershell
python -m py_compile analysis/td3_pretraining_latest_analysis.py
```
