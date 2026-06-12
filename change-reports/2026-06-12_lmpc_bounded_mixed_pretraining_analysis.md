# Bounded-Mixed LMPC Pretraining Analysis

Date: 2026-06-12

## Summary

Added a reproducible analysis report for the new bounded-mixed Direct LMPC
pretrained TD3 checkpoint and its comparison rollout.

## Artifacts

- Analysis script: `analysis/lmpc_bounded_mixed_pretraining_analysis.py`
- Report: `report/lmpc_bounded_mixed_pretraining_analysis_2026-06-12.md`
- Figures: `report/figures/2026-06-12_lmpc_bounded_mixed_pretraining_analysis/`
- Tables: `report/tables/2026-06-12_lmpc_bounded_mixed_pretraining_analysis/`

## Main Findings

- The new LMPC pretraining/comparison path uses the same TD3 scaler, setpoint
  range, input range, MPC weights, and network architecture as the OF-MPC path.
- The weak LMPC pretrained actor result is not explained by a scaler mismatch.
- Direct LMPC and OF-MPC expert baselines track almost identically in the
  comparison rollout, but LMPC-TD3 remains much farther from its expert than
  OF-TD3 is from OF-MPC.
- The LMPC label map is harder to imitate because bounded target selection,
  contraction checks, and solver post-checks create a more nonlinear accepted
  label surface.

## Validation

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile analysis\lmpc_bounded_mixed_pretraining_analysis.py utils\lmpc_td3_workflow.py utils\online_disturbance_runner.py ComparePretrainedTD3LyapunovMPC.py
```

The analysis script was also run end-to-end with `rl-env` and regenerated the
report, CSV tables, and figures.
