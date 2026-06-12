# Online Low-Noise Regression Analysis

Date: 2026-06-12

## Summary

Added a reproducible analysis of the online TD3 low-noise BC/handoff batch
against the prior bounded-mixed online batch.

## Artifacts

- Analysis script: `analysis/online_low_noise_regression_analysis.py`
- Report: `report/online_low_noise_regression_analysis_2026-06-12.md`
- Figures: `report/figures/2026-06-12_online_low_noise_regression_analysis/`
- Tables: `report/tables/2026-06-12_online_low_noise_regression_analysis/`

## Main Finding

The low-noise change is not uniformly bad:

- pretrained runs regress badly during handoff and early full RL
- cold-start runs improve because the old `0.1` BC noise was too aggressive
- LMPC-pretrained comparisons are confounded by a checkpoint change

The recommended next schedule is split by family:

- pretrained: restore moderate BC/handoff exploration
- cold-start: keep low/tiny BC exploration

## Validation

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile analysis\online_low_noise_regression_analysis.py
```

The analysis script was also run end-to-end to regenerate the Markdown report,
tables, and figures.
