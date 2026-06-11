# Bounded-Mixed Online Disturbance Analysis

Date: 2026-06-11

## Summary

Added an extended analysis for the eight disturbance-only runners after switching the online Direct LMPC gate/diagnostic selector to the previous bounded mixed target selector.

The analysis compares the latest full bounded-mixed batch against the latest full governed-reference batch under the same runner roots.

## Artifacts

- Analysis script: `analysis/online_disturbance_bounded_mixed_analysis.py`
- Report: `report/online_disturbance_bounded_mixed_8_runner_analysis_2026-06-11.md`
- Figures and CSV tables: `report/figures/2026-06-11_online_disturbance_bounded_mixed_analysis/`

## Main Findings

- The bounded-mixed selector improves the pretrained safety-gate runs relative to governed-reference in `reward_no_penalty` and logged training reward.
- The no-gate control rewards are unchanged across selectors, which is expected because Direct LMPC is diagnostic-only there.
- No-gate diagnostic unsafe rates decrease under bounded mixed, so this bounded selector is less restrictive than the governed-reference monitor for the same executed no-gate actions in this batch.
- Cold-start safety worsens in raw control-performance metrics even though fallback penalty and intervention rate are lower, suggesting a learning-trajectory or fallback-target-quality mechanism rather than a simple penalty-accounting issue.
- Direct LMPC and OF-MPC baselines remain very close under the bounded-mixed selector.

## Validation

Generated the analysis with:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe analysis/online_disturbance_bounded_mixed_analysis.py
```

Static validation passed:

```powershell
python -m py_compile analysis/online_disturbance_bounded_mixed_analysis.py
```

The report explicitly keeps actual online safety-gate interventions separate from Direct LMPC baseline internal controller activity.
