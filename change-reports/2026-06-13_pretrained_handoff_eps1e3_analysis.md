# Pretrained Handoff Epsilon 1e-3 Analysis

Date: 2026-06-13

## Summary

Extended the pretrained online TD3 critic-reset report with a current-only
analysis of the four final pretrained disturbance runs using:

- critic reset,
- pretrained actor loading,
- BC exploration std `1e-4`,
- 10-episode actor-frozen handoff,
- bounded-mixed Direct LMPC target selector,
- restored `lyap_eps=1e-3`.

## Main Finding

The handoff catastrophe is fixed for the final setup. The report now excludes
the older noise, no-reset, short-handoff, and relaxed-epsilon batches, and only
reports the final `lyap_eps=1e-3` critic-reset plus calibrated-handoff batch.

In the strict-epsilon batch, LMPC-pretrained no-gate has the best tail
reward/RMSE among the four single-seed runs. The safety-gate runners remain
more conservative and pay a small tracking/reward cost, but stay controlled.

## Files Changed

- `analysis/online_pretrained_critic_reset_analysis.py`
- `report/online_pretrained_critic_reset_analysis_2026-06-12.md`
- `report/tables/2026-06-12_online_pretrained_critic_reset_analysis/pretrained_handoff_eps1e3_current_*.csv`
- `report/figures/2026-06-12_online_pretrained_critic_reset_analysis/pretrained_handoff_eps1e3_*.png`

## Validation

Ran:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe analysis\online_pretrained_critic_reset_analysis.py
python -m py_compile analysis\online_pretrained_critic_reset_analysis.py
```
