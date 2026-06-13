# Pretrained Handoff Calibration Analysis

## Summary

Extended the pretrained online TD3 critic-reset report with the four newest pretrained disturbance runs using the calibrated handoff schedule. The report now compares the new batch against the previous critic-reset batch, the no-reset low-noise batch, and the older moderate-noise batch.

## Data Added

Newest analyzed runs:

- `results/OnlineTD3_LMPCPretrained_SafetyGate/20260612_205458`
- `results/OnlineTD3_LMPCPretrained_NoSafetyGate/20260612_205455`
- `results/OnlineTD3_OFMPCPretrained_SafetyGate/20260612_205504`
- `results/OnlineTD3_OFMPCPretrained_NoSafetyGate/20260612_205501`

These runs used critic reset, BC std `1e-4`, pretrained `lyap_eps=1e-2`, and a 10-episode handoff with `handoff_update_mode="critic_td_plus_actor_bc"`.

## Main Finding

The calibrated handoff removed the severe OF-MPC handoff collapse seen in the previous critic-reset batch. The worst transient moved to episode 31, the first full-RL episode after handoff, and is far smaller than the old episode-23 failure.

## Artifacts Updated

- `analysis/online_pretrained_critic_reset_analysis.py`
- `report/online_pretrained_critic_reset_analysis_2026-06-12.md`
- New handoff-calibrated CSV tables under `report/tables/2026-06-12_online_pretrained_critic_reset_analysis/`
- New handoff-calibrated figures under `report/figures/2026-06-12_online_pretrained_critic_reset_analysis/`

## Validation

Regenerated the report with:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe analysis\online_pretrained_critic_reset_analysis.py
```

The updated report embeds the new figures and records the caveat that the latest batch changes both handoff logic and Lyapunov epsilon, so it is not a pure handoff-only ablation.

