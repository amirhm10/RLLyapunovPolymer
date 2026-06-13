# LMPC Target Selector Research Bundle

Date: 2026-06-13

## Summary

Created a shareable research bundle diagnosing why the latest Direct LMPC TD3
pretraining results remain weak even after increasing the actor and critic to
`[512, 512, 512, 512, 512]`.

The bundle compares:

- historical governed-reference LMPC pretraining and comparison,
- bounded-mixed 256x3 LMPC pretraining and comparison,
- latest bounded-mixed 512x5 LMPC pretraining and comparison,
- OF-MPC TD3 pretraining as the positive-control imitation case.

## Main Finding

The evidence points away from a simple network-size or scaler mismatch
explanation. The latest 512x5 LMPC actor reaches a lower supervised BC loss,
but its closed-loop comparison is worse than the 256x3 bounded-mixed actor and
worse than the historical governed-reference actor. The governed-reference
selector also failed to produce a satisfactory LMPC-pretrained policy, so the
bundle frames both selector families as historical evidence for a deeper
target-selector and label-map issue.

The exported scaler table confirms the LMPC pretraining and comparison runs use
the same TD3 scaled-deviation contract. The comparison setpoints are inside the
exported physical setpoint scaler for all included LMPC runs.

## Online Evidence Expansion

Extended the same bundle so the deep-research packet also includes the final
online pretrained disturbance runners and the two disturbance controller
baselines:

- Direct LMPC disturbance baseline,
- OF-MPC disturbance baseline,
- LMPC-pretrained TD3 with and without the safety gate,
- OF-MPC-pretrained TD3 with and without the safety gate.

The added online section reports `reward_no_penalty` for fair controller
comparison, keeps actual safety-gate interventions separate from no-gate
Direct LMPC monitor diagnostics, and embeds reward/RMSE comparison figures.
This makes the packet suitable for asking a broader target-selector question:
why the Direct LMPC selector is weak as an offline label generator while the
online TD3 runners can still outperform the controller baselines under the
online shaped tracking metric.

## Files Added

- `analysis/lmpc_target_selector_research_bundle.py`
- `report/bundles/2026-06-13_lmpc_target_selector_research_bundle/README.md`
- `report/bundles/2026-06-13_lmpc_target_selector_research_bundle/deep_research_prompt.md`
- `report/bundles/2026-06-13_lmpc_target_selector_research_bundle/lmpc_target_selector_research_bundle.html`
- `report/bundles/2026-06-13_lmpc_target_selector_research_bundle/figures/*.png`
- `report/bundles/2026-06-13_lmpc_target_selector_research_bundle/tables/*.csv`

## Validation

Ran:

```powershell
python -m py_compile analysis\lmpc_target_selector_research_bundle.py
```

Also regenerated the bundle with:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe analysis\lmpc_target_selector_research_bundle.py
```
