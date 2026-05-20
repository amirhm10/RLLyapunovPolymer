# Stricter Reward Defaults

## Summary

Updated the cold-start and pretrained direct Lyapunov safety-gate RL scripts to use the stricter reward setup proposed in the latest reward diagnosis report.

## Changes

- Set `Qy_diag = [8.0, 6.0]`.
- Set `gamma_in = 3.0`.
- Set `lam_in = 3.0`.
- Set `beta = 1.0`.
- Set `gamma_fallback = 3.0`.
- Set `fallback_event_penalty = 2.0`.
- Set `maintenance_move_weight = 0.2`.
- Set `jitter_weight = 0.05`.

## Notes

`Qy_diag` is shared by the reward and the direct LMPC safety-gate teacher/fallback setup in the current scripts, so the safety-gate controller is also more temperature-weighted in the next run.

## Validation

- Run `python -m py_compile DirectLyapunovSafetyGateRL_ColdStart.py DirectLyapunovSafetyGateRL_Pretrained.py`.
- Do not run full training as part of this config edit.
