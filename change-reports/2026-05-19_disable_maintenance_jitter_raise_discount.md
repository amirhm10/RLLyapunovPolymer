# Disable Maintenance And Jitter, Raise RL Discount

## Summary

Updated both direct Lyapunov safety-gate RL scripts for the next diagnostic run by disabling maintenance and output-jitter reward penalties while increasing the TD3 discount factor.

## Changes

- Set TD3 `GAMMA = 0.995` in both cold-start and pretrained scripts.
- Set `maintenance_move_weight = 0.0` in both RL reward configs.
- Set `jitter_weight = 0.0` in both RL reward configs.
- Updated the latest analysis report so the documented next-run setup matches the scripts.

## Notes

The Lyapunov contraction factor remains `rho_lyap = 0.99`. Only the TD3 return discount changed from `0.99` to `0.995`.

This diagnostic setup avoids penalizing exploration-induced movement while the new BC and online exploration floors are active.

## Validation

- Run `python -m py_compile DirectLyapunovSafetyGateRL_ColdStart.py DirectLyapunovSafetyGateRL_Pretrained.py`.
- Do not run full training as part of this config edit.
