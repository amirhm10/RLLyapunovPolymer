# 2026-05-11 Pretrained RL Warm-Start Policy

## What changed

Updated `DirectLyapunovSafetyGateRL_Pretrained.ipynb` so the warm-start phase uses the pretrained RL policy instead of `direct_lyapunov_mpc`.

## Why

The pretrained notebook should exploit the available pretrained actor during warm start rather than solving LMPC at each step. That keeps the warm-start phase aligned with the intended pretrained-policy workflow.

## Resulting phase behavior

- Warm start: `policy`
- Warm-start noise: `none`
- BC: teacher data, no behavior noise
- Full RL: parameter-noise behavior exploration

## Validation

- Notebook JSON remained valid after the edit.
- The phase-config cell now shows `\"warmup_behavior_source\": \"policy\"`.
