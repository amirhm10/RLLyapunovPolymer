# Pretrained BC Exploration Floor

## Summary

Updated the pretrained direct Lyapunov safety-gate RL script so exploration remains active during the behavioral cloning phase and does not decay to zero during online RL.

## Changes

- Set pretrained BC exploration to `0.02`.
- Switched pretrained BC behavior noise from `none` to `gaussian`.
- Kept pretrained full-RL exploration start at `0.02`.
- Set pretrained full-RL exploration end to `0.01`.
- Kept pretrained target policy smoothing noise at `0.01`.

## Rationale

The latest analysis showed pretrained RL tracks well but depends on the safety gate more than cold start. This change gives the pretrained actor more room to adapt away from the old policy prior during BC and keeps online exploration from collapsing to zero.

## Validation

- Run `python -m py_compile DirectLyapunovSafetyGateRL_Pretrained.py`.
- Do not run full training as part of this config edit.
