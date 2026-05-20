# Cold-Start BC Exploration Floor

## Summary

Updated the cold-start direct Lyapunov safety-gate RL script so behavioral cloning uses the same high exploration scale as the cold-start online phase and the online exploration does not decay to zero.

## Changes

- Set cold-start BC exploration to `0.2`.
- Switched cold-start BC behavior noise from `none` to `gaussian`.
- Kept cold-start full-RL exploration start at `0.2`.
- Set cold-start full-RL exploration end to `0.1`.
- Kept cold-start target policy smoothing noise at `0.1`.

## Rationale

Cold start needs enough exploration during BC and online learning to avoid becoming too close to the direct LMPC teacher action too quickly. The nonzero exploration floor should keep policy search active while the safety gate protects execution.

## Validation

- Run `python -m py_compile DirectLyapunovSafetyGateRL_ColdStart.py`.
- Do not run full training as part of this config edit.
