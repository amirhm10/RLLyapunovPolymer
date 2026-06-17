# Disable Online Section-16 Projection Override

## Objective

Ensure the online TD3 safety-gate runners use direct candidate acceptance followed by GART-LMPC fallback, not the Section-16 QCQP projection path.

## Changes

- Removed Section-16 projection aliases from the high-level online disturbance preset normalizer.
- Kept the accepted active-gate backend as `direct_accept_or_fallback`.
- Left `mpc_only_diagnostic` available for no-gate diagnostics.
- Updated the GART gate analysis report to distinguish:
  - tight practical Lyapunov tube effects
  - moving GART target and disturbance-estimate effects
  - RL exploration effects

## Interpretation

The latest saved run with `gart_section16_projected` steps was produced before this change. Future runs through `run_online_td3_disturbance_preset(...)` will reject Section-16 projection backend overrides instead of silently using QCQP projection.

## Validation

Run `python -m py_compile` on `utils/online_disturbance_runner.py` and the updated analysis/report support modules.

