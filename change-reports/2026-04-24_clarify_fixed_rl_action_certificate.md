# Clarify Fixed RL Action Certificate

## Summary

Updated `report/direct_lyapunov_nominal_oscillation_analysis.md` to address
the concern that the proposed one-stage safety-layer method could become an
intervention at every RL step.

## Changes

- Explicitly stated that the RL-compatible implementation should not always
  solve a free MPC and apply its first input.
- Added the intended two-part safety-layer workflow:
  - first certify the fixed action `u_0 = u_RL`;
  - only if that certificate fails, solve the correction problem with `u_0`
    free.
- Added feasibility-form equations for the fixed-action certificate, including
  the artificial target and Lyapunov decrease condition.
- Clarified that the proposed method means integrated target/action
  certification every step, but integrated correction only when certification
  of the RL action fails.

## Validation

- Ran markdown link and whitespace checks for the updated report.
