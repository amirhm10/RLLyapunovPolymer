# Clarify RL Safety Check Versus Intervention

## Summary

Updated `report/direct_lyapunov_nominal_oscillation_analysis.md` to clarify
that the proposed one-stage RL safety layer should run a safety certificate
every step, but should not modify the RL action every step.

## Changes

- Distinguished between:
  - safety layer enabled as a per-step checker/certifier;
  - safety layer active as an intervention that changes `u_RL`.
- Added the expected pass-through behavior:
  - if `u_RL` is already feasible and Lyapunov-safe, return
    `u_safe = u_RL` up to numerical tolerance.
- Clarified that this is conceptually the same pass-through behavior as the
  current two-stage filter, but with a more consistent certificate because the
  artificial target and Lyapunov center are considered with the action.
- Added a practical two-level workflow:
  - fast certification of `u_RL`;
  - full one-stage correction only if the fast certification fails.

## Validation

- Ran markdown link and whitespace checks for the updated report.
