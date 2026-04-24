# Clarify Unknown Artificial Target Certificate

## Summary

Updated `report/direct_lyapunov_nominal_oscillation_analysis.md` to address
the concern that a fixed-RL-action safety check cannot evaluate Lyapunov
contraction unless an artificial target `x_a` is known.

## Changes

- Clarified that a cheap scalar check is only valid when a trusted previously
  certified artificial target is available.
- Clarified that the real pass-through certificate must solve a feasibility
  problem with:
  - `u_0 = u_RL` fixed;
  - `x_a,u_a` as decision variables.
- Added explicit wording that the certificate asks whether there exists an
  admissible artificial target that certifies `u_RL`.
- Preserved the intervention logic:
  - apply `u_RL` unchanged if the fixed-action certificate succeeds;
  - solve the correction problem with `u_0` free only if that certificate
    fails.

## Validation

- Ran markdown link and whitespace checks for the updated report.
