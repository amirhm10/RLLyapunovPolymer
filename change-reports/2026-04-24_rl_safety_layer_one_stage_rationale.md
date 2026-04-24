# RL Safety Layer One-Stage Rationale

## Summary

Extended `report/direct_lyapunov_nominal_oscillation_analysis.md` to explain
why the proposed one-stage artificial-target MPC is better aligned with the
final RL-with-safety-layer project than the current two-stage target-selector
plus safety-filter architecture.

## Changes

- Added a dedicated section explaining that the one-stage formulation is not
  intended to replace RL; it should act as the certificate-producing safety
  layer around an RL action.
- Compared the current two-stage structure:
  - external target selector,
  - RL/MPC candidate action,
  - safety filter around the preselected target,
  against a one-stage safety projection that jointly chooses `u_safe` and the
  artificial target.
- Added the key argument that target selection and action correction are
  competing projections in the two-stage method, while the one-stage method can
  trade off target admissibility, Lyapunov contraction, and minimal
  intervention on `u_RL` in one optimization.
- Updated the engineering plan so the artificial-target variant is explicitly
  framed as an RL safety-layer projection.
- Added recommended RL safety-layer metrics:
  - accepted RL action rate,
  - intervention size,
  - Lyapunov margin,
  - slack activity,
  - artificial target motion,
  - output RMSE and reward.

## Validation

- Checked existing figure links still resolve.
- Ran `git diff --check` on the updated report.
