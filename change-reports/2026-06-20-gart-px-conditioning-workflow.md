# GART Px Conditioning Workflow Note

## Summary

Updated the GART stability workflow report with an explicit Riccati Lyapunov matrix design and conditioning check before the contraction-admissible target step.

## Scientific Change

- Documented that $P_x$ is computed from the physical-state block of the output-disturbance augmented model.
- Added the DARE used to compute $P_x$ and the local terminal feedback $K_x$.
- Recorded the current saved-model diagnostics:
  - $\lambda_{\min}(P_x)=4.10\times 10^{-5}$.
  - $\lambda_{\max}(P_x)=5.01$.
  - $\kappa_2(P_x)=1.22\times 10^5$.
  - relative Riccati residual $9.65\times 10^{-14}$.
  - closed-loop spectral radius $0.946$.
  - controllability rank $7/7$.
- Added the interpretation that the current method should be retained because the Riccati audit passes, while performance changes should first be explored through weights, target-motion limits, governor settings, and exploration scale.

## Files Changed

- `report/gart_stability_workflow.md`

## Validation

- Ran `git diff --check -- report/gart_stability_workflow.md`.
- No code files were changed, so Python compilation was not required.
