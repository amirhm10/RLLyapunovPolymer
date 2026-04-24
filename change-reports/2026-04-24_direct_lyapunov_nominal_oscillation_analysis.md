# Direct Lyapunov Nominal Oscillation Analysis

## Summary

Added a dedicated diagnostic report explaining the bounded hard/soft
oscillations in the nominal direct Lyapunov MPC run.

## Main Findings

- The large oscillations are not caused by the corrected physical-unit plots.
- Around step 100, the bounded cases are mostly smooth; the stronger events are
  later, especially around steps 220-320 and 450-560.
- The bounded target projection moves sharply while the scheduled setpoint is
  flat because the output-disturbance estimate changes during nonlinear
  nominal transients.
- The bounded target selector has no smoothing or input-reference term in the
  direct wrapper, so active input-bound corner flips are undamped.
- The direct objective tracks `y_sp`, while the Lyapunov contraction is centered
  at `x_s`; when bounded projection places `y_s` far from `y_sp`, those two
  references can fight.

## Artifacts

- `report/direct_lyapunov_nominal_oscillation_analysis.md`
- `report/figures/direct_lyapunov_nominal_oscillation_analysis/*.svg`
- `report/figures/direct_lyapunov_nominal_oscillation_analysis/*.csv`
- `report/figures/direct_lyapunov_nominal_oscillation_analysis/*_table.md`

## Validation

- Recomputed window statistics directly from the nominal export arrays and
  step tables.
- Checked the direct-controller target, objective, observer-update, and plotting
  paths for scaling or sign mismatches.
