# GART Disturbance Results Report

Date: 2026-06-14

## Summary

Added a disturbance-mode analysis report for the latest saved GART-LMPC run:

- `results/GARTLMPC/20260614_003718`

The report documents:

- exact run setup and disturbance values
- GART target-selector and MPC parameter values from saved configs
- performance, feasibility, target, governor, and disturbance-estimator diagnostics
- why `gart_target_mixed_objective` and `gart_target_mixed_soft` performed poorly
- why `gart_target_raw_objective` remains the working GART candidate

## Main Conclusion

In the disturbance run, the mixed methods failed as performance controllers even though solver success and Lyapunov contraction rates were 100%. The failure mechanism is the mixed objective's attraction to a conservative or held GART target, not numerical infeasibility.

## Files Added

- `report/gart_lmpc_disturbance_results_2026-06-14.md`

## Validation

This is a Markdown-only report update. Validation consisted of checking that the referenced result artifacts and figures exist, and running `git diff --check`.
