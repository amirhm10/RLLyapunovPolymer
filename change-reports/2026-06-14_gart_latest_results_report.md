# GART Latest Results Report

Date: 2026-06-14

## Summary

Added a scientific analysis report for the latest saved GART-LMPC run:

- Closed-loop artifacts: `results/GARTLMPC/20260613_235051`
- Target-only artifacts: `results/GARTTargetSelectorStudy/20260613_235051`

The report concludes that:

- `old_governed_reference` and `gart_target_raw_objective` did not fail.
- `gart_target_mixed_objective` and `gart_target_mixed_soft` failed.
- The failure mechanism is the mixed objective's target-centered pull toward a conservative or held GART target, not the raw GART target-selector path itself.

## Files Added

- `report/gart_lmpc_latest_results_2026-06-14.md`

## Validation

This is a Markdown-only report update. Validation consisted of checking that the referenced result artifacts and figures exist in the saved run directories.
