# GART Post-Patch Disturbance Run Report

Date: 2026-06-14

## Summary

Added a detailed analysis report for the latest post-correctness-patch GART-LMPC disturbance closed-loop run:

- `results/GARTLMPC/20260614_124954`
- companion target-only run `results/GARTTargetSelectorStudy/20260614_124509`

The report documents that the enabled raw GART controller completed all 4000 disturbed closed-loop steps with optimal solver status, hard Lyapunov contraction on every step, and tracking performance essentially identical to the old governed-reference baseline.

## Key Findings Documented

- `gart_target_raw_objective` is solver-clean and stable in this run.
- Mixed objective cases were not run because the GART target is still too conservative to use as a performance target.
- The GART target selector returned `hold_previous` on 3584 of 4000 steps.
- `y_s` changed on only 415 of 3999 target transitions.
- Target mismatch remains high: mean infinity-norm mismatch is `2.966007`, max is `4.953267`.
- `92.225%` of targets were classified unreachable relative to the raw setpoint tolerance.
- The apparent constant `y_s` behavior is mainly due to governor hold and stage-2 smoothing, not an LMPC `u_mid` penalty.

## Files Changed

- Added `report/gart_lmpc_post_patch_disturbance_run_2026-06-14.md`.
- Kept the root GART runner configured for the user's closed-loop disturbance test workflow.

## Validation

This report is based on local artifacts from:

- `results/GARTLMPC/20260614_124954/summary.json`
- `results/GARTLMPC/20260614_124954/gart_target_raw_objective/config.json`
- `results/GARTLMPC/20260614_124954/gart_target_raw_objective/steps.csv`
- `results/GARTLMPC/20260614_124954/gart_target_raw_objective/payload.pickle`
- generated direct-style plots under the latest result directory

