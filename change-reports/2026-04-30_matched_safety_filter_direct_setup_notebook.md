# 2026-04-30 Matched Safety-Filter Direct-Setup Notebook

## Why This Change Was Made

The repository already had:

- the focused direct notebook configuration
- the original selector-ablation safety-filter notebook

but it did not yet have a safety-filter notebook that matches the focused direct setup closely enough for cleaner comparison.

The user requested a new notebook that keeps the ablation-method controller architecture while matching the direct-study setup:

- nominal plant mode
- one setpoint only, using the first setpoint
- one episode
- `set_points_len = 2000`
- only `all_terms_on` and `objective_zero`

The user also requested a clearer explanation in the report of why the refined selector and the `objective_zero` architecture behave differently from the direct controller.

## What Was Added

New notebook:

- `LyapunovSafetyFilterMPCTargetSelectorDirectSetup.ipynb`

This notebook is derived from `LyapunovSafetyFilterMPCTargetSelectorTermAblation.ipynb` and now uses:

- `setpoint_y_phys = np.array([[4.5, 324.0]])`
- `n_tests = 1`
- `set_points_len = 2000`
- `mode = "nominal"`
- `augmentation_style = "rawlings"`
- `augmentation_mode = "output_disturbance"`
- `term_studies = [all_terms_on, objective_zero]`

It saves results under:

- `Data/debug_exports/mpc_selector_term_ablation_direct_setup/<timestamp>/`

## Report Update

Updated:

- `report/latest_lyapunov_results_synthesis_2026-04-30.md`

The report now:

- distinguishes between the saved direct run analyzed in the report and the newer direct notebook defaults
- explains more clearly why `objective_zero` does not let the target dominate the closed loop in the same way as the direct controller
- records that a matched follow-up safety-filter notebook now exists for the next comparison

## Output Handling

- Cleared outputs in the new notebook
- Reset execution counts
- Added missing cell IDs so notebook validation is clean

## Validation

- JSON load check passed for `LyapunovSafetyFilterMPCTargetSelectorDirectSetup.ipynb`
- `nbformat.validate` passed for `LyapunovSafetyFilterMPCTargetSelectorDirectSetup.ipynb`

## Notes

- The new notebook was created but not executed in this change batch
- The report still summarizes saved historical runs, not newly generated results from the new notebook
