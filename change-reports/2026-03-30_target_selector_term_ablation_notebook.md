# 2026-03-30 Target Selector Term Ablation Notebook

## Why

We needed a notebook that mirrors the current safety-MPC experiment structure but lets us compare the refined Step A target-selector objective terms directly. The existing safety notebook only runs the full selector objective and does not support exact term-by-term ablations.

## What Changed

- Added an exact `term_activation` mask to the refined selector config in `Lyapunov/target_selector.py`.
- The selector objective now omits masked-off terms exactly instead of relying on tiny or indirect weights.
- Masked-off selector terms are logged as `0.0` in the exported `objective_terms` data so the diagnostics stay consistent with the optimization.
- Added a new notebook:
  - `LyapunovSafetyFilterMPCTargetSelectorTermAblation.ipynb`
- Added a study note:
  - `report/target_selector_term_ablation_study.md`

## Study Notebook Behavior

The new notebook keeps the same safety-MPC backend and config surface as `LyapunovSafetyFilterMPC.ipynb`, and runs this fixed sweep:

- `all_terms_on`
- `objective_zero`
- `only_target_tracking`
- `only_u_applied_anchor`
- `only_u_prev_smoothing`
- `only_x_prev_smoothing`
- `only_xhat_anchor`

For each run it:

- reuses the same base `run_config`
- changes only `target_selector_config["term_activation"]`
- runs the normal safety-MPC controller
- saves the standard safety debug export into a study-specific folder
- appends a row to a comparison table

## Comparison Outputs

The notebook writes:

- one debug-export folder per study case
- a combined `comparison_table.csv`
- compact comparison plots for:
  - mean reward
  - per-output RMSE
  - maximum target-error infinity norm

## Validation

- `python -m py_compile Lyapunov/target_selector.py`
- JSON parse check for `LyapunovSafetyFilterMPCTargetSelectorTermAblation.ipynb`

## Notes

- The existing `LyapunovSafetyFilterMPC.ipynb` notebook was not modified.
- Unrelated local files already present in the worktree were intentionally excluded from this change.
