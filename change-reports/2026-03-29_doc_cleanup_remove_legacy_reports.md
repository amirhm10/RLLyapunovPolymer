# Documentation Cleanup: Remove Obsolete Root Notes And Legacy Reports

## Why
- The repository had accumulated several planning notes and legacy report files that were no longer part of the active workflow.
- The current canonical selector documentation now lives in:
  - `report/refined_step_a_selector_method.tex`
  - `report/refined_step_a_selector_method.pdf`
  - `report/refined_step_a_selector_parameters.md`
- Keeping obsolete notes in the root and legacy reports in `report/` made the project harder to navigate.

## What Was Removed
### Root Markdown Files
- `codex_plan_lyapunov_safety_filter.md`
- `codex_plan_refined_step_a_selector.md`
- `codex_plan_refined_target_selector.md`
- `target_selector_four_modes_codex_plan.md`
- `LyapunovCoreMapping.md`
- `LyapunovHyperparameters.md`
- `SafeMPCFilterProcedure.md`

### Legacy Report Files
- `report/current_target_selector_implementation.tex`
- `report/current_target_selector_implementation.pdf`
- `report/target_selector_report.tex`
- `report/target_selector_report.pdf`
- `report/target_selector_four_modes_report.tex`
- `report/target_selector_four_modes_report.pdf`
- `report/lyapunov_safety_filter_report.tex`
- `report/lyapunov_safety_filter_report.pdf`

## What Was Kept
- Old notebooks were kept untouched.
- Python files were kept untouched.
- The canonical refined-selector report remains in `report/`.
- Historical context remains available through:
  - `change-reports/`
  - `change-reports/snapshots/`
  - Git history

## Additional Cleanup
- Updated `report/README.md` so it no longer points to removed legacy reports and instead points users to the canonical selector report plus Git/change-report history.
