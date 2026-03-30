# Refined Selector Report Refresh

## Why
- The active selector path is now the single refined Step A selector.
- The `report/` directory still contained several older selector notes, and the current method note was too short to serve as the canonical paper-style reference.
- The root directory also contains several planning Markdown files that look obsolete, but those deletions need explicit approval first.

## What Changed
- Rewrote `report/refined_step_a_selector_method.tex` into a fuller paper-style document covering:
  - augmented offset-free model and coordinate conventions
  - runtime selector inputs
  - controlled-output map and reference definition
  - exact refined Step A optimization problem
  - weight construction and default hierarchy
  - convexity and solver behavior
  - returned diagnostics
  - interaction with the safe-MPC filter
  - practical interpretation of `y_s`, `C x_s`, and `C_d d_s`
  - tuning implications and scope limitations
- Rebuilt `report/refined_step_a_selector_method.pdf`.
- Added `report/README.md` to mark the current canonical selector report and separate it from legacy reports.

## What Was Not Removed Yet
- No root Markdown files were deleted yet.
- No legacy report files were deleted yet.
- No notebooks or `.py` files were removed.

## Planned Cleanup Pending Approval
Likely root Markdown removal candidates:
- `codex_plan_lyapunov_safety_filter.md`
- `codex_plan_refined_step_a_selector.md`
- `codex_plan_refined_target_selector.md`
- `target_selector_four_modes_codex_plan.md`
- possibly `LyapunovCoreMapping.md`
- possibly `LyapunovHyperparameters.md`
- possibly `SafeMPCFilterProcedure.md`

Likely report-directory legacy candidates:
- `report/current_target_selector_implementation.tex`
- `report/current_target_selector_implementation.pdf`
- `report/target_selector_report.tex`
- `report/target_selector_report.pdf`
- `report/target_selector_four_modes_report.tex`
- `report/target_selector_four_modes_report.pdf`
- possibly `report/lyapunov_safety_filter_report.tex`
- possibly `report/lyapunov_safety_filter_report.pdf`

Those removals should be handled in a separate commit after explicit approval.
