# Report Directory

## Canonical Current Documents
- `refined_step_a_selector_method.tex`
  Main paper-style description of the current selector implementation.
- `refined_step_a_selector_parameters.md`
  Practical tuning guide for the current selector weights and options.

## Legacy / Historical Reports
These files are still present for reference, but they are no longer the primary documentation for the active selector path:
- `current_target_selector_implementation.tex`
- `target_selector_report.tex`
- `target_selector_four_modes_report.tex`
- `lyapunov_safety_filter_report.tex`

Their PDFs are also legacy artifacts unless you explicitly want to keep them as current deliverables.

## Build Notes
The main selector report can be built with:

```powershell
pdflatex -interaction=nonstopmode refined_step_a_selector_method.tex
pdflatex -interaction=nonstopmode refined_step_a_selector_method.tex
```

Run that command from the `report/` directory.
