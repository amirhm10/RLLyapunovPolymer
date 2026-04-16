# Report Directory

## Canonical Current Documents
- `refined_step_a_selector_method.tex`
  Main paper-style description of the current selector implementation.
- `refined_step_a_selector_parameters.md`
  Practical tuning guide for the current selector weights and options.
- `pretraining_rl_controller_review.md`
  Notebook-by-notebook review of the offline MPC-to-TD3 pretraining pipeline in `pretraining_rl_controller.ipynb`.

## Historical Context
Older selector and safety-filter writeups were removed from the active `report/` directory once the refined Step A selector became the canonical path.

If historical context is needed, use:
- `change-reports/`
- `change-reports/snapshots/`
- the Git commit history

## Build Notes
The main selector report can be built with:

```powershell
pdflatex -interaction=nonstopmode refined_step_a_selector_method.tex
pdflatex -interaction=nonstopmode refined_step_a_selector_method.tex
```

Run that command from the `report/` directory.
