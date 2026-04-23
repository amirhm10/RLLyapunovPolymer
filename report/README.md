# Report Directory

## Canonical Current Documents
- `direct_lyapunov_mpc_frozen_output_disturbance_run_report.md`
  Browsable report for the direct frozen-output-disturbance Lyapunov MPC four-scenario notebook, including the April 23, 2026 diagnostic run interpretation, copied figures, ablation comparison, and literature context.
- `direct_lyapunov_mpc_frozen_output_disturbance.tex`
  LaTeX source for the direct frozen-output-disturbance Lyapunov MPC four-scenario report.
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

The direct Lyapunov rewrite note can be built with:

```powershell
pdflatex -interaction=nonstopmode direct_lyapunov_mpc_frozen_output_disturbance.tex
pdflatex -interaction=nonstopmode direct_lyapunov_mpc_frozen_output_disturbance.tex
```

Run that command from the `report/` directory.
