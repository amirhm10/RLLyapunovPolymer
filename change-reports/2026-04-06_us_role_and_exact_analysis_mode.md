# 2026-04-06: `u_s` Role Report and Exact-Analysis Mode

## Summary

This change separates online controller targeting from offline steady-state analysis more explicitly.

- The steady-state sidecar now supports a configurable analysis target mode:
  - `exact`
  - `bounded`
  - `hybrid`
- The selected analysis mode is applied consistently to the sidecar plots, mismatch norms, and summary fields through a dedicated analysis-target view.
- The bounded frozen-`dhat` first-step notebook and the offset-free steady-state debug notebook now expose a notebook-level `analysis_target_variant` knob.
- A new LaTeX report documents where `u_s` is and is not used in the first-step-contraction Lyapunov workflow, and why exact-vs-bounded targets answer different questions.

## Technical Details

### Sidecar analysis mode

`analysis/steady_state_debug_analysis.py` now:

- accepts `analysis_target_variant` in `analysis_config`
- normalizes the requested mode to one of:
  - `exact`
  - `bounded`
  - `hybrid`
- builds one analysis-target view containing the selected `x_s`, `u_s`, `y_s`, `d_s`, and reduced-equation traces
- stores the selected view in bundle fields such as:
  - `analysis_x_s`
  - `analysis_u_s_dev`
  - `analysis_y_s_phys`
  - `analysis_xhat_minus_x_s`
  - `analysis_u_applied_minus_u_s`
- updates the main plot exporters to use the selected analysis target rather than the old implicit mixed behavior

The exact and bounded targets are still both preserved in the bundle, so the sidecar continues to support explicit exact-vs-bounded comparison plots.

### Notebook knobs

The following notebooks now expose `analysis_target_variant = "exact"` as an editable cell value:

- `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`
- `MPCOffsetFree_SteadyStateDebug.ipynb`

This changes only the offline analysis view. It does not change the online controller target-generation mode.

### Mathematical report

Added:

- `report/first_step_contraction_us_role_and_analysis_modes.tex`
- `report/first_step_contraction_us_role_and_analysis_modes.pdf`

The report explains:

- exact frozen-`dhat` equilibrium targets
- bounded feasible frozen-`dhat` targets
- why the exact target can be the better unconstrained diagnostic
- where `u_s` enters the Lyapunov-centered prediction and candidate evaluation
- why the current hard first-step contraction inequality is centered directly on `x_s`

## Validation

- `python -m py_compile analysis\steady_state_debug_analysis.py`
- notebook JSON parse:
  - `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`
  - `MPCOffsetFree_SteadyStateDebug.ipynb`
- `rl-env` import check for exact/bounded/hybrid analysis-target selection
- `pdflatex -interaction=nonstopmode -halt-on-error report/first_step_contraction_us_role_and_analysis_modes.tex`

## Notes

- Unrelated pre-existing worktree items were left untouched.
- The online bounded frozen-`dhat` controller path is unchanged by this update.
