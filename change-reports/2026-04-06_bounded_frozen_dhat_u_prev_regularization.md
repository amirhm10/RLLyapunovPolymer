# 2026-04-06 Bounded Frozen-dhat `u_prev` Regularization

## Summary

Updated the bounded frozen-`dhat` target method to add the requested `\|u_s - u_{\mathrm{ref}}\|^2` regularization term, with `u_ref = u_prev`, and exposed the weight directly in the first-step-contraction notebook.

## What Changed

- `analysis/steady_state_debug_analysis.py`
  - added `u_ref_weight` to the analysis config defaults
  - extended `solve_bounded_steady_state_least_squares(...)` with:
    - `u_ref`
    - `u_ref_weight`
  - implemented the regularized bounded least-squares solve by augmenting the linear least-squares system with weighted identity rows
  - passed the previous applied input sequence into the box-analysis path so the notebook sidecar uses the same bounded objective shape as the controller target helper

- `Lyapunov/frozen_dhat_target.py`
  - added `u_ref_weight` to the frozen target config defaults
  - passed `u_applied_k` as the bounded target reference input
  - reported the regularization term through `objective_terms["u_applied_anchor"]`
  - populated `R_u_ref_diag_used` and related debug fields for this mode

- `Lyapunov/safety_debug.py`
  - relabeled the paper-plot target traces from generic `xs_` / `ds_` to explicit effective-target labels `x_s_eff_` / `d_s_eff_`

- `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`
  - added `u_ref_weight = 1.0` as a notebook-level knob
  - passed that value into `frozen_target_config`
  - passed the same value into the steady-state analysis sidecar so the inline plots are produced from the same bounded objective definition

## Notes On The Plot Check

I checked the latest saved first-step export against the latest sidecar bundle before changing the objective. The controller-exported bounded `x_s` matched the sidecar bounded `x_s` numerically. So there was not a separate hidden `x_s` series being plotted in the paper plots. The more important fix was to align the bounded target definition itself and make the effective-target labeling explicit.

## Validation

Ran:

- `python -m py_compile analysis\steady_state_debug_analysis.py Lyapunov\frozen_dhat_target.py Lyapunov\safety_debug.py Simulation\run_mpc_first_step_contraction.py`
- notebook JSON parse for `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`
- synthetic weighted solve check in `rl-env` verifying:
  - the bounded LS solution moves toward `u_ref` when `u_ref_weight > 0`
  - the bounded frozen-`dhat` helper returns the same regularized bounded target and reports the anchor penalty term

## Notes

- This change does not touch the original refined-selector notebook or default first-step runner mode.
- Existing unrelated worktree changes were left untouched.
