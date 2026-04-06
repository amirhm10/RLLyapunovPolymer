# 2026-04-06 First-Step-Contraction Bounded Frozen-dhat Target

## Summary

Added a new bounded frozen-`dhat` target-generation path for the first-step-contraction MPC workflow without changing the default refined-selector path. The new path is exposed through a shared helper, an optional runner branch, and a new notebook cloned from the maintained first-step-contraction MPC notebook.

## Files Changed

- `Lyapunov/frozen_dhat_target.py`
- `Simulation/run_mpc_first_step_contraction.py`
- `Lyapunov/safety_debug.py`
- `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`
- `report/first_step_contraction_bounded_frozen_dhat_target.tex`

## Technical Details

### New bounded frozen-`dhat` helper

`Lyapunov/frozen_dhat_target.py` adds `prepare_filter_target_from_bounded_frozen_dhat(...)`.

The helper:

- splits the augmented observer state into `xhat` and `dhat`
- validates the Rawlings output-offset structure used by the notebook
- recovers the unaugmented `A`, `B`, and `C` blocks
- solves the exact frozen-`dhat` steady-state equations in scaled deviation coordinates
- checks the exact target against the input box
- falls back to bounded least squares when the exact target is infeasible
- returns a target-info package shaped to match the existing first-step-contraction debug/export flow

### First-step runner branch

`Simulation/run_mpc_first_step_contraction.py` now accepts:

- `target_generation_mode="refined_selector"` by default
- `frozen_target_config=None`

When `target_generation_mode="bounded_frozen_dhat"`, the runner calls the new helper instead of the refined selector. The rest of the loop is unchanged:

- effective-target reuse
- upstream offset-free MPC candidate solve
- first-step contraction replacement
- debug bundle construction

The per-step info dict now also carries:

- `target_generation_mode`
- `selector_name`
- `effective_selector_name`
- `box_solve_mode`
- `exact_within_bounds`
- `exact_bound_violation_inf`
- `bounded_residual_norm`

### Debug export compatibility

`Lyapunov/safety_debug.py` now recognizes the new target stages:

- `frozen_dhat_exact`
- `frozen_dhat_bounded_fallback`

The safety-filter step records now also export the bounded frozen-`dhat` diagnostics so the new notebook uses the same debug-export path cleanly.

### New notebook

`LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb` is cloned from `LyapunovFirstStepContractionMPC.ipynb` and only changes:

- notebook labeling
- run configuration
- bundle/export naming

It keeps:

- the same plant setup
- the same disturbance scenario
- the same upstream MPC
- the same first-step replacement logic
- the same debug/export tooling

### New math report

`report/first_step_contraction_bounded_frozen_dhat_target.tex` is a self-contained derivation covering:

- scaled deviation coordinates
- Rawlings augmentation split
- frozen-`dhat` steady-state equations
- exact stacked and reduced solves
- bounded least-squares fallback
- projection and KKT interpretation
- mapping into the first-step-contraction target package
- upstream MPC candidate problem
- Lyapunov value and first-step contraction inequality
- nominal versus disturbance interpretation

## Validation

Ran:

- `python -m py_compile Lyapunov\frozen_dhat_target.py Simulation\run_mpc_first_step_contraction.py Lyapunov\safety_debug.py`
- notebook JSON parse for `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`
- synthetic `rl-env` checks for:
  - zero-RHS exact target returning `x_s = 0`, `u_s = 0`, `y_s = 0`
  - out-of-box exact target triggering bounded fallback and respecting bounds
- import-only check in `rl-env` for:
  - `Lyapunov.frozen_dhat_target`
  - `Simulation.run_mpc_first_step_contraction`
  - `Lyapunov.safety_debug`

## Notes

- The original `LyapunovFirstStepContractionMPC.ipynb` was left untouched.
- The default runner behavior remains the refined-selector path.
- Existing unrelated worktree changes were not modified.
