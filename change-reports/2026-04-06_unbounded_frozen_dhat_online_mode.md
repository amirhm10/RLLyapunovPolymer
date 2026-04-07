# 2026-04-06: Unbounded Frozen-`dhat` Online Target Mode

## Summary

This change adds an online first-step target-generation mode that always uses the unbounded frozen-`dhat` steady-state target.

- Added `unbounded_frozen_dhat` beside the existing `bounded_frozen_dhat` and `refined_selector` modes.
- Kept the bounded target path intact.
- Updated the first-step notebook so its online target-generation mode now defaults to `unbounded_frozen_dhat`.
- Kept the offline sidecar analysis controls separate from the online target-generation mode.

## Technical Details

### New helper path

`Lyapunov/frozen_dhat_target.py` now exposes:

- `prepare_filter_target_from_unbounded_frozen_dhat(...)`

This helper:

- recovers the unaugmented `A`, `B`, and `C` blocks from the Rawlings augmentation
- freezes `d_s = \hat d_k`
- solves the exact unbounded frozen-`dhat` steady-state equations with `solve_legacy_ss_exact(...)`
- does not project the target back into the input box
- records the exact input-box violation in:
  - `bound_violation_inf`
  - `exact_bound_violation_inf`
  - `exact_within_bounds`

So the controller can now be run around the exact unbounded equilibrium even when that equilibrium is infeasible with respect to the nominal input box.

### Runner support

`Simulation/run_mpc_first_step_contraction.py` now accepts:

- `target_generation_mode="unbounded_frozen_dhat"`

The accepted modes are now:

- `refined_selector`
- `bounded_frozen_dhat`
- `unbounded_frozen_dhat`

Only the target-generation branch changes. The upstream MPC solve, first-step contraction replacement logic, effective-target reuse, and debug export flow remain the same.

### Notebook default

`LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb` now exposes:

- `online_target_generation_mode = "unbounded_frozen_dhat"`

and uses that value in:

- `run_config["target_generation_mode"]`
- debug export source/prefix naming

The notebook still retains the same sidecar analysis section, where the offline analysis target view can be chosen independently.

## Validation

- `python -m py_compile Lyapunov\frozen_dhat_target.py Simulation\run_mpc_first_step_contraction.py`
- notebook JSON parse for `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`
- synthetic `rl-env` check confirming `prepare_filter_target_from_unbounded_frozen_dhat(...)` returns a successful exact target and preserves input-bound violation reporting

## Notes

- This change intentionally does not remove or replace the bounded frozen-`dhat` mode.
- The notebook filename was left unchanged even though its default online mode is now unbounded.
