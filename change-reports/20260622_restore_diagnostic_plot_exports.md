# Restore Diagnostic Plot Exports For Compact Runs

## Objective

Bring back the visually helpful diagnostic plots for two-phase runs while keeping the compact data export profile. The compact profile should reduce saved table/array size, but it should not hide exploratory plots needed to understand interventions, safety activity, and full-sample behavior.

## Changes

- Updated TD3 safety debug artifact saving so compact runs call the full diagnostic plotting path.
- Updated direct/GART artifact saving so compact runs also use the full diagnostic plotting path.
- Data export remains compact. Only plot generation behavior changed.

## Expected TD3 Plot Outputs

New compact TD3 runs will again include diagnostic plots such as:

- `reward_average_summary.png`
- `activation_contraction_counts.png`
- `qcqp_status.png`
- `correction_modes.png`
- `episode_samples_by_tens/...`
- `last_episode_summary/...`
- phase-window plots under `ph/`

These are especially useful for seeing whether a no-gate run would have activated the gate, whether an active gate actually intervened, and where rewards or setpoint transitions became strange.

## Validation

- `python -X pycache_prefix=... -m py_compile Lyapunov/safety_debug.py Lyapunov/direct_lyapunov_mpc.py`
