# 2026-03-30 First-Step Replacement Notebooks

## Why

The existing safety-filter notebooks were built around QCQP projection and optional fallback MPC. The new desired experiment is different:

- keep the same refined selector and effective-target logic
- keep the same safety-style debug/export workflow
- remove QCQP projection
- replace projection with a first-step-contraction MPC solve
- do not use a fallback controller
- if the constrained replacement solve fails, still apply the original candidate and log that explicitly

## What Changed

### New notebooks

- Added `LyapunovFirstStepContractionMPC.ipynb`
- Added `LyapunovFirstStepContractionRL.ipynb`

Both notebooks were copied from the current safety notebooks and then retargeted to the first-step-replacement path so their structure, config layout, and export flow stay aligned.

### Shared replacement logic

- Added shared first-step replacement helper logic in `Lyapunov/upstream_controllers.py`

This helper now handles:

- candidate hard Lyapunov first-step check
- constrained MPC activation when the candidate violates the bound
- constrained action application when the replacement solve succeeds
- original-candidate application when the replacement solve fails

### MPC path

- Updated `Simulation/run_mpc_first_step_contraction.py`

The runner now:

- computes the ordinary upstream MPC candidate first
- hard-checks it against the first-step Lyapunov bound
- triggers constrained MPC only on violation
- no longer calls fallback MPC
- logs constrained-solve failure while still applying the original candidate

### RL path

- Updated `Simulation/run_rl_lyapunov.py`

Added a new backend:

- `projection_backend="first_step_contraction_mpc"`

This backend:

- keeps the RL action as the candidate
- uses the same refined selector and effective target as the safety path
- activates constrained MPC only when the RL candidate violates the first-step bound
- applies the RL candidate if the constrained replacement solve fails

Legacy backends remain unchanged:

- `legacy_augstate`
- `safety_filter`

### Debug/export

- Updated `Lyapunov/safety_debug.py`

Added replacement-specific fields and plots:

- `candidate_first_step_lyap_ok`
- `first_step_contraction_triggered`
- `constrained_mpc_attempted`
- `constrained_mpc_solved`
- `constrained_mpc_applied`
- `constrained_mpc_failed_applied_candidate`
- candidate/applied first-step values and margins

The debug exporter now detects first-step-replacement runs and changes the status plots accordingly instead of showing inactive QCQP/fallback semantics.

### Method note

- Added `report/lyapunov_first_step_contraction_replacement.md`

This note documents the exact control law used by the new notebooks and explains the “no fallback, apply candidate if constrained replacement is infeasible” rule.

## Validation

Validated with:

- `python -m py_compile Lyapunov/upstream_controllers.py Simulation/run_mpc_first_step_contraction.py Simulation/run_rl_lyapunov.py Lyapunov/safety_debug.py`
- JSON parsing for:
  - `LyapunovFirstStepContractionMPC.ipynb`
  - `LyapunovFirstStepContractionRL.ipynb`

## Notes

- Existing safety notebooks were left unchanged.
- Existing local untracked docs/data were not included in this change.
- Existing user-edited notebook parameter values in the source safety notebooks were preserved when creating the new notebooks; I did not reset them to older defaults.
