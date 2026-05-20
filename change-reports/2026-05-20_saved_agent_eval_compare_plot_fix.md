# Saved-Agent Evaluation Compare Folder And Plot Fix

## Summary

Fixed the saved-agent evaluation workflow after the first full run reached plotting and failed on an output/setpoint length mismatch.

## Changes

- Refactored the root `DirectLyapunovSavedAgentEvaluation.py` into a thin entrypoint.
- Moved the implementation into `Simulation/saved_agent_evaluation.py` so callable experiment logic lives with the rest of the simulation/evaluation code.
- Changed the evaluation study folder from `results/SavedAgentEval/<timestamp>/` to `results/Compare/<timestamp>/`.
- Fixed output tracking plots and summary error metrics to align post-step outputs with the per-control-step setpoint schedule:
  - `y_system` can have `nFE + 1` rows because it includes the initial condition.
  - `y_sp` has `nFE` rows.
  - The comparison now uses `y_system[1:]` when that one-step offset is present.

## Validation

- `python -m py_compile DirectLyapunovSavedAgentEvaluation.py Simulation/saved_agent_evaluation.py`
- `python DirectLyapunovSavedAgentEvaluation.py --dry-run`
- Synthetic plot check with a `(4001, 2)` output array and `(4000, 2)` setpoint array.

## Notes

- The previous run likely completed the controller rollouts and failed only during final comparison plotting.
- Re-running the entrypoint will create a new `results/Compare/<timestamp>/` folder with the corrected plots.
