# Notebook-Style Saved-Agent Evaluation Refactor

## Summary

Refactored the saved-agent evaluation entrypoint so the root script reads like the other converted notebook-style experiment files, while reusable implementation logic remains in `Simulation/`.

## Changes

- `DirectLyapunovSavedAgentEvaluation.py`
  - Now contains the visible experiment setup:
    - user-editable agent path and dry-run switches
    - plant constants and steady-state setup
    - five-episode fixed disturbance scenario setup
    - scaling, observer, TD3, reward, and solver setup
    - explicit calls for cold saved RL, pretrained saved RL, MPC-only diagnostics, and direct LMPC
    - comparison CSV, plot, and JSON summary export
  - Contains no reusable function or class definitions.

- `Simulation/saved_agent_evaluation.py`
  - Now acts as a helper module instead of an import-time experiment script.
  - Defines lightweight configuration/context dataclasses and callable helpers for:
    - agent resolution and loading
    - fixed disturbance profile construction
    - per-controller evaluation runs
    - output/setpoint alignment
    - unified comparison records
    - CSV and plot export
  - No longer builds the plant, reward, observer, solvers, or saved-result folder at import time.

## Preserved Behavior

- Results still save under `results/Compare/<timestamp>/`.
- Latest-agent mode still skips `mpc_only` trained-agent files.
- Saved RL agents still run with the direct Lyapunov safety gate active.
- MPC-only fallback/activation plots still use would-be Lyapunov gate activation.
- Post-step output alignment is preserved for `y_system` length `nFE + 1`.

## Validation

- `python -m py_compile DirectLyapunovSavedAgentEvaluation.py Simulation/saved_agent_evaluation.py`
- `python DirectLyapunovSavedAgentEvaluation.py --dry-run`
- Synthetic plot check for `(4001, 2)` outputs against `(4000, 2)` setpoints.
- `git diff --check`

## Notes

- Full saved-agent evaluation was not rerun during this refactor.
