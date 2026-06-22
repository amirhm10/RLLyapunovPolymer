# Per-Method Two-Phase Study Runners

Date: 2026-06-22

## Summary

Added separate root entrypoints for each two-phase online study method so methods can be launched independently and in parallel while sharing the same implementation in `RunOnlineTD3TwoPhaseStudy.py`.

New entrypoints:

- `RunTwoPhase_OFMPCPretrained_SafetyGate.py`
- `RunTwoPhase_OFMPCPretrained_NoSafetyGate.py`
- `RunTwoPhase_ColdStart_NoSafetyGate.py`
- `RunTwoPhase_ColdStart_SafetyGate.py`
- `RunTwoPhase_GART_LMPC.py`

## Implementation Notes

- Added `main_for_method(...)` and `main_for_methods(...)` helpers to the shared two-phase runner.
- Each wrapper forces its own method and removes any conflicting `--methods` argument from the wrapper CLI.
- Unless the caller supplies `--timestamp`, each wrapper adds a method-specific timestamp suffix. This keeps simultaneously launched wrappers from writing into the same batch folder.
- Added traceback capture to per-method failure manifests and `error.json` files so long multi-seed runs retain enough information to debug export or runtime failures.

## Archived Entrypoints

Moved these legacy Lyapunov/GART scripts under `archive/lyapunov_legacy_runners_20260622/`:

- `ComparePretrainedTD3LyapunovMPC.py`
- `DirectLyapunovMPC.py`
- `DirectLyapunovMPC_DisturbanceRunner.py`
- `GARTLyapunovMPC_ExplorationProbe.py`
- `PretrainTD3LyapunovMPC.py`

## Validation

- Passed `py_compile` for the shared runner and all five new wrappers.
- Passed a tiny TD3 wrapper smoke run:
  `RunTwoPhase_ColdStart_NoSafetyGate.py --n-seeds 1 --phase1-episodes 1 --phase2-episodes 1 --set-points-len 5 --no-save-plots`
- Passed a tiny GART wrapper smoke run:
  `RunTwoPhase_GART_LMPC.py --n-seeds 1 --phase1-episodes 1 --phase2-episodes 1 --set-points-len 5 --no-save-plots`
