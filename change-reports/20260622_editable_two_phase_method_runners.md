# Editable Two-Phase Method Runners

Date: 2026-06-22

## Summary

Converted the five two-phase method launchers from thin wrappers into old-style editable runner files. Each runner now exposes the main experiment settings at the top of the file while still calling the shared two-phase implementation.

## Files Updated

- `RunTwoPhase_OFMPCPretrained_SafetyGate.py`
- `RunTwoPhase_OFMPCPretrained_NoSafetyGate.py`
- `RunTwoPhase_ColdStart_NoSafetyGate.py`
- `RunTwoPhase_ColdStart_SafetyGate.py`
- `RunTwoPhase_GART_LMPC.py`
- `RunOnlineTD3TwoPhaseStudy.py`
- `TD3Agent/agent.py`

## Runner Settings Now Visible

Each method runner exposes:

- seed controls: `N_SEEDS`, `SEED_START`, `SEEDS`
- phase lengths: `PHASE1_EPISODES`, `PHASE2_EPISODES`, `SET_POINTS_LEN`
- output controls: `OUTPUT_ROOT`, `TIMESTAMP`, `SAVE_PLOTS`, `EXPORT_PROFILE`
- two-phase setpoint schedule
- nominal disturbance values and phase disturbance multipliers

The four TD3 runners also expose:

- checkpoint path and pretrained critic reset flag
- actor/critic architecture
- replay buffer size and batch size
- actor and critic learning rates
- target-policy smoothing noise and clip
- exploration start/end and handoff noise settings

## Shared Runner Changes

- `RunOnlineTD3TwoPhaseStudy.py` now accepts optional profile fields from runner namespaces, so per-method files can override setpoints and disturbance multipliers directly.
- Pretrained checkpoint resolution is only required for methods that actually load a pretrained actor.
- The pretrained critic reset flag is now controlled by the runner namespace instead of being hard-coded.

## Path-Length Fix

During validation, a tiny pretrained safety-gate smoke run failed while saving the trained TD3 checkpoint on a 265-character Windows path. `TD3Agent.save()` and `TD3Agent.load()` now use Windows extended paths internally while returning/printing the normal path.

## Validation

- Passed `py_compile` for the shared runner, all five method runners, and `TD3Agent/agent.py`.
- Passed a tiny `RunTwoPhase_OFMPCPretrained_SafetyGate.py` smoke run with `PHASE1_EPISODES = 1`, `PHASE2_EPISODES = 1`, `SET_POINTS_LEN = 5`, and plots disabled.
- Passed a tiny `RunTwoPhase_GART_LMPC.py` smoke run with the same tiny two-phase profile and plots disabled.
