# Default RL Two-Phase Runners To Five Seeds

## Objective

Set the four online RL two-phase runner defaults to five paired seeds for the main comparison runs while keeping the deterministic GART-LMPC baseline at one run.

## Changes

- Updated `N_SEEDS = 5` in:
  - `RunTwoPhase_OFMPCPretrained_SafetyGate.py`
  - `RunTwoPhase_OFMPCPretrained_NoSafetyGate.py`
  - `RunTwoPhase_ColdStart_SafetyGate.py`
  - `RunTwoPhase_ColdStart_NoSafetyGate.py`
- Left `RunTwoPhase_GART_LMPC.py` at `N_SEEDS = 1`.

With `SEED_START = 0` and `SEEDS = None`, the RL runners now execute seeds `0, 1, 2, 3, 4` by default.

## Validation

- `python -X pycache_prefix=... -m py_compile RunTwoPhase_OFMPCPretrained_SafetyGate.py RunTwoPhase_OFMPCPretrained_NoSafetyGate.py RunTwoPhase_ColdStart_SafetyGate.py RunTwoPhase_ColdStart_NoSafetyGate.py RunTwoPhase_GART_LMPC.py`
- Confirmed the four RL runners show `N_SEEDS = 5` and GART remains `N_SEEDS = 1`.
