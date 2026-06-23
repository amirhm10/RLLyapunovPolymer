# Default One-Seed Two-Phase Runner Settings

## Objective

Restore the method-specific two-phase TD3 runners to a safe default of one seed while keeping the old reporting semantics where each printed sub-episode average covers one full 800-sample two-setpoint episode.

## Changes

- Set `N_SEEDS = 1` in the four TD3 method runners.
- Kept `PHASE1_SETPOINT_HOLD_STEPS = 400` and `REPORTING_WINDOW_STEPS = 800`.
- Added a short runner comment documenting that one reporting episode is `2 * 400 = 800` samples.

## Files Updated

- `RunTwoPhase_OFMPCPretrained_SafetyGate.py`
- `RunTwoPhase_OFMPCPretrained_NoSafetyGate.py`
- `RunTwoPhase_ColdStart_SafetyGate.py`
- `RunTwoPhase_ColdStart_NoSafetyGate.py`

## Validation

- Syntax compilation passed with bytecode redirected away from the OneDrive `__pycache__` folder.
- Runner config check with the `rl` conda environment confirmed:
  - `n_seeds = 1`
  - `set_points_len = 400`
  - `reporting_window_steps = 800`
  - `phase2_episodes = 50`
  - `phase2_steps = None`

Thus both phase 1 and phase 2 report/print averages over complete 800-sample episodes.
