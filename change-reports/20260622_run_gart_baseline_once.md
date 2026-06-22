# Run GART-LMPC Baseline Once

## Objective

Avoid repeating the GART-LMPC baseline across random seeds, since the two-phase GART-LMPC formulation is deterministic for a fixed setpoint and disturbance profile.

## Changes

- Updated `RunTwoPhase_GART_LMPC.py` so it defaults to one reference run:
  - `REFERENCE_SEED = 42`
  - `SEEDS = (REFERENCE_SEED,)`
  - `N_SEEDS = 1`
- Added deterministic-baseline seed handling in `RunOnlineTD3TwoPhaseStudy.py`.
  - If the requested methods are only deterministic baselines such as `gart_lmpc`, the effective seed list is collapsed to the first requested seed.
  - TD3 method runs keep the full requested seed list.
- Added `requested_seeds` to the batch manifest so the manifest records both the user-requested seed set and the effective seed set actually used.

## Validation

- Checked that the GART runner builds arguments with only seed `42`.
- Checked that GART-only multi-seed requests collapse to the first seed.
- Checked that TD3-only and mixed TD3/GART requests keep the full seed list.
- In-memory Python syntax compile passed for:
  - `RunTwoPhase_GART_LMPC.py`
  - `RunOnlineTD3TwoPhaseStudy.py`
