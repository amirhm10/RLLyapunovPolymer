# Soften Phase-2 Disturbance And Use Ten RL Seeds

## Objective

Apply the softer, offset-free-MPC-achievable Phase-2 disturbance profile to the five main two-phase runners and increase the online RL default seed count to ten for the comparison study.

## Changes

- Updated Phase-2 disturbance multipliers in all five main runners:
  - `PHASE2_QI_MULTIPLIER = 1.02`
  - `PHASE2_QS_MULTIPLIER = 0.97`
  - `PHASE2_HA_MULTIPLIER = 0.90`
- Updated the shared `TwoPhaseExperimentSpec` defaults to the same softened Phase-2 disturbance.
- Updated the four online RL runners to default to `N_SEEDS = 10`.
- Left `RunTwoPhase_GART_LMPC.py` at `N_SEEDS = 1`.

## Resulting Phase-2 Disturbance

With the nominal values:

- `Qi = 108.0`
- `Qs = 459.0`
- `hA = 1.05e6`

Phase 2 now ramps from the Phase-1 final disturbance to:

- `Qi = 110.16`
- `Qs = 445.23`
- `hA = 945000.0`

## Validation

- `python -X pycache_prefix=... -m py_compile RunTwoPhase_OFMPCPretrained_SafetyGate.py RunTwoPhase_OFMPCPretrained_NoSafetyGate.py RunTwoPhase_ColdStart_SafetyGate.py RunTwoPhase_ColdStart_NoSafetyGate.py RunTwoPhase_GART_LMPC.py RunOnlineTD3TwoPhaseStudy.py utils/two_phase_profiles.py`
- Runtime profile construction in the `rl` conda environment confirmed:
  - pretrained runner default `n_seeds = 10`
  - cold-start runner default `n_seeds = 10`
  - Phase-2 multipliers `1.02`, `0.97`, `0.90`
  - Phase-2 final disturbance `Qi = 110.16`, `Qs = 445.23`, `hA = 945000.0`
  - total profile length `160000` steps and `200` reporting windows

## Note

`OffsetFreeMPC_DisturbanceRunner.py` was already locally modified for feasibility probing and is intentionally not part of this change report.
