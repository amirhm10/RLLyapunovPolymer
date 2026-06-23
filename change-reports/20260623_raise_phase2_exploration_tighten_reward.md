# Raise Phase-Two Exploration Floor And Tighten Reward Band

## Summary

Adjusted the two-phase online TD3 study to keep a modest amount of Phase-2 input-deviation exploration and made the online reward less tolerant of residual steady-state tracking error.

## Changes

- Raised the full-RL exploration floor from `0.005` to `0.01` in all four RL two-phase runners.
- Raised handoff exploration end noise from `0.005` to `0.01` so the handoff and full-RL floor are aligned.
- Updated the shared online runner defaults to use the same `0.01` exploration floor.
- Updated two-phase profile validation to expect the `0.01` Phase-1 endpoint and Phase-2 floor.
- Tightened the reward band from `[0.006, 0.08]` to `[0.005, 0.07]` in physical output units.
- Increased inside-band tracking pressure:
  - `gamma_in: 3.0 -> 5.0`
  - `lam_in: 3.0 -> 5.0`
- Applied the reward change to both the online TD3 runner and GART-LMPC baseline reward reporting so `reward_no_penalty` remains comparable.

## Rationale

The first Phase-2 setpoint cycle is mostly a robustness/generalization test, but a floor of `0.005` in input-deviation space may be too quiet for adaptation after the disturbance profile changes. Raising it to `0.01` keeps exploration modest while allowing more useful Phase-2 data.

The previous reward band was also intentionally forgiving near the setpoint. Tightening the band and strengthening the inside-band tracking terms should reduce the chance that the actor treats a visible steady-state offset as good enough.

## Expected Effect

- More active Phase-2 adaptation without returning to large exploratory moves.
- Stronger reward gradient for small-to-moderate residual output errors.
- Slightly more sensitivity to input perturbation during Phase 2, so compare input movement and safety activity alongside tracking metrics.

## Validation

Run:

```powershell
python -m py_compile RunTwoPhase_OFMPCPretrained_SafetyGate.py RunTwoPhase_OFMPCPretrained_NoSafetyGate.py RunTwoPhase_ColdStart_SafetyGate.py RunTwoPhase_ColdStart_NoSafetyGate.py RunOnlineTD3TwoPhaseStudy.py utils\online_disturbance_runner.py experiments\run_gart_target_selector_study.py
```
