# Strict Offset Reward Defaults

## Summary

Implemented the strict offset-aligned reward setup for the direct Lyapunov safety-gated TD3 cold-start and pretrained notebooks. The new defaults increase near-setpoint tracking pressure, reduce broad inside-band reward forgiveness, and make safety-gate fallback events more expensive.

## Changes

- Added `fallback_event_penalty` to `make_reward_fn_relative_QR` with a backward-compatible default of `0.0`.
- The fallback penalty now combines the weighted correction gap and a fixed event cost whenever `fallback_active=True`.
- Added fallback correction and event-penalty diagnostics to reward components and rollout info.
- Updated the cold-start and pretrained safety-gate RL notebooks to use the strict offset reward defaults:
  `Qy_diag=[8.0, 4.0]`, `k_rel=[0.0015, 0.00015]`, `band_floor_phys=[0.003, 0.035]`, `gate="prod"`, `bonus_kind="quadratic"`, `gamma_fallback=2.0`, and `fallback_event_penalty=0.5`.
- Preserved the latest exploration/noise setup:
  cold start uses exploration `0.2` and policy smoothing noise `0.1`, while pretrained uses exploration `0.02` and policy smoothing noise `0.01`.

## Validation

- Passed `py_compile` for `TD3Agent/reward_functions.py` and `Simulation/run_rl_lyapunov.py`.
- Passed `nbformat` validation for both edited notebooks.
- Passed a synthetic reward check confirming `fallback_event_penalty=0.0` preserves old behavior, inactive fallback applies no fixed penalty, and active fallback with `fallback_event_penalty=0.5` subtracts exactly `0.5` when the correction gap is zero.

## Notes

No full notebook training or result regeneration was performed for this change.
