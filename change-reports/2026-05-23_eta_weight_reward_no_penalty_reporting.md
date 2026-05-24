# Eta Weight And Penalty-Free Reward Reporting

Date: 2026-05-23

## Summary

This change prepares the next ColdStart and Pretrain RL runs by increasing the viscosity-like output reward weight and by storing a penalty-free reward channel for fair reporting across Direct Lyapunov MPC, ColdStart RL, and Pretrain RL.

## Changes

- Increased the RL reward output weight from `Qy_diag = [8.0, 6.0]` to `Qy_diag = [12.0, 6.0]` in both active RL runners.
- Preserved the strict RL training reward with safety-gate fallback penalties, fixed fallback-event penalty, and all existing reward terms.
- Added `reward_no_penalty` to the reward component dictionary returned by `make_reward_fn_relative_QR(...)`.
- Propagated `reward_no_penalty` through RL safety-gate step records, bundles, summaries, episode tables, NPZ exports, comparison tables, and comparison plots.
- Added Direct Lyapunov MPC aliases where `reward_base`, `reward_no_penalty`, and `reward_augmented` equal the existing direct reward, since Direct Lyapunov MPC does not train with the RL safety-gate fallback penalty.
- Updated `AGENTS.md` so future reports remember to include `reward_no_penalty` alongside the actual RL training reward.
- Set the three active runners to `rho_lyap = 0.98` and `lyap_eps = 1e-9` for the next experiment.
- Strengthened `AGENTS.md` to require a local Git commit after requested code/config/report changes whenever it is safe to commit only the intended files.

## Interpretation

`reward` remains the actual training reward for RL. It includes the fallback/event penalty and is therefore the signal that shapes the learned policy.

`reward_no_penalty` is intended for cross-method reporting. It represents the control-performance reward before the safety penalty is applied, so comparisons do not automatically punish the RL cases twice for using the safety layer.

## Validation

- Passed: `python -m py_compile DirectLyapunovMPC.py DirectLyapunovSafetyGateRL_ColdStart.py DirectLyapunovSafetyGateRL_Pretrained.py TD3Agent/reward_functions.py Simulation/run_rl_lyapunov.py Lyapunov/safety_debug.py Lyapunov/direct_lyapunov_mpc.py`.
- Passed: synthetic reward-component check confirming `reward_no_penalty == reward_base`, no fallback penalty is applied when `fallback_active=False`, and an active fixed fallback event penalty of `10.0` lowers `reward` by `10.0`.
- Passed: `git diff --check`.

## Notes

Existing local run-setting edits in the active runner files were preserved.
