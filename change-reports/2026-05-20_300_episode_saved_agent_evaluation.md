# 300-Episode Training And Saved-Agent Evaluation

## Summary

Implemented the next experiment workflow:

- Increased the active training horizon from 200 to 300 episodes.
- Removed the forced final evaluation episode from the two RL training scripts.
- Added a separate saved-agent evaluation entrypoint that loads trained cold-start and pretrained RL agents and compares them against offset-free MPC and direct Lyapunov MPC without retraining.

## Code Changes

- `utils/direct_lyapunov_study.py`
  - Set `DIRECT_DISTURBANCE_N_TESTS = 300`.

- `utils/helpers.py`
  - Extended `generate_setpoints_training_rl_gradually(...)` with:
    - `force_final_test=True`
    - `disturbance_profile=None`
  - Preserved backward compatibility by keeping the final forced test episode as the default.
  - Added full-length fixed disturbance profile validation for `qi`, `qs`, and `ha`.

- `Simulation/run_rl_lyapunov.py`
  - Extended `run_rl_train(...)` to pass through `force_final_test` and `disturbance_profile`.

- `Lyapunov/direct_lyapunov_mpc.py`
  - Extended direct LMPC and offset-free MPC diagnostic rollout helpers with matching `force_final_test` and `disturbance_profile` options.

- `DirectLyapunovMPC.py`
  - Set `n_episodes = 300`.

- `DirectLyapunovSafetyGateRL_ColdStart.py`
  - Uses the shared 300-episode setting.
  - Sets `FORCE_FINAL_TEST = False`.
  - Passes `force_final_test=False` into RL training.

- `DirectLyapunovSafetyGateRL_Pretrained.py`
  - Uses the shared 300-episode setting.
  - Sets `FORCE_FINAL_TEST = False`.
  - Passes `force_final_test=False` into RL training.

- `DirectLyapunovSavedAgentEvaluation.py`
  - New root evaluation script.
  - Auto-loads latest non-`mpc_only` saved agents from `results/ColdStart/...` and `results/Pretrain/...`.
  - Supports manual path overrides with `COLD_AGENT_PATH` and `PRETRAIN_AGENT_PATH`.
  - Runs a 5-episode fixed disturbance suite:
    - nominal
    - `Qi = 0.95 Qi_nom`
    - `Qs = 1.05 Qs_nom`
    - `hA = 0.92 hA_nom`
    - all three steps together
  - Keeps the direct Lyapunov safety gate active for both saved RL agents.
  - Compares:
    - cold saved RL with safety gate
    - pretrained saved RL with safety gate
    - offset-free MPC with direct Lyapunov diagnostic would-be activation
    - direct Lyapunov MPC
  - Saves under `results/SavedAgentEval/<timestamp>/`.
  - Exports case debug bundles, comparison CSVs, scenario table, JSON summary, and comparison plots.
  - Includes `--dry-run` to resolve agents, build scenarios, and print the plan without running rollouts.

## Validation

- `python -m py_compile DirectLyapunovMPC.py DirectLyapunovSafetyGateRL_ColdStart.py DirectLyapunovSafetyGateRL_Pretrained.py DirectLyapunovSavedAgentEvaluation.py Simulation/run_rl_lyapunov.py Lyapunov/direct_lyapunov_mpc.py utils/helpers.py utils/direct_lyapunov_study.py`
- Synthetic helper check confirmed:
  - default `force_final_test=True` still marks the final cycle as test
  - `force_final_test=False` leaves all cycles trainable when the supplied cycle is trainable
  - fixed disturbance profiles remain constant inside each episode block
- `python DirectLyapunovSavedAgentEvaluation.py --dry-run`
  - Confirmed latest cold/pretrained non-`mpc_only` agents resolve correctly.
  - Confirmed five scenarios and 4000 planned control steps.
- `git diff --check`

## Notes

- Full saved-agent evaluation was not run during implementation.
- MPC-only fallback counts in the evaluation comparison are treated as would-be direct Lyapunov gate activations, not actual fallback intervention.
