# Separate MPC Objective and RL Reward Weights

Date: 2026-06-09

## Summary

Fixed a weight-coupling mismatch between controller objectives and RL rewards.

The intended convention is now explicit:

- MPC and Direct LMPC objective weights use `Q = [5, 1]`, `R = [1, 1]`.
- Offline TD3 pretraining rewards use the same one-step quadratic MPC stage cost with `Q = [5, 1]`, `R = [1, 1]`.
- Online RL training uses the shaped relative-QR reward with its own reward weights, currently `Q_reward = [12, 6]`, `R_reward = [1, 1]`.
- Shaped RL reward weights no longer set the LMPC fallback/safety-gate or OF-MPC diagnostic objective weights.

## Changes

- Updated `DirectLyapunovSafetyGateRL_Pretrained.py`.
  - Split `Qy_mpc_diag`, `Su_mpc_diag`, `Rdu_mpc_diag` from `Qy_reward_diag`, `Rdu_reward_diag`.
  - Direct LMPC fallback and OF-MPC diagnostic baselines now use `[5, 1]`.
  - Online shaped reward still uses `[12, 6]`.
  - Case configs now record both controller and reward weights.

- Updated `DirectLyapunovSafetyGateRL_ColdStart.py`.
  - Applied the same controller/reward separation as the pretrained runner.

- Updated `utils/lmpc_td3_workflow.py`.
  - Direct LMPC expert labels now use `[5, 1]`.
  - Offline pretraining replay reward now uses `make_reward_fn_mpc_quadratic(...)` with `[5, 1]`, not the online shaped reward.
  - OF-MPC diagnostic baseline in the LMPC comparison runner now uses `[5, 1]`.
  - LMPC comparison baseline cache filenames now include the objective-weight token, such as `_q5_1_r1_1`, to avoid reusing older `[12, 6]` cached baselines.

- Updated `report/lmpc_td3_pretraining_process_2026-06-09.md`.
  - Replaced the old `[12, 6]` LMPC pretraining description with the corrected `[5, 1]` controller objective and offline quadratic reward.
  - Added corrected smoke validation paths and metrics.

## Validation

- Static validation passed:

```powershell
python -m py_compile DirectLyapunovSafetyGateRL_Pretrained.py DirectLyapunovSafetyGateRL_ColdStart.py utils/lmpc_td3_workflow.py PretrainTD3LyapunovMPC.py ComparePretrainedTD3LyapunovMPC.py
```

- Corrected LMPC smoke pretraining passed in `rl-env`:

```text
results/PretrainLMPC/20260609_192848/
```

Observed:

- accepted labels: `4`
- attempted labels: `4`
- actor BC loss: `0.0594`
- critic TD loss: `182.21`

- Corrected LMPC smoke comparison passed:

```text
results/PretrainLMPCComparison/20260609_192911/
```

The Direct LMPC and OF-MPC baseline paths include `_q5_1_r1_1`, confirming that the corrected objective weights are part of the cache key.

## Notes

- Existing result bundles produced before this fix may have used `[12, 6]` as controller objective weights in the online safety-gate RL runners or the first LMPC pretraining smoke bundle.
- OF-MPC pretraining and standalone `DirectLyapunovMPC.py` already used `[5, 1]` and were not the source of this mismatch.
