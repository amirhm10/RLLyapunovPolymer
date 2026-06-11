# Align LMPC Pretraining With Bounded-Mixed Selector

Date: 2026-06-11

## Summary

Updated the Direct LMPC TD3 pretraining workflow so new LMPC expert labels use the same bounded mixed Direct LMPC target selector as the current online disturbance runners:

```python
target_mode = "bounded"
target_selector_variant = "bounded_mixed_u0p1_x0p1"
target_config = {
    "u_ref_weight": 0.1,
    "x_ref_weight": 0.1,
}
rho_lyap = 0.99
lyap_eps = 1e-3
lyap_tol = 1e-10
slack_penalty = 1e6
```

The online disturbance runner and LMPC pretraining workflow now import these values from a shared defaults module so future selector changes cannot drift silently.

## Implementation

- Added `utils/direct_lmpc_selector_defaults.py` as the shared source of Direct LMPC bounded-mixed selector constants, target config, metadata, and cache token.
- Updated `utils/online_disturbance_runner.py` to consume the shared selector constants without changing the current online runner behavior.
- Updated `utils/lmpc_td3_workflow.py` so LMPC label generation, saved `config.json`, `summary.json`, `label_diagnostics.json`, and LMPC comparison baselines use and record the bounded-mixed selector.
- Added the selector token to LMPC comparison baseline cache filenames so governed-reference cached baselines cannot be reused for bounded-mixed comparisons.

## Interpretation

New checkpoints created under `results/PretrainLMPC/<timestamp>/` after this change should be interpreted as bounded-mixed-selector LMPC-pretrained checkpoints. Older LMPC checkpoints produced before this change remain valid historical artifacts, but reports should not mix them with new online bounded-mixed runs without recording the checkpoint selector used during pretraining.

## Validation

Static validation passed:

```powershell
python -m py_compile utils/direct_lmpc_selector_defaults.py utils/online_disturbance_runner.py utils/lmpc_td3_workflow.py PretrainTD3LyapunovMPC.py ComparePretrainedTD3LyapunovMPC.py
```

Tiny LMPC pretraining smoke passed in `rl-env`:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe PretrainTD3LyapunovMPC.py --lmpc-samples 2 --steady-samples 1 --candidate-chunk-size 4 --worker-batch-size 2 --max-attempt-multiplier 20 --label-n-jobs 1 --parallel-backend sequential --actor-epochs 1 --critic-epochs 1 --pretrain-batch-size 3 --device cpu --output-root results/PretrainLMPCBoundedMixedSmoke
```

Smoke artifact:

`results/PretrainLMPCBoundedMixedSmoke/20260611_003339/`

Confirmed in smoke `summary.json`, `config.json`, and `label_diagnostics.json`:

- `target_mode = "bounded"`
- `target_selector_variant = "bounded_mixed_u0p1_x0p1"`
- `target_config = {"u_ref_weight": 0.1, "x_ref_weight": 0.1}`
- `rho_lyap = 0.99`
- `lyap_eps = 0.001`
- accepted labels: 3 of 3 requested

Online LMPC-pretrained load checks passed:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe OnlineTD3_LMPCPretrained_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
C:\Users\hamediaa\.conda\envs\rl-env\python.exe OnlineTD3_LMPCPretrained_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
```

The two online checks loaded the latest existing `results/PretrainLMPC` checkpoint successfully and used the bounded selector online, with `target_stage = "frozen_output_disturbance_bounded_ls"`.
