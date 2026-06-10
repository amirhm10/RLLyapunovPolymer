# Align LMPC Pretraining Lyapunov Config

Date: 2026-06-09

## Summary

Aligned the Direct LMPC TD3 pretraining and comparison workflow with the active Direct Lyapunov runner configuration.

The active runners use:

- `rho_lyap = 0.99`
- `lyap_eps = 5e-3`
- `slack_penalty = 1e6`
- governed-reference target mode
- hard Lyapunov tracking mode
- first-step contraction enabled
- `use_target_output_for_tracking = False`
- `use_target_on_solver_fail = False`

The LMPC pretraining/comparison helper already matched the other values, but it still used `lyap_eps = 1e-9`. That has now been changed to `5e-3`.

## Changes

- Updated `utils/lmpc_td3_workflow.py`.
  - Changed `LYAP_EPS` from `1e-9` to `5e-3`.
  - LMPC label generation now uses the same contraction tolerance as `DirectLyapunovMPC.py`.
  - LMPC and OF-MPC diagnostic comparison baselines now use the same contraction tolerance.
  - LMPC comparison baseline cache filenames now include both objective-weight and Lyapunov-contraction tokens, such as `_q5_1_r1_1_rho0p99_eps0p005`.

- Updated documentation.
  - `report/lmpc_td3_pretraining_process_2026-06-09.md`
  - `change-reports/2026-06-09_lmpc_td3_pretraining_workflow.md`

## Validation

Static validation passed:

```powershell
python -m py_compile utils/lmpc_td3_workflow.py PretrainTD3LyapunovMPC.py ComparePretrainedTD3LyapunovMPC.py
```

Tiny LMPC smoke pretraining passed:

```text
results/PretrainLMPCEpsSmoke/20260609_194353/
```

Observed:

- `config.json` recorded `rho_lyap = 0.99`
- `config.json` recorded `lyap_eps = 0.005`
- accepted labels: `2`
- attempted candidates: `2`
- actor BC loss entries: `1`
- critic TD loss entries: `1`

Tiny nominal comparison passed:

```text
results/PretrainLMPCEpsComparisonSmoke/20260609_194410/
```

The generated baseline artifacts used the corrected cache token:

```text
direct_lmpc_nominal_n1_len10_disturb_before_q5_1_r1_1_rho0p99_eps0p005.pickle
offset_free_mpc_nominal_n1_len10_disturb_before_q5_1_r1_1_rho0p99_eps0p005.pickle
```
