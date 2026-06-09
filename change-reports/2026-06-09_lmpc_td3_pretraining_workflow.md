# Add Direct LMPC TD3 Pretraining Workflow

Date: 2026-06-09

## Summary

Added a Direct Output-Disturbance Lyapunov MPC expert-pretraining workflow for TD3. The new workflow mirrors the OF-MPC migration structure with reusable helpers separated from root runners.

## Changes

- Added `utils/lmpc_td3_workflow.py`.
  - Builds the active Direct Lyapunov MPC expert with governed-reference target selection.
  - Generates replay labels from successful hard-LMPC solves only.
  - Skips failed target/tracking solves and writes label diagnostics.
  - Trains TD3 with actor behavioral cloning and critic TD warm-up.
  - Adds comparison helpers for TD3, Direct LMPC, and OF-MPC diagnostic baselines.
  - Infers checkpoint architecture from metadata or saved network weights.

- Added `PretrainTD3LyapunovMPC.py`.
  - Provides the LMPC pretraining/saving CLI.
  - Defaults to the moderate LMPC workload of `100_000` broad labels and `10_000` near-steady labels.

- Added `ComparePretrainedTD3LyapunovMPC.py`.
  - Loads a saved checkpoint.
  - Infers actor/critic architecture unless explicitly overridden.
  - Compares TD3 against Direct LMPC and OF-MPC baselines.

- Updated `TD3Agent/agent.py`.
  - Stores `state_dim`, `action_dim`, actor hidden layers, and critic hidden layers in new checkpoint `hparams`.
  - Keeps old checkpoint load behavior unchanged.

- Added `report/lmpc_td3_pretraining_process_2026-06-09.md`.
  - Documents the mathematical workflow, replay-buffer construction, artifacts, CLI, smoke validation, limitations, and next experiment.

## Validation

- Static validation passed:

```powershell
python -m py_compile PretrainTD3LyapunovMPC.py ComparePretrainedTD3LyapunovMPC.py utils/lmpc_td3_workflow.py TD3Agent/agent.py
python -m py_compile PretrainTD3OffsetFreeMPC.py ComparePretrainedTD3OffsetFreeMPC.py utils/of_mpc_td3_workflow.py TD3Agent/agent.py
```

- The default shell Python lacks `cvxpy`, so runtime validation used:

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe
```

- LMPC smoke pretraining passed:

```powershell
python PretrainTD3LyapunovMPC.py --lmpc-samples 2 --steady-samples 2 --candidate-chunk-size 2 --worker-batch-size 1 --max-attempt-multiplier 20 --actor-epochs 1 --critic-epochs 1 --pretrain-batch-size 4 --device cpu
```

Generated bundle:

```text
results/PretrainLMPC/20260609_163533/
```

- LMPC smoke comparison passed:

```powershell
python ComparePretrainedTD3LyapunovMPC.py --agent-path results/PretrainLMPC/20260609_163533/lmpc_pretrained_td3_20260609_163535.pkl --set-points-len 10 --n-tests 1 --modes nominal --device cpu
```

Generated bundle:

```text
results/PretrainLMPCComparison/20260609_163544/
```

- Failure handling was verified with a low attempt multiplier. The run failed clearly after accepting 1 of 3 requested labels and wrote diagnostics:

```text
results/PretrainLMPC/20260609_163559/label_diagnostics.json
```

- Existing OF-MPC smoke pretraining passed after allowing Windows joblib worker processes:

```text
results/PretrainOFMPCSmoke/20260609_191259/
```

## Notes

- The LMPC label generator is currently sequential because CVXPY solver objects are not passed through joblib workers.
- The implementation follows the requested strict `lyap_eps = 1e-9`; this is stricter than the converted Direct Lyapunov runners that currently use `lyap_eps = 5e-3`.
- Existing local user edits in `PretrainTD3OffsetFreeMPC.py` and `ComparePretrainedTD3OffsetFreeMPC.py` were not reverted.
