# Add Pretraining Loss Logging Artifacts

Date: 2026-06-09

## Summary

Fixed the OF-MPC and LMPC TD3 pretraining workflows so actor and critic losses are saved in analyzable artifacts after each run.

The previous OF-MPC result bundle had an empty `loss_arrays.json`, even though training printed epoch losses to the console. Future runs now save explicit loss histories and fail clearly if requested pretraining epochs do not produce loss entries.

## Changes

- Updated `TD3Agent/agent.py`.
  - `pretrain_from_buffer(...)` now returns a structured pretraining history.
  - The history includes actor behavioral-cloning losses, critic TD losses, learning rates, and sample counts.
  - New checkpoints also include a `training_losses` block.

- Updated `utils/of_mpc_td3_workflow.py`.
  - `save_loss_artifacts(...)` now writes `loss_arrays.json`, `loss_arrays.csv`, `loss_summary.json`, and `pretraining_history.json`.
  - The writer validates actor and critic loss counts against the requested epoch counts.
  - OF-MPC pretraining now passes the returned history into the artifact writer before saving the checkpoint.

- Updated `utils/lmpc_td3_workflow.py`.
  - LMPC pretraining uses the same loss-artifact writer and validation path.

- Updated reports.
  - `report/of_mpc_td3_pretraining_process_2026-06-08.md`
  - `report/lmpc_td3_pretraining_process_2026-06-09.md`

## Validation

Static validation passed:

```powershell
python -m py_compile TD3Agent/agent.py utils/of_mpc_td3_workflow.py utils/lmpc_td3_workflow.py PretrainTD3OffsetFreeMPC.py PretrainTD3LyapunovMPC.py
```

OF-MPC smoke pretraining passed after allowing `joblib/loky` worker process spawning:

```text
results/PretrainOFMPCSmokeLoss/20260609_193807/
```

Observed loss artifacts:

- actor BC loss entries: `1`
- critic TD loss entries: `1`
- actor BC loss: `0.0693`
- critic TD loss: `250.68`
- files verified: `loss_arrays.json`, `loss_arrays.csv`, `loss_summary.json`, `pretraining_history.json`

LMPC smoke pretraining passed:

```text
results/PretrainLMPCSmokeLoss/20260609_193832/
```

Observed loss artifacts:

- actor BC loss entries: `1`
- critic TD loss entries: `1`
- actor BC loss: `0.0620`
- critic TD loss: `215.22`
- files verified: `loss_arrays.json`, `loss_arrays.csv`, `loss_summary.json`, `pretraining_history.json`
