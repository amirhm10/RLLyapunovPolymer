# Pretraining Runner Local Defaults

Date: 2026-06-08

## Summary

Moved run-tunable OF-MPC TD3 pretraining defaults out of `utils/of_mpc_td3_workflow.py` and into the root runners. The helper module now receives sample counts, epoch counts, pretraining batch size, and actor/critic hidden layer sizes through configuration.

## Changes

- `PretrainTD3OffsetFreeMPC.py`
  - Defines the editable defaults for sample counts, chunk size, actor/critic epochs, pretraining batch size, and actor/critic layer sizes.
  - Adds CLI overrides for `--pretrain-batch-size`, `--actor-layers`, and `--critic-layers`.
- `ComparePretrainedTD3OffsetFreeMPC.py`
  - Defines actor/critic layer defaults locally for loading checkpoints.
  - Adds matching `--actor-layers` and `--critic-layers` overrides.
- `utils/of_mpc_td3_workflow.py`
  - Removes the pretraining workload and architecture constants as helper-level source-of-truth values.
  - Threads batch size and layer sizes through `PretrainingRunConfig` and `ComparisonRunConfig`.
- `report/of_mpc_td3_pretraining_process_2026-06-08.md`
  - Documents that the editable defaults live in the root runner and shows the new CLI overrides.

## Validation Plan

```powershell
python -m py_compile PretrainTD3OffsetFreeMPC.py ComparePretrainedTD3OffsetFreeMPC.py utils/of_mpc_td3_workflow.py
python PretrainTD3OffsetFreeMPC.py --help
python ComparePretrainedTD3OffsetFreeMPC.py --help
python PretrainTD3OffsetFreeMPC.py --mpc-samples 32 --steady-samples 8 --chunk-size 16 --actor-epochs 1 --critic-epochs 1 --pretrain-batch-size 16
```

## Validation Results

- `python -m py_compile PretrainTD3OffsetFreeMPC.py ComparePretrainedTD3OffsetFreeMPC.py utils/of_mpc_td3_workflow.py` passed.
- `python PretrainTD3OffsetFreeMPC.py --help` shows runner-owned defaults for sample counts, chunk size, epochs, pretraining batch size, and actor/critic layers.
- `python ComparePretrainedTD3OffsetFreeMPC.py --help` shows actor/critic layer overrides for checkpoint loading.
- Explicit small pretraining smoke passed:
  - command: `python PretrainTD3OffsetFreeMPC.py --mpc-samples 32 --steady-samples 8 --chunk-size 16 --actor-epochs 1 --critic-epochs 1 --pretrain-batch-size 16`
  - checkpoint: `results/PretrainOFMPC/20260608_171258/of_mpc_pretrained_td3_20260608_171307.pkl`
  - config recorded `pretrain_batch_size = 16` and `[512, 512, 512, 512, 512]` actor/critic layer sizes.
- Saved-agent comparison smoke passed with explicit layer sizes:
  - command: `python ComparePretrainedTD3OffsetFreeMPC.py --agent-path results/PretrainOFMPC/20260608_171258/of_mpc_pretrained_td3_20260608_171307.pkl --set-points-len 10 --modes nominal --actor-layers 512,512,512,512,512 --critic-layers 512,512,512,512,512`
  - summary: `results/PretrainOFMPCComparison/20260608_171320/summary.json`

The comparison smoke emitted the existing observer pole-placement convergence warning from `Simulation/mpc.py`, but completed and saved metrics.
