# OF-MPC Pretrain Interrupt Checkpoints

## Problem

An OF-MPC TD3 pretraining run stopped during critic TD warm-up after epoch 93 and only printed `joblib` resource-tracker cleanup warnings at shutdown. The log did not include a Python traceback, and the existing workflow only wrote the final checkpoint, config, loss artifacts, and summary after all actor and critic epochs completed.

## Change

- Added optional periodic pretraining checkpoints to `TD3Agent.pretrain_from_buffer`.
- Enabled OF-MPC pretraining partial checkpoints every 25 actor or critic epochs by default.
- Added `--checkpoint-interval-epochs` to `PretrainTD3OffsetFreeMPC.py`.
- Wrote OF-MPC `config.json` before the long pretraining stage starts.
- Added `KeyboardInterrupt` handling in `run_of_mpc_pretraining` to save:
  - an interrupted TD3 checkpoint,
  - partial loss arrays and pretraining history,
  - a `summary.json` with `status = "interrupted"`.
- Disabled joblib array memmapping for OF-MPC expert-label workers with `max_nbytes=None` to reduce Windows temp-folder cleanup warnings after shutdown.

## Practical Effect

Future partial checkpoints use filenames beginning with `of_mpc_pretrained_td3_`, so the existing pretrained online runners can discover them through the normal latest-checkpoint resolver. For example, if the run stops around critic epoch 93, the latest periodic checkpoint should usually be the critic epoch 75 checkpoint with the cloned actor preserved.

## Validation

- `python -m py_compile PretrainTD3OffsetFreeMPC.py utils/of_mpc_td3_workflow.py utils/td3_helpers.py TD3Agent/agent.py` passed with `PYTHONPYCACHEPREFIX` pointed at the system temp directory.

The first compile attempt using the default in-repo `__pycache__` failed with a Windows access-denied error, so the temp bytecode cache was used for validation.
