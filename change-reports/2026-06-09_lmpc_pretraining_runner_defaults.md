# Move LMPC Pretraining Defaults To Runner

Date: 2026-06-09

## Summary

Updated `PretrainTD3LyapunovMPC.py` so the main workload and architecture defaults are easy to edit directly in the runner, matching the current OF-MPC pretraining runner style.

## Changes

- Added runner-local defaults:
  - `DEFAULT_MPC_SAMPLES = 2_000_000`
  - `DEFAULT_LMPC_SAMPLES = DEFAULT_MPC_SAMPLES`
  - `DEFAULT_STEADY_SAMPLES = 100_000`
  - `DEFAULT_CHUNK_SIZE = 100_000`
  - `DEFAULT_CANDIDATE_CHUNK_SIZE = DEFAULT_CHUNK_SIZE`
  - `DEFAULT_ACTOR_EPOCHS = 1000`
  - `DEFAULT_CRITIC_EPOCHS = 500`
  - `DEFAULT_PRETRAIN_BATCH_SIZE = 8192`
  - `DEFAULT_ACTOR_LAYER_SIZES = [256, 256, 256]`
  - `DEFAULT_CRITIC_LAYER_SIZES = [256, 256, 256]`

- Kept LMPC-specific runner defaults near the same block:
  - `DEFAULT_WORKER_BATCH_SIZE = DEFAULT_PRETRAIN_BATCH_SIZE`
  - `DEFAULT_MAX_ATTEMPT_MULTIPLIER = 5.0`
  - `DEFAULT_LABEL_N_JOBS = -1`
  - `DEFAULT_PARALLEL_BACKEND = "loky"`

- Updated `report/lmpc_td3_pretraining_process_2026-06-09.md` to describe the new runner-local defaults.

## Validation

Static validation passed:

```powershell
python -m py_compile PretrainTD3LyapunovMPC.py
```

CLI default validation passed in `rl-env`:

```powershell
python PretrainTD3LyapunovMPC.py --help
```

The help text shows:

- `--lmpc-samples` default `2000000`
- `--steady-samples` default `100000`
- `--candidate-chunk-size` default `100000`
- `--worker-batch-size` default `8192`
- `--label-n-jobs` default `-1`
- `--parallel-backend` default `loky`
- `--actor-epochs` default `1000`
- `--critic-epochs` default `500`
- `--pretrain-batch-size` default `8192`
- `--actor-layers` default `256,256,256`
- `--critic-layers` default `256,256,256`
