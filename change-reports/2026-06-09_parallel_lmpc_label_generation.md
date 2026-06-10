# Parallelize LMPC Label Generation

Date: 2026-06-09

## Summary

Parallelized Direct LMPC TD3 pretraining label generation so the production-scale LMPC replay buffer can be generated in a practical amount of time.

The OF-MPC pretraining workflow already used `joblib.Parallel`. The LMPC workflow was sequential because it includes governed-reference target selection and CVXPY-backed Direct LMPC solves. The new design keeps the replay buffer owned by the parent process, while each worker constructs its own LMPC solver from serializable matrices.

## Changes

- Updated `utils/lmpc_td3_workflow.py`.
  - Added `joblib.Parallel` and `delayed` label generation.
  - Added worker-local batch labeling helpers.
  - Each worker builds its own `LMPCComponents` and offline quadratic reward function.
  - The parent process samples candidates, aggregates diagnostics, accepts only the requested number of successful labels, and writes accepted transitions into the replay buffer.
  - Label diagnostics now record `label_n_jobs`, `parallel_backend`, `worker_batch_size`, `discarded_successes`, `successful_solves`, and `solve_success_rate`.

- Updated `LMPCPretrainingRunConfig`.
  - Added `label_n_jobs`.
  - Added `parallel_backend`.
  - Aligned helper defaults with runner defaults.

- Updated `PretrainTD3LyapunovMPC.py`.
  - Default `DEFAULT_WORKER_BATCH_SIZE` is now `DEFAULT_PRETRAIN_BATCH_SIZE`, i.e. `8192`.
  - Added `DEFAULT_LABEL_N_JOBS = -1`.
  - Added `DEFAULT_PARALLEL_BACKEND = "loky"`.
  - Added CLI flags `--label-n-jobs` and `--parallel-backend`.

- Updated documentation.
  - `report/lmpc_td3_pretraining_process_2026-06-09.md`
  - `change-reports/2026-06-09_lmpc_pretraining_runner_defaults.md`
  - `change-reports/2026-06-09_lmpc_td3_pretraining_workflow.md`

## Safety Notes

- The parent process still owns the replay buffer.
- Live CVXPY problem objects are not passed into workers.
- Parallel workers return plain Python/NumPy diagnostics and transitions.
- `--label-n-jobs 1` or `--parallel-backend sequential` can be used for debugging or environments where process spawning is restricted.

## Validation

Static validation passed:

```powershell
python -m py_compile utils/lmpc_td3_workflow.py PretrainTD3LyapunovMPC.py
```

CLI validation passed in `rl-env`:

```powershell
python PretrainTD3LyapunovMPC.py --help
```

The help text shows `--worker-batch-size` default `8192`, `--label-n-jobs` default `-1`, and `--parallel-backend` default `loky`.

Tiny parallel smoke validation passed after allowing Windows `loky` worker process spawning:

```text
results/PretrainLMPCParallelSmoke/20260609_214340/
```

Command:

```powershell
python PretrainTD3LyapunovMPC.py --lmpc-samples 2 --steady-samples 1 --candidate-chunk-size 4 --worker-batch-size 2 --max-attempt-multiplier 20 --label-n-jobs 2 --parallel-backend loky --actor-epochs 1 --critic-epochs 1 --pretrain-batch-size 3 --device cpu --output-root results/PretrainLMPCParallelSmoke
```

Observed:

- accepted replay labels: `3`
- attempted candidates: `4`
- successful LMPC solves: `4`
- solve success rate: `1.0`
- parallel backend: `loky`
- label workers: `2`
- discarded successful solves after requested count: `1`
- actor BC loss entries: `1`
- critic TD loss entries: `1`
