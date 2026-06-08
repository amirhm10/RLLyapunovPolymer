# OF-MPC TD3 Runner Split and Comparison

Date: 2026-06-08

## Summary

Refactored the OF-MPC TD3 pretraining migration into the `Polymer_example` structure: reusable helpers are separated from root runners, the pretraining runner saves checkpoints, and a new comparison runner loads a saved TD3 agent and compares it with OF-MPC.

## Changes

- Added `utils/of_mpc_td3_workflow.py`.
  - Centralizes Polymer CSTR setup, Rawlings OF-MPC system data, computed TD3 dimensions, TD3 agent construction, OF-MPC construction, artifact writing, latest-checkpoint resolution, and saved-agent comparison helpers.
  - Computes TD3 dimensions from `A_aug`, `B_aug`, and `C_aug` instead of hard-coding `STATE_DIM = 13` and `ACTION_DIM = 2`.
- Refactored `PretrainTD3OffsetFreeMPC.py`.
  - Removed `PRESETS`, `--preset`, `smoke`, and `legacy-full`.
  - Uses one full default workload matching the original `Polymer_example` sizes: 4.9M OF-MPC samples, 100k near-steady samples, chunk size 100k, 1000 actor epochs, and 500 critic epochs.
  - Keeps small smoke runs as explicit CLI overrides only.
- Added `ComparePretrainedTD3OffsetFreeMPC.py`.
  - Loads a saved TD3 checkpoint.
  - Runs nominal and disturbance comparisons against OF-MPC.
  - Generates cached OF-MPC baselines under `results/PretrainOFMPCComparison/baselines/` when missing.
  - Saves comparison metrics, pickles, and plots under `results/PretrainOFMPCComparison/<timestamp>/`.
- Updated `report/of_mpc_td3_pretraining_process_2026-06-08.md`.
  - Documents the helper/runner split, computed dimensions, one full default configuration, explicit smoke override, and comparison runner.

## Validation Plan

```powershell
python -m py_compile PretrainTD3OffsetFreeMPC.py ComparePretrainedTD3OffsetFreeMPC.py utils/of_mpc_td3_workflow.py
python PretrainTD3OffsetFreeMPC.py --help
python PretrainTD3OffsetFreeMPC.py --mpc-samples 32 --steady-samples 8 --chunk-size 16 --actor-epochs 1 --critic-epochs 1
python ComparePretrainedTD3OffsetFreeMPC.py --agent-path results/PretrainOFMPC/<latest>/of_mpc_pretrained_td3_<latest>.pkl --set-points-len 10 --modes nominal
```

## Validation Results

- `python -m py_compile PretrainTD3OffsetFreeMPC.py ComparePretrainedTD3OffsetFreeMPC.py utils/of_mpc_td3_workflow.py` passed.
- `python PretrainTD3OffsetFreeMPC.py --help` confirmed there is no `--preset`, no `smoke`, and no `legacy-full`. The visible defaults are 4,900,000 OF-MPC samples, 100,000 near-steady samples, chunk size 100,000, 1000 actor epochs, and 500 critic epochs.
- Explicit small pretraining smoke passed:
  - command: `python PretrainTD3OffsetFreeMPC.py --mpc-samples 32 --steady-samples 8 --chunk-size 16 --actor-epochs 1 --critic-epochs 1`
  - checkpoint: `results/PretrainOFMPC/20260608_165406/of_mpc_pretrained_td3_20260608_165414.pkl`
- Saved-agent comparison smoke passed:
  - command: `python ComparePretrainedTD3OffsetFreeMPC.py --agent-path results/PretrainOFMPC/20260608_165406/of_mpc_pretrained_td3_20260608_165414.pkl --set-points-len 10 --modes nominal`
  - summary: `results/PretrainOFMPCComparison/20260608_165430/summary.json`
  - generated OF-MPC baseline cache: `results/PretrainOFMPCComparison/baselines/mpc_results_nominal_n2_len10.pickle`

The first smoke pretraining attempt inside the sandbox failed because Windows denied joblib/loky multiprocessing pipe creation. The same command succeeded when rerun with permission for multiprocessing.

## Notes

- The workflow still uses OF-MPC expert labels. LMPC labels are intentionally left for the next migration.
- The actor and critic hidden layers remain `[512, 512, 512, 512, 512]` for current `Lyapunov_polymer` checkpoint compatibility.
- New generated checkpoints and comparison artifacts are written under `results/`, not `Data/`.
