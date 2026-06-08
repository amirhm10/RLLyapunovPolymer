# OF-MPC TD3 Pretraining Workflow

Date: 2026-06-08

## Summary

Added a repo-native offset-free MPC pretraining workflow for the TD3 agent. The new workflow preserves the historical OF-MPC expert-label method, uses Rawlings output-disturbance augmentation, and writes generated checkpoints under `results/PretrainOFMPC/` instead of `Data/`.

## Files Changed

- `PretrainTD3OffsetFreeMPC.py`
  - New root entrypoint for TD3 pretraining from OF-MPC labels.
  - Supports `smoke` and `legacy-full` presets.
  - Exposes CLI overrides for sample counts, chunk size, actor/critic epochs, seed, device, and output root.
  - Saves checkpoint, config, summary, and loss-array artifacts under `results/PretrainOFMPC/<timestamp>/`.
- `DirectLyapunovSafetyGateRL_Pretrained.py`
  - Added `PRETRAINED_TD3_AGENT_PATH` override.
  - Kept the default checkpoint as `Data/agent_2507171027.pkl`.
  - Resolves relative override paths from the repository root.
- `report/of_mpc_td3_pretraining_process_2026-06-08.md`
  - Documents the pretraining objective, coordinates, OF-MPC problem, replay-buffer construction, TD3 stages, run commands, limitations, and LMPC conversion path.

## Technical Notes

The new script uses:

- Rawlings output-disturbance augmentation with $B_d = 0$ and $C_d = I$.
- Broad legacy setpoint envelope `[[2.8, 320.0], [5.0, 326.0]]`.
- Physical input bounds `[71.6, 78.0]` to `[870.0, 670.0]`.
- OF-MPC horizons `NP = 9` and `NC = 3`.
- OF-MPC weights `Q = [5, 1]` and `R = [1, 1]`.
- TD3 dimensions `STATE_DIM = 13` and `ACTION_DIM = 2`.
- Actor and critic hidden layers `[512, 512, 512, 512, 512]`.

## Validation Plan

- Compile the touched Python entrypoints:

```powershell
python -m py_compile PretrainTD3OffsetFreeMPC.py DirectLyapunovSafetyGateRL_Pretrained.py
```

- Run a tiny smoke pretraining pass:

```powershell
python PretrainTD3OffsetFreeMPC.py --mpc-samples 32 --steady-samples 8 --chunk-size 16 --actor-epochs 1 --critic-epochs 1
```

- Instantiate the same TD3 architecture as the pretrained runner and load the smoke checkpoint.

## Validation Results

- `python -m py_compile PretrainTD3OffsetFreeMPC.py DirectLyapunovSafetyGateRL_Pretrained.py` passed.
- The tiny smoke pretraining run completed with 32 OF-MPC samples and 8 near-steady samples.
- The smoke run wrote:
  - `results/PretrainOFMPC/20260608_162205/of_mpc_pretrained_td3_20260608_162213.pkl`
  - `results/PretrainOFMPC/20260608_162205/config.json`
  - `results/PretrainOFMPC/20260608_162205/summary.json`
  - `results/PretrainOFMPC/20260608_162205/loss_arrays.json`
- The smoke checkpoint loaded successfully into the same `STATE_DIM = 13`, `ACTION_DIM = 2`, `[512, 512, 512, 512, 512]` TD3 architecture used by `DirectLyapunovSafetyGateRL_Pretrained.py`.

The first smoke attempt inside the sandbox failed when Windows denied joblib/loky multiprocessing pipe creation. The same command succeeded when rerun with permission to use multiprocessing outside the sandbox.

## Limitations

- This change migrates OF-MPC pretraining only. It does not yet generate LMPC expert labels.
- The smoke preset is a plumbing validation preset and is not expected to produce a useful controller.
- New checkpoints are intentionally not written under `Data/`.
