# OF-MPC Joblib Loky Cleanup

## Objective

Reduce `joblib`/`loky` `resource_tracker` warnings that can appear at shutdown after OF-MPC TD3 pretraining label generation.

The observed log showed actor behavioral cloning progressing normally through epoch 216, followed by shutdown warnings such as:

```text
resource_tracker: There appear to be 42 leaked folder objects to clean up at shutdown
resource_tracker: ... joblib_memmapping_folder_... FileNotFoundError
```

There was no Python traceback in the pasted log.

## Interpretation

The actor BC objective was still being optimized normally:

$$
\min_\theta \|\pi_\theta(s) - u_{\mathrm{MPC}}\|^2.
$$

The warnings are emitted by joblib's process-based `loky` backend during Python shutdown, not by the TD3 loss computation. They are consistent with reusable worker resources or temporary folders lingering after the earlier parallel expert-label generation stage.

The latest local run directory contained periodic actor checkpoints through epoch 200, but no `summary.json` or interrupted checkpoint. That suggests the process was killed or stopped outside the script's normal `KeyboardInterrupt` handler.

## Change

`utils/td3_helpers.py` now:

- adds `_shutdown_loky_executor()` to explicitly shut down joblib's reusable `loky` executor
- calls that cleanup in a `finally` block after broad OF-MPC sample generation
- calls the same cleanup after near-steady OF-MPC sample generation

This does not change the generated MPC labels, replay tuples, actor cloning loss, critic TD target, or online TD3 behavior.

## Validation

Passed syntax validation:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'codex-pycache-lyapunov-polymer'
& "C:\Users\HAMEDI\miniconda3\envs\rl\python.exe" -m py_compile "utils\td3_helpers.py"
```

Passed a small `joblib.Parallel(..., backend="loky")` worker-spawn and explicit-shutdown smoke test after allowing Windows worker processes.

## Practical Notes

The most recent pasted run stopped after actor epoch 216. The latest safe artifact in that run appears to be:

```text
results/PretrainOFMPC/20260621_155559/of_mpc_pretrained_td3_partial_actor_bc_ep0200_20260621_165409.pkl
```

If Python is hard-killed by the IDE or OS, no cleanup code can write a final `summary.json`. Reducing `--checkpoint-interval-epochs` can lower the maximum lost work between checkpoints.
