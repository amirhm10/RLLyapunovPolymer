# OF-MPC Buffer Interrupt Handling

## Objective

Handle `KeyboardInterrupt` during OF-MPC expert-label generation instead of exiting with a traceback before any run metadata is written.

The observed interruption occurred in the first broad-label chunk:

```text
Processing chunk 1/20 (size=100000)
KeyboardInterrupt
```

At this point TD3 actor or critic training had not started. The interruption occurred while `joblib.Parallel` was solving MPC labels for the replay buffer.

## Change

- `PretrainTD3OffsetFreeMPC.py` now defaults to `--chunk-size 10000` instead of `100000`.
- The runner prints a clean interrupted status for all interrupted pretraining summaries.
- `utils/of_mpc_td3_workflow.py` writes `config.json` before expensive buffer generation begins.
- Buffer generation is now wrapped with `KeyboardInterrupt` handling.
- If interrupted during buffer generation, the run writes `summary.json` with:
  - `status = "interrupted_buffer_generation"`
  - elapsed and buffer-generation seconds
  - requested and completed sample counts
  - current replay-buffer stats
  - `checkpoint_path = null`
- If at least one full chunk had already completed, the current replay arrays are saved as:

```text
of_mpc_replay_partial.npz
```

## Interpretation

The OF-MPC label generation still solves the same one-step expert action problem used for replay construction. The generated TD3 transition remains:

$$
(s_k, a_k, r_k, s_{k+1}, d_k),
$$

where $a_k$ is the scaled OF-MPC action label and $r_k$ is the diagnostic replay reward. The TD3 actor cloning and critic TD objectives are unchanged.

The smaller default chunk size does not change labels. It only reduces the amount of work lost if the process is interrupted before a chunk is committed to the replay buffer.

## Validation

Passed syntax validation:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'codex-pycache-lyapunov-polymer'
& "C:\Users\HAMEDI\miniconda3\envs\rl\python.exe" -m py_compile "PretrainTD3OffsetFreeMPC.py" "utils\of_mpc_td3_workflow.py" "utils\td3_helpers.py"
```

Passed a monkeypatched interrupt smoke test that:

- injected one completed fake replay chunk
- raised `KeyboardInterrupt` during buffer generation
- verified `status = "interrupted_buffer_generation"`
- verified `config.json` and `summary.json` were written
- verified `of_mpc_replay_partial.npz` was saved
- exited without a Python traceback

## Practical Note

If the interrupt happens inside the first chunk before any chunk completes, the summary will still be written, but no partial replay archive can be saved because no complete replay transitions have been committed yet.
