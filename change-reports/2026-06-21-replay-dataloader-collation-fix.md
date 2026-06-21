# Replay DataLoader Collation Fix

## Objective

Fix the OF-MPC TD3 pretraining crash where PyTorch `DataLoader` failed while collating replay-buffer batches during actor behavioral cloning.

The failure occurred before the TD3 update step, inside PyTorch's default collation path, so the fix targets replay dataset formatting rather than controller logic or loss functions.

## Change

`utils/td3_helpers.py` now normalizes `ReplayDataset` inputs at construction time:

- states, actions, rewards, next states, and done flags are converted to contiguous `float32` NumPy arrays
- scalar reward and done fields are flattened to one-dimensional arrays
- first-dimension consistency is checked before training starts
- the returned tuple still includes `(state, action, reward, next_state, done)` for critic TD training

This keeps the critic transition data intact while avoiding brittle NumPy scalar/object dtype behavior in `torch.utils.data.default_collate`.

## Interpretation

The actor cloning objective remains:

$$
\min_\theta \|\pi_\theta(s) - u_{\mathrm{MPC}}\|^2.
$$

The critic warm-up still uses:

$$
y = r + \gamma (1-d) Q_{\bar{\phi}}(s^+, \pi_{\bar{\theta}}(s^+)).
$$

Only the data packaging between the replay arrays and PyTorch mini-batches changed.

## Validation

Passed:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'codex-pycache-lyapunov-polymer'
& "C:\Users\HAMEDI\miniconda3\envs\rl\python.exe" -m py_compile "utils\td3_helpers.py"
```

Passed a `ReplayDataset` plus `DataLoader` smoke test with:

- float64 state/action/next-state inputs
- object-dtype reward scalars
- boolean NumPy done flags
- shuffled mini-batches containing the full five-field transition tuple

## Notes

A direct `py_compile` attempt into the local `utils/__pycache__` directory hit a Windows access-denied rename error, which is consistent with OneDrive or another file watcher locking bytecode files. Redirecting `PYTHONPYCACHEPREFIX` avoided that filesystem issue.

The original training traceback still points to PyTorch collation, not to checkpoint file I/O, so the primary runtime fix is dtype normalization in `ReplayDataset`.
