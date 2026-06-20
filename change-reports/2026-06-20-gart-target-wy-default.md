# GART Target Wy Default

## Objective

Set the final GART target-selector output weight default to
`TARGET_WY_DIAG = [2.0, 1.0]` for the active GART-LMPC and online TD3 paths.

## Files Changed

- `utils/gart_defaults.py`
- `GARTLyapunovMPC.py`

## Change

The shared final GART target overrides now use:

```python
"Wy_diag": [2.0, 1.0]
```

The main editable GART-LMPC runner now uses:

```python
TARGET_WY_DIAG = [2.0, 1.0]
```

`GARTLyapunovMPC_ExplorationProbe.py` already had this value and was not
changed by this update.

## Mathematical Interpretation

The GART target selector penalizes output-target mismatch with:

$$
J_y = (y_s - y_{\mathrm{sp}})^\top W_y (y_s - y_{\mathrm{sp}}).
$$

Changing `Wy_diag` from `[1.0, 1.0]` to `[2.0, 1.0]` doubles the relative
weight on the first output target mismatch while leaving the second output
weight unchanged.

## Validation

Passed:

```powershell
python -m py_compile utils/gart_defaults.py GARTLyapunovMPC.py GARTLyapunovMPC_ExplorationProbe.py utils/online_disturbance_runner.py
```

The successful local run redirected `PYTHONPYCACHEPREFIX` to a Windows temp
directory to avoid the known OneDrive `.validation-pyc` pycache permission issue.
