# GART Probe Final Default Consistency

## Change

- Aligned `GARTLyapunovMPC_ExplorationProbe.py` with the final GART tuning block used by the direct GART runner and shared defaults:
  - `RHO_LYAP = 0.99`
  - `TARGET_WY_DIAG = [2.0, 1.0]`
  - `INPUT_HEADROOM_FRAC = 0.05` verified in the working configuration.
- Kept the probe input excitation at `INPUT_EXPLORATION_STD = [0.05, 0.05]` for the exploration probe path.

## Evidence

The shared final GART target overrides in `utils/gart_defaults.py` use:

```python
"input_headroom_frac": 0.05,
"dx_s_max_abs": 0.05,
"du_s_max_abs": [0.2, 0.2],
"dy_s_max_abs": 0.25,
"d_rate_scale": 0.25,
"alpha_d": 0.05,
"W_u_smooth_diag": [2.0, 2.0],
"Wy_diag": [2.0, 1.0],
```

The online runner path pulls these values through `GART_FINAL_TARGET_CONFIG_OVERRIDES`.

## Validation

- `python -m py_compile GARTLyapunovMPC_ExplorationProbe.py` passed with `PYTHONPYCACHEPREFIX` pointed at the system temp directory.
