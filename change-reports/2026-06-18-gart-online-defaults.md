# GART Online Default Tuning

## Objective

Propagate the current GART-LMPC runner tuning to the shared final GART defaults used by the online TD3 runners, and keep the GART-LMPC exploration probe aligned with the same target-selector geometry.

## Files Changed

- `utils/gart_defaults.py`
- `GARTLyapunovMPC.py`
- `GARTLyapunovMPC_ExplorationProbe.py`

## New Shared Final GART Target Defaults

The shared `GART_FINAL_TARGET_OVERRIDES` now uses:

```python
input_headroom_frac = 0.05
primary_tol_rel = 1.0e-4
dx_s_max_abs = 0.05
du_s_max_abs = [0.2, 0.2]
dy_s_max_abs = 0.25
d_rate_scale = 0.25
alpha_d = 0.05
W_u_smooth_diag = [2.0, 2.0]
Wy_diag = [1.0, 1.0]
```

These defaults are consumed by `utils.online_disturbance_runner` for the online TD3 GART target selector, GART-LMPC BC teacher, and GART-LMPC fallback controller.

## Notes

- The safety-gate runners still use `PROJECTION_BACKEND = "direct_accept_or_fallback"`, so unsafe TD3 actions fallback to GART-LMPC rather than Section 16 QCQP projection.
- The exploration probe keeps explicit local constants, but its headroom was updated to match the current GART-LMPC runner.
- The local quick-run `N_TESTS = 2` edit in `GARTLyapunovMPC.py` was intentionally left out of the committed change.

## Validation

Passed:

```powershell
python -m py_compile utils/gart_defaults.py GARTLyapunovMPC.py GARTLyapunovMPC_ExplorationProbe.py utils/online_disturbance_runner.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py OnlineTD3_ColdStart_NoSafetyGate.py
```

The successful local run redirected `PYTHONPYCACHEPREFIX` to a temp directory to avoid a Windows/OneDrive pycache permission issue under `.validation-pyc`.
