# Finalize GART-LMPC Runner

## Summary

Cleaned the GART-LMPC experiment surface so the exposed runner now uses only the selected final controller:

- case/result name: `gartlmpc`
- result path: `results/GARTLMPC/<timestamp>/gartlmpc`
- objective: raw setpoint tracking
- Lyapunov mode: hard first-step contraction
- slack: not used by the final runner
- disturbance certificate: fixed symmetric bounded-rate update

## Final Parameters

```python
rho = 0.98
eps = 1.0e-3
dx_s_max_abs = 0.05
dy_s_max_abs = 1.0
input_headroom_frac = 0.01
d_rate_scale = 1.0
adaptive_rate_enabled = False
eta_y = 0.0
eta_u = 0.0
disable_u_mid_tiebreak = True
disable_x_smoothing = True
disable_y_smoothing = True
```

## Code Changes

- Simplified `GARTLyapunovMPC.py` to a single editable runner config block.
- Removed target-only toggles, full/confirm smoke gates, resource caps, and comparison case menus from the root runner.
- Simplified `experiments/run_gart_target_selector_study.py` so `run_closed_loop(...)` always runs the final `gartlmpc` case.
- Removed experiment-facing paths for old governed-reference baseline, target-only diagnostics, observer replay, ablations, mixed objective cases, soft/slack cases, adaptive disturbance cases, asymmetric disturbance cases, and no-`dx_s` cases.
- Kept reusable core adaptive/mixed/soft implementation internals in `Lyapunov/gart_target.py` and `Lyapunov/gart_lmpc.py`.
- Updated `GARTMPCConfig` and `make_gart_mpc_config(...)` defaults to final raw/hard values.
- Removed the obsolete observer-replay unit test that referenced a removed experiment helper.
- Rewrote `report/gart_lmpc_design_notes.md` around the final selected method.

## Rationale

The final disturbed-run evidence favored the raw GART-LMPC path with a hard contraction certificate and a finite but not overly tight target-state motion bound. The selected `dx_s_max_abs=0.05`, `dy_s_max_abs=1.0`, `rho=0.98`, and `eps=1e-3` setting avoided the late jumps seen with `eps=1e-4` and the lag seen with tighter output-motion or adaptive disturbance-rate settings.

The removed runner variants were useful during tuning, but keeping them exposed made it too easy to launch stale or scientifically rejected cases by accident. The core code remains available for reproducibility, while the active runner now represents the final method directly.

## Validation

- Passed:
  `python -m py_compile GARTLyapunovMPC.py experiments/run_gart_target_selector_study.py utils/gart_defaults.py Lyapunov/gart_target.py Lyapunov/gart_lmpc.py`
- Blocked:
  `python -m pytest tests/test_gart_target.py`

`pytest` is not installed in the local `rlenv` environment:

```text
C:\Users\hamed\miniconda3\envs\rlenv\python.exe: No module named pytest
```

The full final closed-loop runner was not launched during cleanup validation because the editable default is now the real disturbed closed-loop study rather than a smoke check.
