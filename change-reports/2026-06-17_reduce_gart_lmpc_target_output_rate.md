# Reduce GART-LMPC Target Output Rate

Date: 2026-06-17

## Objective

Test whether limiting the governed target output movement reduces the late-run input jumps seen in the latest GART-LMPC disturbance runs.

## Change

`GARTLyapunovMPC.py` now sets the local GART-LMPC runner override:

```python
DY_S_MAX_ABS = 0.25
```

This replaces the previous local value of `1.0`. The existing local `dx_s_max_abs = 0.05` and `du_s_max_abs = [0.05, 0.05]` settings are unchanged.

## Rationale

The latest tightened-input run showed that the steady-input target rate cap was active, but large applied input jumps still occurred. The diagnostic traces showed repeated `dy_s` and disturbance-target changes near `0.69`, so this experiment caps the target output movement directly.

## Expected Diagnostic Signal

The next GART-LMPC run should be checked for:

- reduced `target_rate_inf_max` and `dc_rate_inf_max`
- fewer large `dy_s` spikes near settled operation
- lower applied-input jump maxima
- no increase in `hold_previous_rate`
- no worsening of target-reference or output-reference error from excessive target lag

## Risk

If `DY_S_MAX_ABS = 0.25` is too small, the governed target may lag the raw setpoint or disturbance estimate more strongly. In that case, the controller may hold previous targets more often or produce larger tracking offsets despite smoother target motion.
