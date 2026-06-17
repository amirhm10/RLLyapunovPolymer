# Tighten GART-LMPC Target Input Rate

## Objective

Test whether the GART-LMPC settle-then-jump behavior is caused by excessive movement of the selected steady input target `u_s`.

## Change

`GARTLyapunovMPC.py` now overrides the GART target rate limits used by the root GART-LMPC experiment:

```python
DX_S_MAX_ABS = 0.05
DU_S_MAX_ABS = [0.05, 0.05]
DY_S_MAX_ABS = 1.0
```

Only `du_s_max_abs` is tightened relative to the previous final setting:

```python
du_s_max_abs = [0.998, 0.740]
```

The shared defaults are intentionally unchanged so RL runners are not affected before the GART-LMPC-only experiment is checked.

## Reasoning

The latest GART-LMPC run showed near-steady windows where the output error was small and the setpoint was unchanged, but `u_s` moved substantially while `y_s` barely changed. This is consistent with target-selection drift along a weakly output-visible steady-state direction. A tighter `du_s_max_abs` should force the selected target to move more gradually and make this failure mode easier to diagnose.

## Validation

Run:

```powershell
python -m py_compile GARTLyapunovMPC.py
```

Then rerun `GARTLyapunovMPC.py` and compare `target_rate_inf_max`, `mean_abs_delta_u`, output RMSE, and any target-unusable or solver failure counts against `results/GARTLMPC/20260616_210316`.

