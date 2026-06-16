# Expose GART Runner Rho And Epsilon

## Summary

`GARTLyapunovMPC.py` now exposes the first-step Lyapunov contraction settings
as top-level editable runner values:

```python
RHO_LYAP = GART_FINAL_RHO_LYAP
LYAP_EPS = GART_FINAL_LYAP_EPS
```

Changing these values in the root runner now updates both:

- the GART target governor and contraction probe through `target_overrides`
- the GART-LMPC solver contraction constraint through `mpc_overrides`

## Validation

Completed checks:

```powershell
python -m py_compile GARTLyapunovMPC.py experiments/run_gart_target_selector_study.py
& "C:\Users\hamed\miniconda3\envs\rlenv\python.exe" -c "<assert runner target/mpc overrides use RHO_LYAP and LYAP_EPS>"
```
