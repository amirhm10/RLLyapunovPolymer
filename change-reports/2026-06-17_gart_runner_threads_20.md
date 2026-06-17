# Set GART Runner Thread Cap To 20

## Summary

Updated the active standalone GART-LMPC runner to request 20 math-library threads for faster full disturbance studies on the local 24-logical-processor workstation.

## Files Changed

- `GARTLyapunovMPC.py`
  - Changed `THREADS` from `4` to `20`.

## Runtime Effect

`GARTLyapunovMPC.py` calls `set_single_thread_env(THREADS)` before importing the experiment runner and again before launching the configured study. With `THREADS = 20`, the runner now sets:

- `OMP_NUM_THREADS=20`
- `OPENBLAS_NUM_THREADS=20`
- `MKL_NUM_THREADS=20`
- `VECLIB_MAXIMUM_THREADS=20`
- `NUMEXPR_NUM_THREADS=20`

This affects threaded numerical kernels used by the GART target-selector and GART-LMPC solve path. It does not create parallel closed-loop episodes by itself.

## Validation

Run:

```powershell
python -m py_compile GARTLyapunovMPC.py utils/gart_runtime.py
```
