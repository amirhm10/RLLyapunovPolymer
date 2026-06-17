# Remove GART Runtime Thread Controls

## Summary

Removed the explicit GART runtime control that set numerical-library thread environment variables from the standalone GART-LMPC runner.

## Files Changed

- `GARTLyapunovMPC.py`
  - Removed the `THREADS` constant.
  - Removed both calls to `set_single_thread_env(...)`.
  - Removed the no-longer-needed import from `utils.gart_runtime`.
- `utils/gart_runtime.py`
  - Removed `set_single_thread_env(...)`.
  - Removed the unused `os` import.

## Runtime Effect

The standalone GART-LMPC runner no longer changes process-level numerical-library environment variables. Threading behavior is now left to the active Python environment, solver packages, and any external shell settings.

The existing `ResourceGuard` limits remain unchanged.

## Validation

Run:

```powershell
python -c "import ast, pathlib; files=['GARTLyapunovMPC.py','utils/gart_runtime.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8'), filename=f) for f in files]; print('ast parse ok', len(files))"
```
