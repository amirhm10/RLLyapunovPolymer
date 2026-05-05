## Summary

Fixed a Windows export-path failure in the RL safety-gate notebook debug artifact saver.

The failure was not caused by plotting logic or missing directories inside Matplotlib. The generated export tree for paper plots could exceed the Windows path-length limit when the study root already contained a timestamped folder and `save_safety_filter_debug_artifacts(...)` added a second nested timestamp plus verbose case names.

## Root Cause

The failing path shape was:

- `.../<study timestamp>/<case name>/<case timestamp>/paper_plots/safety_selector/first_step_contraction_diagnostics.png`

For the reported direct safety-gate RL case, that path reached length `260`, which is enough to trigger `FileNotFoundError` on Windows in normal path mode.

## Change

Updated [Lyapunov/safety_debug.py](/c:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/safety_debug.py) so `save_safety_filter_debug_artifacts(...)` now selects a shorter output directory automatically when the projected artifact tree would exceed a Windows-safe path budget.

The saver now:

- keeps the original `directory/prefix_name/timestamp` layout when it is safe
- falls back to shorter layouts on Windows when needed
- preserves the existing notebook API and artifact contents

## Validation

- `python -m py_compile Lyapunov/safety_debug.py`
- direct path-budget check for the reported study/case path
  - selected output dir: `.../sf_41d4b0ae_040114`
  - projected max artifact path length: `230`

## Expected Notebook Impact

The direct safety-gate RL notebooks should now save debug bundles and paper plots successfully on Windows even for long study and case names. The only user-visible change is that some debug export folders may use a shorter auto-generated case directory name when needed to stay within the filesystem limit.
