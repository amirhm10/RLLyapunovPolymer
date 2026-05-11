# Safety Debug Output Scaling Unification

## Why
- Some safety-debug plots mixed physical output units with scaled deviation quantities.
- In particular, the decomposition plots compared `y_s` in physical units against `Cx_s` and `Cd d_s` still in scaled coordinates.
- This made several direct RL safety-gate figures visually inconsistent and scientifically misleading.

## What Changed
- Updated [`Lyapunov/safety_debug.py`](../Lyapunov/safety_debug.py) so `build_safety_filter_run_bundle(...)` now stores:
  - physical output trajectories for `y_sp`, `y_s`, `r_s`, and `yhat`
  - physical output-deviation trajectories for `y_s`, `r_s`, `Cx_s`, and `Cd d_s`
- Added shared helper conversions for:
  - scaled output deviations to physical absolute outputs
  - scaled output deviations to physical deviations from steady state
- Updated the safety-debug plotting path to use:
  - physical absolute units for output overlays such as `y`, `y_sp`, `y_s`, and `yhat`
  - physical deviation units for decomposition plots comparing `Cx_s`, `Cd d_s`, and `y_s - y_ss`

## Effect On Plots
- `outputs_vs_ysp_vs_ys*.png` now remain in physical output units consistently.
- `ys_decomposition_summary*.png` and the per-channel decomposition folders now use one consistent physical deviation basis instead of mixing physical absolute `y_s` with scaled `Cx_s` and `Cd d_s`.
- The decomposition legend now uses `y_s_dev_*` to reflect that the plotted target term is the physical deviation part compatible with the decomposition.

## Validation
- Compiled `Lyapunov/safety_debug.py` in memory with Python `compile(...)`.
- Verified that the new physical-store keys are present and consumed by the plotting path.
- Did not rerun the full notebooks end-to-end in this change.

## Notes
- This update is limited to the safety-debug plotting stack used by the RL safety-gate notebooks.
- State-space plots such as `xhat` versus `x_s` remain in their native internal coordinates because they are not output-unit plots.
