# Direct Lyapunov Plot Units And Nominal Default

## Summary

Fixed the direct Lyapunov MPC plotting path so future notebook runs do not mix
physical measured outputs with scaled-deviation target traces.

## Changes

- Updated `Lyapunov/direct_lyapunov_mpc.py` so `01_outputs_vs_targets.png`
  plots measured `y`, scheduled `y_sp`, selected `y_s`, and stage target in
  the same physical output units.
- Added explicit titles to direct diagnostic plots and comparison overlay plots.
- Updated `Plotting_fns/mpc_plot_fns.py` with titles for shared output/input
  full-horizon and last-window MPC plots.
- Changed the direct four-scenario notebook default plant run mode to
  `plant_mode = "nominal"` with `disturbance_after_step = False`.
- Added corrected physical-unit SVG output-target figures to the supervisor
  report for the latest saved run.

## Validation

- `python -m py_compile Lyapunov/direct_lyapunov_mpc.py Plotting_fns/mpc_plot_fns.py`
- notebook JSON load check for `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
- report image-link validation
- SVG XML parse check for the newly added physical-unit figures
