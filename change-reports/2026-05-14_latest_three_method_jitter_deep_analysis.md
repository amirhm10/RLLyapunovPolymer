# Change Report: Latest Three-Method Jitter Deep Analysis

## What Changed

Added a new deep-dive report comparing the latest pretrained RL, cold-start RL, and direct no-RL Lyapunov runs:

- [report/latest_three_method_jitter_deep_analysis_2026-05-14.md](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/latest_three_method_jitter_deep_analysis_2026-05-14.md>)

Added three supporting figures:

- [report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_seg2_input_motion.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_seg2_input_motion.png>)
- [report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_seg2_error_decomposition.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_seg2_error_decomposition.png>)
- [report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_tail_summary.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_tail_summary.png>)

## Analysis Scope

The report studies:

- latest pretrained RL run: `20260513_232640`
- latest cold-start RL run: `20260513_232632`
- latest direct no-RL run: `20260513_191433`

It isolates jitter on the final constant-setpoint plateau and separates:

- base direct-method jitter
- extra RL-wrapper jitter
- solver-warning effects

## Main Conclusion

The report concludes that jitter is driven primarily by:

- moving admissible targets under bounded least-squares selection
- mismatch between contraction certification around `y_s` and tracking of raw `y_sp`
- RL accepted actions that hug the first-step contraction boundary
- memoryless switching between accepted RL and fallback MPC

`optimal_inaccurate` is not the primary source.

## Validation

Validation was analysis-only:

- recomputed steady-tail motion and error metrics from `arrays.npz`
- cross-checked mode counts and solver-status counts from `step_table.csv`
- generated the three figure files from the saved latest result bundles
