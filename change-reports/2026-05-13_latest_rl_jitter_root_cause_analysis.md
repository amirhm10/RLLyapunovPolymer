# Change Report: Latest RL Jitter Root-Cause Analysis

## What Changed

Added a focused analysis report for the latest completed RL direct-Lyapunov runs:

- [report/latest_rl_jitter_root_cause_analysis_2026-05-13.md](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/latest_rl_jitter_root_cause_analysis_2026-05-13.md>)

Added two supporting figures:

- [report/figures/2026-05-13_rl_jitter_root_cause/latest_rl_seg2_mechanism_timeseries.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-13_rl_jitter_root_cause/latest_rl_seg2_mechanism_timeseries.png>)
- [report/figures/2026-05-13_rl_jitter_root_cause/latest_rl_seg2_tail_summary.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-13_rl_jitter_root_cause/latest_rl_seg2_tail_summary.png>)

## Analysis Scope

The report studies the latest completed May 13, 2026 RL exports:

- pretrained RL: `20260513_191437`
- cold-start RL: `20260513_191435`

It compares the no-anchor and mixed-anchor selector variants and tests whether the observed jitter is mainly caused by `optimal_inaccurate` solver warnings.

## Main Conclusion

The analysis shows that `optimal_inaccurate` is not the primary jitter source. The dominant mechanism is:

- moving admissible targets in the no-anchor selector case
- mismatch between contraction certification around `(x_s,u_s,y_s)` and raw-setpoint tracking in the RL/fallback loop
- repeated short switching between accepted RL actions and fallback direct MPC

## Validation

Validation was done by:

- direct inspection of the saved `step_table.csv` and `arrays.npz` bundles
- recomputing tail metrics on the final constant-setpoint plateau
- generating the two figure files from the saved latest result bundles

No code path was changed in this task. This update is analysis-only.
