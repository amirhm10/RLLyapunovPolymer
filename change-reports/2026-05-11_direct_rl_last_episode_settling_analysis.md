# Direct RL Last-Episode Settling Analysis

Date: 2026-05-11

## Summary

Added a research note analyzing the apparent last-episode settling issue in the direct safety-gate RL studies.

## Added artifacts

- [report/direct_rl_last_episode_settling_analysis_2026-05-11.md](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/direct_rl_last_episode_settling_analysis_2026-05-11.md>)
- [last_episode_settling_compare_2026-05-11.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/last_episode_settling_compare_2026-05-11.png>)

The generated figure is embedded directly in the report.

## Main conclusions captured in the report

- The last episode is structurally a test episode, so behavior noise should not be active there.
- The accessible saved runs do not support “last-episode exploration noise” as the explanation for the observed non-settling.
- Episode-level metrics for the bounded-hard case are comparable or slightly better in the later accessible runs, but tail behavior does change.
- The more likely cause is policy-quality or fallback-interaction changes entering the final deterministic evaluation, not noise injected during the last episode itself.

## Validation

- Inspected the train/test schedule in [utils/helpers.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/utils/helpers.py>).
- Inspected the phase/noise resolution logic in [Simulation/run_rl_lyapunov.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Simulation/run_rl_lyapunov.py>).
- Computed episode-200 metrics and last-episode tail errors from saved run bundles.
