# Latest Online TD3 Analysis Report

## Objective

Create a dated comparison report for the latest completed active online TD3
runners, with figures and metric tables for tracking, safety-gate activity,
phase behavior, and final-episode trajectories.

## Artifacts Added

- `analysis/online_td3_latest_analysis_2026_06_19.py`
- `report/online_td3_latest_analysis_2026-06-19.md`
- `report/figures/2026-06-19_online_td3_latest/summary_metrics.csv`
- `report/figures/2026-06-19_online_td3_latest/phase_metrics.csv`
- `report/figures/2026-06-19_online_td3_latest/episode_metrics.csv`
- `report/figures/2026-06-19_online_td3_latest/pending_runs.csv`
- `report/figures/2026-06-19_online_td3_latest/summary_bar_metrics.png`
- `report/figures/2026-06-19_online_td3_latest/phase_metrics.png`
- `report/figures/2026-06-19_online_td3_latest/episode_trends.png`
- `report/figures/2026-06-19_online_td3_latest/final_episode_tracking.png`
- `report/figures/2026-06-19_online_td3_latest/final_episode_inputs.png`

## Data Selection

The report uses the latest completed non-diagnostic 300-episode runs:

- `results/OnlineTD3_ColdStart_SafetyGate/20260618_191134`
- `results/OnlineTD3_ColdStart_NoSafetyGate/20260618_191130`
- `results/OnlineTD3_OFMPCPretrained_SafetyGate/20260618_191141`
- `results/OnlineTD3_OFMPCPretrained_NoSafetyGate/20260618_191137`

Four newer `20260619_1316xx` result folders were detected but did not yet have
final `step_table.csv` or `episode_table.csv` exports, so the report lists them
as pending and does not mix them into the completed-run comparison.

## Main Finding

The latest completed runs still show a performance-versus-protection tradeoff:
the no-gate cases have better `reward_no_penalty` and output RMSE, while the
gate cases replace a subset of candidate actions with the GART-LMPC fallback.
The OF-MPC-pretrained no-gate run is the strongest completed result by average
tracking metrics.

## Validation

- Regenerated the report and figures with:
  - `C:\Users\hamediaa\.conda\envs\rl-env\python.exe -B analysis\online_td3_latest_analysis_2026_06_19.py`
- Visually inspected the summary, phase, and final-episode tracking figures.
- Confirmed the report lists the current `20260619_1316xx` runs as pending
  rather than completed.
