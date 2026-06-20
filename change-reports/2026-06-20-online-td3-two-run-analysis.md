# Online TD3 Two-Run Analysis Report

## Summary
- Added a reproducible analysis script for the two latest completed executions of each active online TD3 runner.
- Generated a Markdown report with reward, tracking, intervention, phase, and late-training comparisons.
- Saved supporting metric CSVs and figures under a dated report figure directory.

## Files Added
- `analysis/online_td3_two_run_analysis_2026_06_20.py`
- `report/online_td3_two_run_analysis_2026-06-20.md`
- `report/figures/2026-06-20_online_td3_two_run/`

## Data Selection
The script selected the two latest completed 300-episode runs for each active runner:

- `OnlineTD3_ColdStart_SafetyGate`
- `OnlineTD3_ColdStart_NoSafetyGate`
- `OnlineTD3_OFMPCPretrained_SafetyGate`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate`

Each selected run required `record.json`, `run_summary.json`, `summary.json`, `episode_table.csv`, and `arrays.npz`.

## Main Finding
The two executions per runner used the same configured seed, `123`, and produced identical episode tables and selected trajectory arrays. The report therefore treats the repeated runs as deterministic reproducibility checks, not independent seed replicates.

Across the selected trajectories:

- OF-MPC pretraining improved reward and tracking relative to cold start.
- OF-MPC pretraining reduced intervention burden in active-gate runs.
- No-gate cases remained useful nominal-performance upper bounds, but they had nonzero diagnostic unsafe rates.
- The strongest methodological claim is the combined one: pretraining keeps the policy closer to the safe controller manifold, while the safety gate handles residual unsafe candidates.

## Validation
- Ran the analysis script successfully:

```powershell
python analysis/online_td3_two_run_analysis_2026_06_20.py
```

- Visually inspected the aggregate metric, episode trend, and final-episode tracking figures.
- Syntax validation was run with bytecode output redirected to `%TEMP%` to avoid the Windows/OneDrive `__pycache__` permission issue.
