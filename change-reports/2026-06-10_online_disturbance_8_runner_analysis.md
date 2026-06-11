# Change Report: Online Disturbance 8-Runner Result Analysis

Date: 2026-06-10

## Summary

Added a reproducible analysis script and extended Markdown report for the latest eight disturbance-only runner results:

- LMPC-pretrained TD3 with and without safety gate
- OF-MPC-pretrained TD3 with and without safety gate
- cold-start TD3 with and without safety gate
- Direct LMPC disturbance baseline
- OF-MPC disturbance baseline

The report compares logged reward, `reward_no_penalty`, physical output RMSE, actual safety-gate interventions, monitor-only Direct LMPC diagnostic failures, setpoint-block behavior, and historical target-selector/monitor activation.

## Files Added

- `analysis/online_disturbance_8_runner_analysis.py`
- `report/online_disturbance_8_runner_analysis_2026-06-10.md`
- `report/figures/2026-06-10_online_disturbance_8_runner_analysis/*`

## Main Findings

- The latest online TD3 configs do not show the previous setpoint-scaling mismatch. The saved TD3 scaler covers the disturbance setpoints.
- Safety-gate logged reward is lower partly because fallback penalties are included. The report separates logged reward from `reward_no_penalty`.
- No-gate TD3 policies track best late in training, but retain nonzero Direct LMPC monitor failures.
- Safety-gate policies remove monitor failures by applying fallback, but fallback and hold-prev events add penalty and can slightly degrade tracking.
- Current governed-reference diagnostics are much less active than several older bounded/intermediate governed selector runs.
- Historical full-length bounded/governed baselines have similar physical output RMSE to the current baselines, so reverting the target selector is not yet justified without a controlled ablation.

## Generated Artifacts

Figures:

- `latest_tail_performance_overview.png`
- `safety_activity_and_penalty.png`
- `safety_gate_fallback_breakdown.png`
- `episode_reward_no_penalty_trends.png`
- `episode_reward_no_penalty_trends_zoom.png`
- `episode_output_rmse_trends.png`
- `episode_output_rmse_trends_zoom.png`
- `last_episode_output_tracking.png`
- `historical_selector_monitor_context.png`

Tables:

- `latest_metrics.csv`
- `online_phase_metrics.csv`
- `setpoint_block_metrics.csv`
- `historical_selector_context.csv`
- `run_manifest.json`

## Validation

Ran:

```powershell
python analysis/online_disturbance_8_runner_analysis.py
python -m py_compile analysis/online_disturbance_8_runner_analysis.py
```

Both completed successfully.
