# Plot Reward Without Penalty

## Summary

- Updated safety-gate debug reward plots to always include `reward_no_penalty` beside the actual TD3 training reward.
- Updated the episode-average reward summary to plot both penalized training reward and penalty-free reward averages.
- Kept existing comparison-level `reward_no_penalty` exports intact.

## Why

Safety-gate runs subtract fallback/event penalties from the actual TD3 training reward, while no-gate runs do not. Plotting only the actual reward can make the safety-gate controller look worse even when the underlying tracking reward is closer. The new plots show both quantities so the penalty contribution is visually explicit.

## Validation

- Passed: `python -m py_compile Lyapunov/safety_debug.py`.
- Passed: `python OnlineTD3_OFMPCPretrained_NoSafetyGate.py --episodes 1 --set-points-len 5 --save-plots`.
- Confirmed reward plot files were written:
  - `results/OnlineTD3_OFMPCPretrained_NoSafetyGate/20260610_200031/onlinetd3_ofmpcpretrained_nosafetygate/plots/reward_trace.png`
  - `results/OnlineTD3_OFMPCPretrained_NoSafetyGate/20260610_200031/onlinetd3_ofmpcpretrained_nosafetygate/plots/reward_average_summary.png`
