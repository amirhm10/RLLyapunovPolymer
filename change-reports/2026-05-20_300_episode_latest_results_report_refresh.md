# 300-Episode Latest Results Report Refresh

## Summary

Updated the active RL/direct Lyapunov analysis report for the latest matched `300`-episode non-evaluation runs.

## Updated Inputs

- Cold start: `results/ColdStart/20260520_165642`
- Pretrained: `results/Pretrain/20260520_165645`
- Direct LMPC: `results/directLyap/20260520_165653`

## Report Changes

- Rewrote the results section of `report/rl_agent_authority_bc_latest_analysis_2026-05-19.md`.
- Regenerated local analysis figures under `report/figures/2026-05-20_300_episode_latest_analysis/`.
- Regenerated the self-contained HTML export at `report/rl_agent_authority_bc_latest_analysis_2026-05-19.html`.
- Updated the project-history handoff notes to reflect `n_episodes = 300`.

## Main Findings Captured

- Pretrained RL remains the strongest learned controller on full-horizon RMSE and reward.
- Cold-start RL improves with the longer run and catches up late, but the early BC phase still hurts full-horizon results.
- MPC-only diagnostics still beat the corresponding safety-gated RL cases on full-horizon tracking.
- Pretrained MPC-only still has the best raw tracking and the highest would-be Lyapunov activation rate.
- Direct LMPC has good final-tail offset but poor full-horizon raw-setpoint RMSE and about `2x` the per-step runtime of safety-gated RL.

## Validation

- Confirmed all Markdown image links exist locally.
- Confirmed the HTML export contains embedded `data:image/...;base64` figures.
- Ran `git diff --check`.

## Notes

- The local figure folder is ignored by `.gitignore`; the committed HTML contains embedded figures for sharing.
- Existing unrelated local changes were not included.
