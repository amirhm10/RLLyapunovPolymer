# Refresh Gamma 0.99 / Epsilon 1e-2 Results Report

## Summary

Updated the active RL/direct Lyapunov analysis report using the newest `300`-episode runs:

- Cold start: `results/ColdStart/20260520_204513`
- Pretrained: `results/Pretrain/20260520_205230`
- Direct LMPC: `results/directLyap/20260520_204510`

The refreshed report now reflects the current setup with `GAMMA = 0.99`, `lyap_eps = 1e-2`, `fallback_event_penalty = 10.0`, and the restored final RL evaluation episode.

## Main Report Updates

- Rewrote the full-horizon result table with the latest reward, RMSE, and wall-clock metrics.
- Updated RL authority diagnostics, phase breakdowns, fallback/intervention counts, and final evaluation episode metrics.
- Replaced tail-offset, MPC-only would-be activation, reward-penalty, and target-diagnostic interpretations with the newest results.
- Updated the project-history section so future reviewers do not re-suggest already-tested discount, epsilon, fallback-penalty, exploration, and reward changes.
- Regenerated the self-contained HTML report with embedded figures for sharing.

## Key Findings Captured

- Pretrained RL remains the best learned controller, but MPC-only still has better full-horizon raw tracking.
- Cold RL and pretrained RL both improved modestly relative to the previous `300`-episode report.
- Relaxing `lyap_eps` from `1e-3` to `1e-2` reduced fallback and MPC-only would-be activation rates.
- Direct LMPC still settles well in the final tail but has worse full-horizon raw-setpoint RMSE and slower runtime than safety-gated RL.
- The unresolved target-selector tension remains: raw-setpoint tracking and Lyapunov contraction compatibility still pull in different directions.

## Validation

- Regenerated `report/rl_agent_authority_bc_latest_analysis_2026-05-19.html` with embedded base64 images.
- Checked Markdown image links resolve locally before HTML generation.
- Confirmed the HTML export embeds 10 figures.
- Planned validation before commit: `git diff --check`.
