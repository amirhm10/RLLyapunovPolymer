# Agent-Authority BC Latest Analysis Report

## Summary

Created a new report for the latest cold-start RL, pretrained RL, and direct Lyapunov MPC reruns after the agent-authority behavioral cloning update.

## Outputs

- Added `report/rl_agent_authority_bc_latest_analysis_2026-05-19.md`.
- Generated report figures under `report/figures/2026-05-19_agent_authority_bc_latest_analysis/`.
- Included full-horizon performance, wall-clock timing, RL authority diagnostics, episode reward and fallback trends, tail-offset comparison, last-episode tracking, and MPC-only would-be activation diagnostics.

## Main Finding

Pretrained RL remains better on full-horizon reward and RMSE, while cold-start RL is better on safety-gate authority and final-tail temperature offset. Cold start has fewer safety interventions, smaller fallback penalty, and smaller actor-versus-executed action gap.

## Validation

- Loaded the latest result bundles from the three new run folders.
- Recomputed tail offset and episode diagnostics from saved arrays and episode tables.
- Checked Markdown table formatting and figure links.
