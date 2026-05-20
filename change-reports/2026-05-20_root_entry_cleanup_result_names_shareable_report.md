# Root Entry Cleanup, Result Names, And Shareable Report

## Summary

Cleaned the repository root so the active experiment workflow is script-first instead of notebook-first. The remaining root entrypoints are the direct Lyapunov MPC script, cold-start safety-gate RL script, and pretrained safety-gate RL script.

## Changes

- Renamed `DirectLyapunovMPC_FourMethodDisturbance.py` to `DirectLyapunovMPC.py`.
- Updated top-level result study names:
  - direct MPC: `results/directLyap/...`
  - cold-start RL: `results/ColdStart/...`
  - pretrained RL: `results/Pretrain/...`
- Moved all remaining root-level notebooks into `archive/` after clearing code-cell outputs and execution counts.
- Updated `AGENTS.md` so the active entrypoint list points to the three Python scripts.
- Extended `report/rl_agent_authority_bc_latest_analysis_2026-05-19.md` with a methodology-first explanation of the plant, scaling, direct LMPC path, MPC-only diagnostics, TD3 action flow, agent-authority BC, safety gate, handoff, reward, timing, and evaluation metrics.
- Added a self-contained HTML export at `report/rl_agent_authority_bc_latest_analysis_2026-05-19.html` with embedded report figures.

## Notes

- Per-case diagnostic names such as `mpc_only` and `lyap_mix_u0p1_x0p1_lex` were intentionally preserved.
- The analyzed historical result folders in the report were not renamed because they refer to already generated data.
- Historical reports were not broadly rewritten, even when they mention older notebook names.

## Validation

- Compile the three active scripts with `python -m py_compile`.
- Validate archived notebooks with `nbformat`.
- Confirm no root-level notebooks remain.
- Confirm active `study_name` values are `directLyap`, `ColdStart`, and `Pretrain`.
- Confirm the HTML report embeds figures with `data:image/...;base64`.
