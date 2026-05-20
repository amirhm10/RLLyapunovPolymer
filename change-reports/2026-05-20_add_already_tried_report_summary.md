# Add Project-Level Already-Tried Summary

## Summary

Added a final handoff section to `report/rl_agent_authority_bc_latest_analysis_2026-05-19.md` so future reviews can see what has already been implemented, tested, and deliberately changed before proposing next steps. The section was expanded from the latest RL setup to a project-level history covering target-selector, safety-filter, direct Lyapunov MPC, and safety-gate RL work.

## What Changed

- Added `Project History And Already-Tried Changes` at the end of the current RL report.
- Documented the active root entrypoints, result folder names, saved-agent behavior, direct LMPC settings, and RL safety-gate settings.
- Recorded the target-selector history: four-mode selector, refined Step A replacement, selector warm-start, and last-valid effective target backup.
- Recorded safety-filter and first-step-contraction work: canonical Lyapunov tolerance semantics, hard-acceptance debug split, upstream MPC first-step contraction, and earlier bounded/frozen target regularization.
- Recorded direct Lyapunov MPC work: frozen output-disturbance target solver, hard/soft modes, bounded/unbounded variants, mixed `u_prev`/`x_s` regularization, raw setpoint tracking, simplified direct objective, and target-quality/lexicographic/guard hooks.
- Recorded the already implemented agent-authority BC flow, soft handoff, MPC-only would-be activation diagnostic, and wall-clock timing instrumentation.
- Separated the latest analyzed reward run from the current next-run reward defaults.
- Captured exploration, policy-noise, discount-factor, maintenance, jitter, and Lyapunov epsilon decisions.
- Added a compact `Do Not Re-Suggest Without New Evidence` checklist for sharing the report with external assistants.

## Validation

- Regenerated the self-contained HTML report after the Markdown edit.
- Confirmed the HTML still embeds figures with `data:image/...;base64`.
- Ran `git diff --check` before commit.
