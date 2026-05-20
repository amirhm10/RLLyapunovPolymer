# Expand RL Methodology Mathematics

## Summary

Expanded the latest agent-authority BC report methodology from a compact overview into a detailed calculation-level description.

## Changes

- Added equations for physical-to-scaled coordinate conversion, scaled deviations, and inverse scaling.
- Added the output-disturbance augmented model and observer update used in the rollout.
- Added direct target equations for the steady target, bounded fallback target, and optional regularization terms.
- Added the direct LMPC objective, input constraints, Lyapunov value, first-step contraction condition, and contraction margin.
- Added TD3 observation construction, actor action mapping, inverse action mapping, BC demo loss, replay-buffer transition, handoff blend, and safety-correction gap.
- Added reward equations directly in the methodology section so the report explains the learning signal before showing results.
- Regenerated the self-contained HTML report with embedded figures and MathJax support for equation rendering.

## Validation

- Confirm the HTML report embeds all current figures.
- Confirm the HTML report includes MathJax configuration.
- Run Markdown/HTML consistency checks for local figure references and `data:image` embeds.
- Run `git diff --check`.
