# 2026-04-30 Latest Lyapunov Results Synthesis

## Why This Change Was Made

The repository had result artifacts for the latest direct Lyapunov study, the selector-term ablation study, and the RL safety-filter study, but no single report that:

- compares them coherently
- explains the mathematical differences between the architectures
- answers why `objective_zero` can still work better than the current direct path
- records the latest direct ten-scenario findings instead of the older four-scenario description

The `research-result-loop` skill was also updated so future result-analysis tasks explicitly use a control-specialist lens rather than stopping at generic ML commentary.

## What Was Added

- New synthesis report:
  - `report/latest_lyapunov_results_synthesis_2026-04-30.md`
- New dedicated figure folder for the report:
  - `report/figures/2026-04-30_lyapunov_results_synthesis/`
- Skill update:
  - `.agents/skills/research-result-loop/SKILL.md`

## Main Technical Content

The new report now documents:

- the actual current direct notebook settings
- the direct target and direct MPC mathematics
- the refined Step A selector mathematics
- the safety-filter QCQP mathematics
- the latest saved direct ten-scenario metrics
- the latest saved selector-ablation metrics
- the latest saved RL safety-filter summary
- the control-theoretic reason `objective_zero` is not equivalent to the direct controller
- the main next matched experiment needed to resolve the remaining uncertainty

## Figures Collected

The report copies and references high-signal saved figures from:

- `report/figures/direct_lyapunov_mpc_frozen_output_disturbance/`
- `Data/debug_exports/mpc_selector_term_ablation/20260330_233901/`
- `Data/debug_exports/rl_safety_filter/20260330_212055/`

No existing raw result directories were modified or overwritten.

## Validation

- Confirmed the referenced report figures were copied into `report/figures/2026-04-30_lyapunov_results_synthesis/`
- Confirmed the report inputs exist:
  - `Data/debug_exports/direct_lyapunov_mpc_ten_scenario/20260424_162348/`
  - `Data/debug_exports/mpc_selector_term_ablation/20260330_233901/`
  - `Data/debug_exports/rl_safety_filter/20260330_212055/`
- No Python-module edits were made in this change batch, so `py_compile` was not applicable
