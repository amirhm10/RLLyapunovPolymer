# 2026-03-26 Target Selector Four Modes

## Why This Change Was Made
- The safe-MPC filter had only one refined target-selector path.
- The repository now needs four selectable selector formulations behind one common API so closed-loop comparisons can be run without rewriting the downstream controller.
- The main control issue behind this stage is target inconsistency: different layers can pull toward different references, while the disturbance channel can dominate `y_s`.

## Snapshot Before Changes
- Source snapshot: `change-reports/snapshots/2026-03-26_target_selector_four_modes/`
- Snapshotted files:
  - `Lyapunov/target_selector.py`
  - `Simulation/run_mpc_lyapunov.py`
  - `Simulation/run_rl_lyapunov.py`
  - `Lyapunov/safety_filter.py`
  - `Lyapunov/safety_debug.py`

## Behavior Before
- Only the current two-stage Rawlings-style selector existed.
- Downstream code consumed the selector through `prepare_filter_target_from_refined_selector(...)`.
- There was no selector-mode dispatch, no common `r_s`, and no unified comparison helper.
- Debug exports did not explicitly log selector mode, `r_s`, disturbance-deviation metrics, or `C x_s` versus `C_d d_s`.

## Behavior After
- New common selector API: `prepare_filter_target(...)`.
- Four selectable modes:
  - `current_exact_fallback_frozen_d`
  - `free_disturbance_prior`
  - `compromised_reference`
  - `single_stage_robust_sstp`
- The old `prepare_filter_target_from_refined_selector(...)` is now a compatibility wrapper over Mode 0.
- New standardized selector output now includes:
  - `selector_mode`
  - `r_s`
  - `d_s_frozen`
  - `d_s_optimized`
  - `d_s_minus_dhat_inf`
  - unified `selector_debug`
- The safe-MPC and safe-RL runners now:
  - accept `selector_mode`, `target_selector_config`, and `selector_H`
  - build selector configs once per run
  - use one selector entrypoint
  - propagate selector mode and target-reference metadata into debug output
- Tracking-target selection is now mode-aware:
  - Mode 0 preserves old behavior
  - Mode 2 forces selector-reference tracking
  - Modes 1 and 3 default to selector-reference tracking
- Debug/export additions now include:
  - `selector_mode`
  - `effective_selector_mode`
  - `r_s`
  - `d_s_minus_dhat_inf`
  - `selector_objective_value`
  - `cx_s`
  - `cd_d_s`
  - new decomposition plots and `r_s` vs `y_sp` plots
- Added comparison helper:
  - `Lyapunov/target_selector_mode_comparison.py`
- Added smoke-test script:
  - `target_selector_mode_smoke_test.py`

## Files Added
- `Lyapunov/target_selector_mode_comparison.py`
- `target_selector_mode_smoke_test.py`
- `report/target_selector_four_modes_report.tex`
- `change-reports/2026-03-26_target_selector_four_modes.md`

## Important Defaults
- Default selector mode remains `current_exact_fallback_frozen_d`.
- Lyapunov tolerance semantics are unchanged:
  - `V_next <= rho * V_k + eps_lyap`
- Effective-target backup policy stays `last_valid`.
- Accepted control remains hard-safe by default.

## Rollback Notes
- To roll back this stage manually, restore the snapshot copies from:
  - `change-reports/snapshots/2026-03-26_target_selector_four_modes/`
- The compatibility wrapper for the previous selector entrypoint remains in place, so user notebooks that still call the old function name should keep importing successfully.
