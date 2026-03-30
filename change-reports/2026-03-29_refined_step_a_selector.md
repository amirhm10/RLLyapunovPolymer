# 2026-03-29 Refined Step A Selector Replacement

## Why This Change Was Made
- The four-mode selector stage added too much surface area without solving the core centering problem.
- The observed failure mode was not mainly output-target generation. It was that the returned steady package could be acceptable in output space while remaining poor as a projection / Lyapunov center.
- The new implementation follows the refined Step A design: keep `d_s = d_hat_k`, keep the soft output objective, and add operating-region anchors through `u_applied_k`, previous-target smoothing, and a weak `xhat_k` state anchor.

## Snapshot Before Changes
- Source snapshot: `change-reports/snapshots/2026-03-29_refined_step_a_selector/`
- Snapshotted files:
  - `Lyapunov/target_selector.py`
  - `Simulation/run_mpc_lyapunov.py`
  - `Simulation/run_rl_lyapunov.py`
  - `Lyapunov/safety_filter.py`
  - `Lyapunov/safety_debug.py`
  - `Lyapunov/target_selector_mode_comparison.py`

## Behavior Before
- The active selector API exposed four runtime modes:
  - `current_exact_fallback_frozen_d`
  - `free_disturbance_prior`
  - `compromised_reference`
  - `single_stage_robust_sstp`
- Notebook configs exposed `run_config1..4` and a mode selector.
- The debug/export stack carried mode-specific fields and plots.
- The target selector objective depended on the mode, rather than one single canonical selector formulation.

## Behavior After
- Only one active selector remains: `refined_step_a`.
- `prepare_filter_target(...)` and `prepare_filter_target_from_refined_selector(...)` now both resolve to the same refined Step A selector.
- The selector solves one steady-state problem with:
  - soft output target term on `r_s - y_sp`
  - input anchor to `u_applied_k`
  - input smoothing on `u_s - u_s_prev`
  - state smoothing on `x_s - x_s_prev`
  - weak state anchor on `x_s - xhat_k`
  - fixed `d_s = d_hat_k`
- The runners now pass `u_applied_k = u_prev_dev` into the selector.
- The active notebooks now expose one selector config block instead of four mode presets.
- Debug exports now log:
  - selector objective-term breakdown
  - previous-target term activation
  - new selector weight diagnostics
  - refined Step A stage labels

## Files Added
- `report/refined_step_a_selector_parameters.md`
- `report/refined_step_a_selector_method.tex`
- `change-reports/2026-03-29_refined_step_a_selector.md`

## Important Defaults
- `alpha_u_ref = 0.5`
- `alpha_du_sel = 0.5`
- `alpha_dx_sel = 0.05`
- `alpha_x_ref = 0.01`
- `x_weight_base = "CtQC"`
- `use_output_bounds_in_selector = True`
- Lyapunov tolerance semantics are unchanged:
  - `V_next <= rho * V_k + eps_lyap`
- Downstream tracking remains on raw `y_sp` by default.

## Rollback Notes
- To roll back this stage manually, restore the snapshot copies from:
  - `change-reports/snapshots/2026-03-29_refined_step_a_selector/`
- The selector still accepts old mode strings as deprecated compatibility inputs, but they now all resolve to the same refined Step A implementation.
