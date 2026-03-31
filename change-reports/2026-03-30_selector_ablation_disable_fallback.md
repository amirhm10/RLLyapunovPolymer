# 2026-03-30 Selector Ablation Disable Fallback

## Why

The target-selector ablation notebook should match the current safety-MPC parameter surface, but it should not use fallback MPC. The goal of this notebook is to isolate selector-term effects, and fallback recovery would mask those differences.

## What Changed

- Updated `LyapunovSafetyFilterMPCTargetSelectorTermAblation.ipynb` so:
  - `fallback_policy` is now `None`
  - the notebook prints an explicit message that fallback MPC is disabled
  - the title text now states that it mirrors the safety notebook except for fallback being disabled
- Updated `report/target_selector_term_ablation_study.md` to document that fallback MPC is intentionally off in this study

## Parameter Alignment Check

The ablation notebook was rechecked against `LyapunovSafetyFilterMPC.ipynb`.

The shared safety/selector parameters on disk already matched:

- `rho_lyap`
- `lyap_eps`
- `lyap_tol`
- `w_mpc`
- `w_track`
- `w_move`
- `w_ss`
- `mpc_target_policy`
- `tracking_target_policy`
- `target_selector_config`
- `selector_H`
- `target_backup_policy`
- `selector_warm_start`
- `lyap_acceptance_mode`
- `reuse_mpc_solution_as_ic`
- `reset_system_on_entry`
- `allow_lyap_slack`
- `trust_region_delta`
- `allow_trust_region_slack`

The only intentional difference after this follow-up change is:

- `fallback_policy = None`

## Validation

- JSON parse check for `LyapunovSafetyFilterMPCTargetSelectorTermAblation.ipynb`
