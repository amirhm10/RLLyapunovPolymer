# 2026-03-26 Safe Filter Consistency Fix

## Why This Change Was Made

This change batch fixes four controller-path inconsistencies that were making the safe-MPC filter hard to analyze and hard to roll back:

1. The repository documentation and the code did not agree on the Lyapunov tolerance semantics.
2. The target selector accepted previous-target information but did not actually use solver warm start, and the active safe-MPC / safe-RL runners forced state-target smoothing off.
3. The last valid target was only used for plotting-style diagnostics, not as an operational backup target for the candidate test or QCQP.
4. The QCQP debug data did not distinguish between “QCQP attempted”, “QCQP solved”, and “QCQP hard-accepted”, which made slack-related behavior hard to interpret from saved runs.

## Behavior Before

- Canonical code path already implemented `V_bound = rho * V_k + eps_lyap`, but some Markdown / LaTeX notes still described the stricter `- eps_lyap` version.
- `prev_target_info` regularized the selector objective only through `Rdu_diag`; the active safe-MPC / safe-RL runners passed `Qdx_diag=None`.
- Selector solves used `warm_start=False`.
- If the current selector failed, the safety filter treated the target as unavailable operationally even if a previous valid target existed.
- Trust-region softness was implicit whenever `trust_region_delta` was enabled.
- Slack-enabled QCQP solves could still end in fallback, but the saved debug artifacts did not clearly separate “solved but rejected by hard post-check” from “never solved”.

## Behavior After

- The canonical Lyapunov tolerance is documented consistently as:
  - `V_next <= rho * V_k + eps_lyap`
  - where `eps_lyap` is a numerical tolerance.
- The selector now supports actual warm start:
  - previous target values are assigned to CVXPY variables,
  - solver calls use `warm_start=True` when enabled.
- The safe-MPC and safe-RL runners now expose state-target smoothing and pass a small default:
  - `Qdx_tgt_diag = 1e-6 * ones(n_x)` unless overridden.
- The safety filter now computes and uses an `effective_target_info`:
  - current target if available,
  - otherwise last valid target when `target_backup_policy="last_valid"`.
- The tracking-target policy is normalized in one place and used consistently by:
  - upstream MPC objective,
  - QCQP output term,
  - fallback MPC objective.
- Trust-region enablement and trust-region slack are now separate knobs.
- Debug/export now records:
  - current target vs effective target,
  - whether the effective target was reused,
  - QCQP attempted / solved / hard-accepted state,
  - QCQP rejection after hard post-check,
  - selector warm-start flags and smoothing weights.

## Affected Files

- `Lyapunov/target_selector.py`
- `Lyapunov/safety_filter.py`
- `Lyapunov/safety_debug.py`
- `Simulation/run_mpc_lyapunov.py`
- `Simulation/run_rl_lyapunov.py`
- `LyapunovSafetyFilterMPC.ipynb`
- `LyapunovSafetyFilterRL.ipynb`
- `SafeMPCFilterProcedure.md`
- `LyapunovHyperparameters.md`
- `report/current_target_selector_implementation.tex`
- `report/lyapunov_safety_filter_report.tex`

## Active Defaults After This Change

- `tracking_target_policy = "raw_setpoint"`
- `target_backup_policy = "last_valid"`
- `selector_warm_start = True`
- `Qdx_tgt_diag = 1e-6 * ones(n_x)` in the runner defaults
- `lyap_acceptance_mode = "hard_only"`
- `allow_trust_region_slack = False`

Existing compatibility knobs remain readable:

- `mpc_target_policy`
- `allow_lyap_slack`
- `trust_region_delta`

## Rollback Snapshot

Readable pre-change source snapshots are stored at:

- `change-reports/snapshots/2026-03-26_safe_filter_consistency_fix/`

Snapshot contents:

- `target_selector.py`
- `safety_filter.py`
- `safety_debug.py`
- `run_mpc_lyapunov.py`
- `run_rl_lyapunov.py`
- `lyapunov_core.py`

If the new behavior needs to be reverted, restore from that snapshot and then rerun the validation checks below.

## Validation Completed

- `python -m py_compile` passed for:
  - `Lyapunov/target_selector.py`
  - `Lyapunov/safety_filter.py`
  - `Lyapunov/safety_debug.py`
  - `Simulation/run_mpc_lyapunov.py`
  - `Simulation/run_rl_lyapunov.py`
  - `Lyapunov/lyapunov_core.py`
- Both notebook files still parse as valid JSON:
  - `LyapunovSafetyFilterMPC.ipynb`
  - `LyapunovSafetyFilterRL.ipynb`
- `pdflatex` succeeded for:
  - `report/current_target_selector_implementation.tex`
  - `report/lyapunov_safety_filter_report.tex`

## Notes For The Next Comparison

- If fallback-MPC frequency drops but projection behavior still looks bad, the next debug split to inspect is:
  - `current_target_success`
  - `effective_target_reused`
  - `qcqp_attempted`
  - `qcqp_solved`
  - `qcqp_hard_accepted`
- If exact-MPC equivalence is the goal, keep:
  - `tracking_target_policy = "raw_setpoint"`
  - `reuse_mpc_solution_as_ic = False`
  - `reset_system_on_entry = True`
