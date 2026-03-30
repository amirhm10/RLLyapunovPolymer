# Lyapunov Core Mapping

## Paper Basis
This repository's standard Lyapunov MPC path now follows a three-layer structure that is consistent with the offset-free and tracking MPC literature:

- Offset-free steady-state target calculation with a frozen disturbance estimate:
  Pannocchia and Rawlings (2003), Pannocchia and Bemporad (2007), and the later tutorial review by Pannocchia et al.
- Closest-admissible fallback when the requested target is infeasible:
  Limon, Alvarado, Alamo, and Camacho (2008).
- Terminal-cost and terminal-set Lyapunov MPC for recursive feasibility and convergence:
  Mayne, Rawlings, Rao, and Scokaert (2000).

In this codebase, those ideas are split intentionally:

1. `Lyapunov/target_selector.py` computes `(x_s, u_s, d_s)`.
2. `Lyapunov/lyapunov_core.py` solves the target-relative finite-horizon MPC.
3. `Lyapunov/run_lyap_mpc.py` handles closed-loop orchestration, plant stepping, and observer updates.

## Notation To Variable Map
The controller uses scaled deviation coordinates inside the MPC stack.

- Augmented observer state:
  `xhatdhat` or `x0_aug = [x_hat; d_hat]`
- Steady-state target from the selector:
  `x_s`, `u_s`, `d_s`
- Augmented target:
  `x_s_aug = [x_s; d_s]`
- Admissible tracking output target:
  `y_target = C_aug @ x_s_aug`
- Physical target-relative errors used in the Lyapunov core:
  `e_x = x - x_s`
  `e_u = u - u_s`
  `e_y = y - y_target`
- Terminal Lyapunov ingredients:
  `P_x` is the Riccati terminal matrix
  `K_x` is the local linear feedback on physical-state error
  `alpha_terminal_raw` is the unscaled input-headroom ellipsoid radius
  `alpha_terminal` is the scaled terminal radius used by policy
  `alpha_terminal_used` is the actual bound given to the solver after any skip logic

The core prediction model still propagates the full augmented state, but the Lyapunov cost and terminal constraint are centered on the physical error state:

`e_N^T P_x e_N`, where `e_N = x_N - x_s`.

## File And Function Data Flow
### 1. Target selector
`Lyapunov/target_selector.py`

- Input: `A_aug`, `B_aug`, `C_aug`, current `xhat_aug`, requested `y_sp`, and bounds.
- Output: admissible steady-state target `(x_s, u_s, d_s)` plus diagnostics.
- Behavior:
  Stage 1 enforces exact offset-free steady-state targeting.
  Stage 2 computes the closest feasible steady-state if Stage 1 fails.

### 2. Terminal ingredients and tracking solver
`Lyapunov/lyapunov_core.py`

- `design_standard_tracking_terminal_ingredients(...)`
  extracts the physical subsystem and solves the DARE for `P_x` and `K_x`.
- `compute_terminal_alpha_input_only(...)`
  computes the largest target-centered ellipsoid that keeps `u = u_s + K_x e_x` inside input bounds.
- `StandardTrackingLyapunovMpcRawlingsTargetSolver.solve_tracking_mpc_step(...)`
  builds a CVXPY problem with:
  horizon output tracking cost,
  input-to-target cost,
  optional move suppression,
  terminal cost,
  and optional terminal ellipsoid constraint.

The optimization variables are the control horizon sequence and the predicted augmented state sequence. The objective is written around the selected equilibrium, not around the raw setpoint.

### 3. Closed-loop rollout
`Lyapunov/run_lyap_mpc.py`

For each step:

1. Read `x0_aug` and the current scaled deviation input.
2. Compute `y_sp_k`.
3. Call the refined target selector.
4. Build `y_target` from the admissible equilibrium by default.
5. Compute terminal-radius diagnostics from `u_s`.
6. Optionally skip the terminal constraint if the radius is too small.
7. Solve the target-relative MPC.
8. Apply the first move or the configured fallback action.
9. Step the plant and update the observer.
10. Store target diagnostics and tracking diagnostics without changing the rollout return tuple.

### 4. Compatibility layer
`standard_lyap_tracking_mpc_v2.py`

- Re-exports the canonical Lyapunov implementation.
- Preserves legacy helper names such as `compute_ss_target_slack_rawlings(...)`.
- Maps old target-weight names like `Qs_tgt_diag` to the canonical target-mismatch weight input.

## Validation Checklist
- `python -m py_compile` should succeed for:
  `Lyapunov/target_selector.py`
  `Lyapunov/lyapunov_core.py`
  `Lyapunov/run_lyap_mpc.py`
  `standard_lyap_tracking_mpc_v2.py`
  `standard_lyap_tracking_mpc.py`
  `utils/lyapunov_utils.py`
- Feasible target case:
  target selector returns `solve_stage == "exact"`,
  tracking MPC solves successfully,
  terminal diagnostics report finite `terminal_value`.
- Infeasible target case:
  target selector returns `solve_stage == "fallback"`,
  rollout tracks the admissible `y_target = y_s`.
- Small terminal-radius case:
  `terminal_constraint_skipped == True`,
  `alpha_terminal` is still logged,
  `alpha_terminal_used` is `None`.
- Legacy compatibility case:
  imports through `standard_lyap_tracking_mpc_v2.py` and `standard_lyap_tracking_mpc.py` still resolve to the canonical path without changing notebook call sites.
