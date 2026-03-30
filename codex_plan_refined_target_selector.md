# Codex Plan: Rebuild `refined_target_selector.py` into a paper-faithful two-stage offset-free target selector

## Purpose

This document is a concrete implementation plan for modifying **only** `refined_target_selector.py` so it follows the standard offset-free MPC target-calculation structure from the literature, while still remaining practical for later use inside the Lyapunov layer.

The immediate goal is **not** to rewire all call sites yet. The goal is to make `refined_target_selector.py` scientifically correct, numerically safer, backward-friendly, and ready to become the common target selector across the project.

---

## Why we are changing it

The current `compute_ss_target_refined_rawlings(...)` mixes two different ideas into one optimization problem:

1. Offset-free steady-state target calculation using the current disturbance estimate.
2. Closest-feasible target selection when the desired setpoint is not reachable.

That combined objective is practical, but it is **not** the clean structure used in the classic offset-free MPC papers.

The literature-consistent structure is:

### Stage 1: Exact offset-free target calculation
Solve for `(x_s, u_s)` such that the plant is at steady state under the frozen disturbance estimate and the controlled outputs match the requested setpoint exactly.

### Stage 2: Closest-feasible fallback
Only if Stage 1 is infeasible, solve a second problem that finds the nearest admissible steady-state output target.

This separation is important for Lyapunov work because the Lyapunov center should be the **true equilibrium target**, not a target that has already been softened by weight tradeoffs even when exact tracking is feasible.

---

## Paper basis to follow

### 1. Offset-free MPC target calculation
Use the structure from the standard offset-free MPC papers:

- Pannocchia and Rawlings, AIChE Journal, 2003
- Pannocchia and Bemporad, IEEE TAC, 2007
- Muske and Badgwell, Journal of Process Control, 2002

Core idea:

- Augment the model with integrating disturbances.
- Update the disturbance estimate through the observer innovation.
- Freeze the current disturbance estimate during target calculation.
- Solve a steady-state target QP for `(x_s, u_s)`.

### 2. Closest admissible target logic
Use the artificial-reference / closest-admissible-target idea from:

- Limon et al., Automatica, 2008

Core idea:

- If the desired steady-state target is not admissible, move to the closest admissible steady state.

---

## Current file to modify

- `refined_target_selector.py`

Current exported function:

- `compute_ss_target_refined_rawlings(...)`

Keep this function name for now so existing imports do not break.

---

## Current codebase context

From the current codebase scan:

- `standard_lyap_tracking_mpc.py` imports `compute_ss_target_refined_rawlings`
- `standard_lyap_tracking_mpc_v2.py` still uses the old slack selector
- `safe_mpc_with_lyapunov_filter.py` still uses the old slack selector
- `safe_mpc_with_lyapunov_filter_v2.py` still uses the old slack selector
- `Simulation/run_rl_lyapunov.py` still uses the old slack selector

So this file can be fixed first, then propagated later.

---

## Scientific target formulation we want

Assume the augmented linear model is split as:

```text
A_aug = [[A, Bd],
         [0,  I ]]

B_aug = [[B],
         [0]]

C_aug = [C, Cd]
```

with augmented estimate:

```text
xhat_aug = [xhat; dhat]
```

Then the frozen steady-state disturbance is:

```text
d_s = d_hat
```

The physical steady-state equations are:

```text
(I - A) x_s - B u_s - Bd d_hat = 0
```

and the steady-state output is:

```text
y_s = C x_s + Cd d_hat
```

If all measured outputs are controlled, then the exact target equation is:

```text
y_s = y_sp
```

If only a subset or linear combination is controlled, then use a selector matrix `H`:

```text
H y_s = y_sp
```

For now, because the current file and call sites assume all outputs are controlled, implement the all-output case first.

---

## High-level design change

Replace the current single weighted objective with a **two-stage solve**.

### Stage 1: Exact target solve
Solve

```text
minimize   (u_s - u_nom)^T Ru (u_s - u_nom) + eps_x * ||x_s||^2
```

subject to

```text
(I - A) x_s - B u_s - Bd d_hat = 0
C x_s + Cd d_hat = y_sp
u_lo <= u_s <= u_hi
(optional) y_lo <= y_s <= y_hi
```

Notes:

- `eps_x * ||x_s||^2` is a very small regularizer to make `x_s` numerically well-posed.
- This is not the main design objective. It is a tie-breaker.
- Use hard exact output equality here.
- Soft output **inequalities** may still be allowed if requested.

### Stage 2: Closest-feasible fallback
Only if Stage 1 fails, solve

```text
minimize   (y_s - y_sp)^T Ty (y_s - y_sp)
         + (u_s - u_nom)^T Ru (u_s - u_nom)
         + eps_x * ||x_s||^2
```

subject to

```text
(I - A) x_s - B u_s - Bd d_hat = 0
u_lo <= u_s <= u_hi
(optional) y_lo <= y_s <= y_hi
```

Optional tiny regularizers can be added later:

```text
+ (x_s - x_s_prev)^T Qdx (x_s - x_s_prev)
+ (u_s - u_s_prev)^T Rdu (u_s - u_s_prev)
```

but these should remain secondary and should not compete with exact target satisfaction when Stage 1 is feasible.

---

## Why `x_s` is still found even if Stage 1 only penalizes `u_s`

This is important.

`x_s` is still a decision variable. The optimizer solves over both `x_s` and `u_s`.
The reason only `u_s` appears strongly in the cost is that the equality constraints define which state is compatible with the chosen steady-state input and the frozen disturbance estimate.

So the solver returns:

- `u_s`: the preferred admissible steady-state input
- `x_s`: the corresponding admissible steady-state state

Then the Lyapunov target becomes:

```text
x_s_aug = [x_s; d_hat]
```

and the Lyapunov deviation is:

```text
x_tilde = xhat_aug - x_s_aug
```

---

## Detailed implementation plan for Codex

## Phase 1: Preserve public API while changing internal logic

### Task 1. Keep the function name
Do not rename:

```python
compute_ss_target_refined_rawlings(...)
```

Reason:
- `standard_lyap_tracking_mpc.py` already imports this name.
- We want to avoid breaking imports while changing behavior.

### Task 2. Preserve current inputs unless clearly obsolete
Keep these inputs for compatibility:

- `A_aug, B_aug, C_aug, xhat_aug, y_sp`
- `u_min, u_max, u_nom`
- `Ty_diag, Ru_diag, Qx_diag, w_x`
- `x_s_prev, u_s_prev, Qdx_diag, Rdu_diag`
- `y_min, y_max, u_tight, y_tight`
- `soft_output_bounds, Wy_low_diag, Wy_high_diag`
- `solver_pref, return_debug`

But reinterpret them under the new two-stage logic.

### Task 3. Add optional future-proof argument `H=None`
Add an optional argument:

```python
H=None
```

Behavior:
- if `H is None`, use identity over all outputs
- if `H` is provided, enforce controlled-output target on `H @ y_s`

This is not required by current call sites, but it makes the function scientifically aligned with the standard formulation.

---

## Phase 2: Strengthen dimensional and model checks

### Task 4. Add explicit shape validation
Before solving, validate:

- `A_aug` is square
- `B_aug.shape[0] == A_aug.shape[0]`
- `C_aug.shape[1] == A_aug.shape[0]`
- `xhat_aug.size == A_aug.shape[0]`
- `u_min.size == u_max.size == n_u`
- `y_sp.size == n_c`, where `n_c = n_y` if `H is None`, else `H.shape[0]`

### Task 5. Validate augmentation consistency assumptions
Assume the file is receiving the standard offset-free augmented model with:

- physical states first
- disturbance states second
- number of disturbance states equal to `n_y`

Add a debug field noting this assumption.

Do **not** try to auto-detect arbitrary augmentations here.

---

## Phase 3: Split the augmented model exactly once

### Task 6. Keep the current block split
Continue using:

```python
A = A_aug[:n_x, :n_x]
Bd = A_aug[:n_x, n_x:]
B = B_aug[:n_x, :]
C = C_aug[:, :n_x]
Cd = C_aug[:, n_x:]
d_hat = xhat_aug[n_x:]
```

This is consistent with the current code.

### Task 7. Define controlled output consistently
If `H is None`:

```python
yc_expr = y_expr
sp_expr = y_sp
```

Else:

```python
yc_expr = H @ y_expr
sp_expr = y_sp
```

where:

```python
y_expr = C @ x + Cd @ d_hat
```

---

## Phase 4: Build Stage 1 exact target problem

### Task 8. Build exact equality target constraint
Stage 1 must use:

```python
yc_expr == sp_expr
```

not a weighted penalty.

### Task 9. Stage 1 objective
Use:

```python
cp.quad_form(u - u_nom, Ru) + cp.quad_form(x, Qx_exact)
```

where `Qx_exact` is small.

Implementation detail:
- If `Qx_diag` is supplied, use it.
- Else use `w_x * I`.
- Make sure `w_x` remains tiny by default.

### Task 10. Stage 1 constraints
Always include:

```python
(np.eye(n_x) - A) @ x - B @ u - Bd @ d_hat == 0
u >= u_lo
u <= u_hi
```

Add optional output bound constraints if `y_min` and/or `y_max` are provided.

Important:
- output inequalities can be hard or soft
- target equality should remain hard in Stage 1

### Task 11. Keep soft output inequalities only for bounds
If `soft_output_bounds=True`, allow slack only on inequality bounds:

```python
y_expr + s_y_low >= y_lo
y_expr - s_y_high <= y_hi
```

Do **not** soften the exact setpoint equality in Stage 1.

---

## Phase 5: Add Stage 2 closest-feasible fallback

### Task 12. Trigger Stage 2 only when Stage 1 fails
Failure conditions:

- no primal values returned
- status not in `("optimal", "optimal_inaccurate")`
- or solver residual checks fail

### Task 13. Stage 2 objective
Use:

```python
cp.quad_form(yc_expr - sp_expr, Ty)
+ cp.quad_form(u - u_nom, Ru)
+ cp.quad_form(x, Qx_fallback)
```

Optional:

```python
+ cp.quad_form(x - x_s_prev, Qdx)
+ cp.quad_form(u - u_s_prev, Rdu)
```

only if previous targets and weights are supplied.

### Task 14. Stage 2 constraints
Use the same steady-state equation and bounds as Stage 1, but remove exact target equality.

This stage computes the nearest admissible equilibrium target.

---

## Phase 6: Restore solver acceptance discipline

### Task 15. Do not trust `optimal_inaccurate` blindly
After solve, explicitly compute:

```python
dyn_res = (I - A) @ x_s - B @ u_s - Bd @ d_hat
```

and reject the solution if:

```python
max(abs(dyn_res)) > dyn_tol
```

Suggested starting tolerance:

```python
dyn_tol = 1e-6 to 1e-5
```

depending on solver.

### Task 16. For Stage 1, also check target equality residual
If Stage 1 solved, compute:

```python
target_eq_res = yc_s - sp_expr
```

and reject if too large.

### Task 17. If using hard output bounds, check them numerically too
Compute bound residuals after solve and reject if clearly violated.

---

## Phase 7: Improve debug output without breaking downstream tools

### Task 18. Add `solve_stage`
Return in debug:

```python
"solve_stage": "exact" or "fallback" or None
```

### Task 19. Keep old-style compatibility fields
Even if the new method does not use an explicit output slack variable for target mismatch, include:

```python
"target_error"
"target_error_inf"
"target_error_norm"
"target_slack"
"target_slack_inf"
```

with:

```python
target_slack = target_error
```

Reason:
- this helps `target_selector_diagnostics.py`
- this avoids NaNs in tools that were written around the old selector language

### Task 20. Return `x_s_aug`
In debug, add:

```python
"x_s_aug": np.concatenate([x_s, d_hat])
```

This is very useful for Lyapunov code.

### Task 21. Add exact/fallback residual fields
Add:

```python
"dyn_residual_inf"
"target_eq_residual_inf"
"bound_violation_inf"
```

where applicable.

### Task 22. Keep margin fields
Preserve and continue reporting:

- `margin_to_u_min`
- `margin_to_u_max`
- `tight_margin_to_u_min`
- `tight_margin_to_u_max`
- output margins if available

---

## Phase 8: Clarify semantics in the docstring

### Task 23. Rewrite the docstring honestly
Do not claim the function is always using a "closest-admissible output target objective".

Instead describe it as:

1. exact offset-free target solve first
2. closest-feasible fallback second
3. optional regularization on target motion and soft output bounds

Also state explicitly:

- all variables are expected in the same coordinate system
- in this codebase that should be scaled deviation coordinates
- `u_nom` should therefore usually be zero in deviation form

---

## Phase 9: Keep scope tight for this patch

### Task 24. Do not modify call sites yet
Do not edit:

- `standard_lyap_tracking_mpc.py`
- `standard_lyap_tracking_mpc_v2.py`
- `safe_mpc_with_lyapunov_filter.py`
- `safe_mpc_with_lyapunov_filter_v2.py`
- `Simulation/run_rl_lyapunov.py`

in this patch.

This patch is only to make `refined_target_selector.py` correct and ready.

### Task 25. Do not redesign Lyapunov matrices in this patch
The target selector should only return a correct equilibrium target. Do not change the Lyapunov `P` design here.

---

## Suggested code structure inside `refined_target_selector.py`

Refactor the function internally into helper sections or small helpers:

1. input validation
2. model split
3. weight construction
4. Stage 1 problem build and solve
5. Stage 2 problem build and solve if needed
6. post-solve residual checks
7. debug packing

If helpful, create small internal helpers such as:

```python
_build_output_bound_terms(...)
_solve_problem_with_preferences(...)
_check_stage_solution(...)
_pack_debug(...)
```

Keep helpers private to the file.

---

## Acceptance criteria for this patch

The patch is successful if all of the following are true.

### Functional
1. The function still imports under the same name.
2. Stage 1 returns an exact target when the requested setpoint is feasible.
3. Stage 2 returns a nearest feasible target when Stage 1 is infeasible.
4. The returned `d_s` remains `d_hat`.
5. The returned `x_s` and `u_s` satisfy the steady-state equation up to tolerance.

### Scientific
6. The exact case does not trade away target equality just to reduce `u_s` or smooth motion.
7. The fallback case is the only place where target mismatch is optimized.
8. The debug output exposes enough information to inspect equilibrium quality for Lyapunov use.

### Compatibility
9. Existing import paths do not break.
10. Old diagnostics expecting `target_slack`-style fields can still run.

---

## Simple test matrix for Codex to run after editing

### Test A: feasible exact target
Use a small stable linear system and a reachable `y_sp`.
Expected:

- `solve_stage == "exact"`
- `target_eq_residual_inf` near zero
- `target_error_inf` near zero
- `dyn_residual_inf` near zero

### Test B: infeasible due to input bounds
Use a target that cannot be reached under `u_min/u_max`.
Expected:

- `solve_stage == "fallback"`
- `target_error_inf > 0`
- `dyn_residual_inf` near zero
- `u_s` on or near the active bound

### Test C: soft output bounds enabled
Pick output bounds that are inconsistent with the exact target.
Expected:

- Stage 1 may fail if equality and bounds conflict
- Stage 2 should solve with nonzero `s_y_low` or `s_y_high` if soft bounds are active

### Test D: previous target regularization
Provide `x_s_prev`, `u_s_prev`, `Qdx_diag`, `Rdu_diag`.
Expected:

- only fallback stage should use those terms
- exact feasible tracking should still keep equality exact

---

## What not to do

1. Do not keep the current one-shot compromise objective as the main solve.
2. Do not soften the exact setpoint equality in Stage 1.
3. Do not let motion regularization override feasibility or exact tracking.
4. Do not silently accept poor `optimal_inaccurate` solutions without residual checks.
5. Do not assume physical units if the rest of the codebase is using scaled deviation coordinates.

---

## Codex execution prompt

Use this prompt with Codex for the patch:

```text
Modify only `refined_target_selector.py`.

Goal:
Rebuild `compute_ss_target_refined_rawlings(...)` into a two-stage offset-free steady-state target selector.

Requirements:
1. Keep the function name unchanged.
2. Preserve the current public arguments as much as possible.
3. Add optional argument `H=None` for future controlled-output selection.
4. Use the standard augmented-model split already present in the file.
5. Stage 1 must solve an exact offset-free target problem:
   - variables: x, u
   - objective: (u-u_nom)^T Ru (u-u_nom) + small state regularization
   - constraints:
       (I-A)x - Bu - Bd d_hat = 0
       H(Cx + Cd d_hat) = y_sp  if H provided
       Cx + Cd d_hat = y_sp     if H is None
       input bounds
       optional output bounds
   - target equality must remain hard.
6. Only if Stage 1 fails, solve Stage 2 fallback:
   - objective: target mismatch penalty + input penalty + small state regularization
   - optional target-motion regularization can be included here only
   - same steady-state and bound constraints
7. Keep `d_s = d_hat`.
8. Add explicit residual checks after each solve and reject numerically bad solutions.
9. Expand debug output with:
   - solve_stage
   - x_s_aug
   - target_eq_residual_inf
   - target_slack and target_slack_inf aliases
10. Preserve backward-friendly debug fields where possible.
11. Update the docstring so it accurately describes the two-stage logic and the expected coordinate convention.
12. Do not modify any other file.

Also add a small internal helper structure if needed, but keep everything inside `refined_target_selector.py`.
```

---

## Recommended next step after this patch

After this file is fixed and tested, the next patch should switch the most important Lyapunov path to use it first:

1. `Simulation/run_rl_lyapunov.py`
2. `safe_mpc_with_lyapunov_filter_v2.py`
3. `standard_lyap_tracking_mpc_v2.py`

But that is a separate patch.

