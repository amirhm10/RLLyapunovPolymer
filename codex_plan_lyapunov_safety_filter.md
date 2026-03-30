# Codex Implementation Plan: General Lyapunov Safety Filter with Offset-Free Target Selector

## Objective

Implement a **general Lyapunov safety filter framework** that can sit **after any candidate control action generator**. The candidate action may come from:

- a reinforcement learning policy,
- a residual policy,
- an approximate MPC,
- a full MPC,
- a heuristic controller,
- or any other control law.

The framework should:

1. estimate the current offset-free equilibrium using the disturbance-augmented model and target selector,
2. construct a local Lyapunov function around that equilibrium,
3. test whether the candidate input satisfies a one-step Lyapunov decrease condition,
4. accept the candidate input if the condition is satisfied,
5. otherwise solve a minimally invasive optimization problem to compute a safe replacement control action.

This is **not just standard terminal-cost MPC stability analysis**. It is a **general safety-filter architecture** with a Lyapunov-based acceptance condition and an optimization-based correction step.

## Deliverables

Codex must produce the following deliverables.

### Deliverable 1: Code changes

Modify the codebase to add a general Lyapunov safety filter framework that can wrap around any candidate controller.

### Deliverable 2: Refined target selector integration

Use the refined target selector as the equilibrium generator for the filter. The target selector should follow the two-stage offset-free design:

- Stage A: exact steady-state target solve if feasible.
- Stage B: closest-admissible target solve if Stage A is infeasible.

### Deliverable 3: Extensive LaTeX report

Create an **extensive paper-style LaTeX report** documenting:

- motivation,
- literature background,
- mathematical formulation,
- code structure,
- design decisions,
- implementation details,
- acceptance condition,
- optimization fallback,
- limitations,
- and recommended next experiments.

The report should be written as if it were an internal technical note or a draft methods appendix for a paper.

The LaTeX report should compile as a standalone `.tex` file.

## High-level architecture

At each sampling instant `k`, the framework should execute the following sequence.

1. Obtain measurement `y_k`.
2. Update the observer on the disturbance-augmented model using the innovation.
3. Compute the current offset-free equilibrium `(x_s, u_s, d_s)` from the target selector.
4. Build the equilibrium-centered error state.
5. Evaluate the Lyapunov decrease condition for the candidate action `u_cand`.
6. If accepted, apply `u_cand`.
7. If rejected, solve an optimization problem to compute `u_safe`.
8. Apply the accepted or corrected action.
9. Log all target, Lyapunov, acceptance, and correction diagnostics.

## Scientific basis and references

The implementation should be grounded in the following references.

### Offset-free MPC and target selector references

1. G. Pannocchia and J. B. Rawlings, "Disturbance models for offset-free model-predictive control," AIChE Journal, 49(2), 426-437, 2003.
   - Use this as the core source for disturbance augmentation and offset-free tracking.

2. G. Pannocchia and A. Bemporad, "Combined design of disturbance model and observer for offset-free model predictive control," IEEE Transactions on Automatic Control, 52(6), 1048-1053, 2007.
   - Use this as the main source for the augmented observer structure and steady-state target calculation using the estimated disturbance.

3. K. R. Muske and T. A. Badgwell, "Disturbance modeling for offset-free linear model predictive control," Journal of Process Control, 12(5), 617-632, 2002.
   - Use this as a supporting source for detectability and steady-state target calculation.

4. U. Maeder and M. Morari, "Offset-free reference tracking with model predictive control," Automatica, 46(9), 1469-1476, 2010.
   - Use this as a supporting source for disturbance augmentation and observer-based offset removal.

5. D. Limon, I. Alvarado, T. Alamo, and E. F. Camacho, "MPC for tracking piecewise constant references for constrained linear systems," Automatica, 44(9), 2382-2387, 2008.
   - Use this as the key source for the closest-admissible target fallback logic.

### Safety-filter / Lyapunov-filter references

6. K. P. Wabersich and M. N. Zeilinger, "A predictive safety filter for learning-based control of constrained nonlinear dynamical systems," Automatica, 129, 109597, 2021.
   - Use this as the main conceptual source for a framework that receives a proposed control input and modifies it only when needed.

7. A. D. Ames, X. Xu, J. W. Grizzle, and P. Tabuada, "Control barrier function based quadratic programs for safety critical systems," IEEE Transactions on Automatic Control, 62(8), 3861-3876, 2017.
   - Use as a supporting source for optimization-based safety correction around a candidate input.

8. A. D. Ames, K. Galloway, K. Sreenath, and J. W. Grizzle, "Rapidly exponentially stabilizing control Lyapunov functions and hybrid zero dynamics," IEEE Transactions on Automatic Control, 59(4), 876-891, 2014.
   - Use as a supporting source for CLF-style one-step decrease enforcement.

### MPC / Lyapunov references

9. J. B. Rawlings, D. Q. Mayne, and M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, 2nd ed.
   - Use this as the main source for local terminal Lyapunov functions, Riccati-based quadratic Lyapunov design, and standard MPC stability background.

10. P. D. Christofides, R. Scattolini, D. M. de la Pena, and J. Liu, "Distributed model predictive control: A tutorial review and future research directions," Computers and Chemical Engineering, 51, 21-41, 2013.
   - Use only as a supporting process-control reference if needed for terminology.

11. J. Liu, D. Munoz de la Pena, and P. D. Christofides, "Lyapunov-based model predictive control of nonlinear systems subject to data losses," Automatica, 45(3), 790-797, 2009.
   - Use as a supporting process-systems reference for Lyapunov-based MPC ideas.

## Mathematical formulation

### 1. Augmented model

Assume the disturbance-augmented linear model is already available in discrete time:

```text
z_{k+1} = A_aug z_k + B_aug u_k
y_k = C_aug z_k
```

with

```text
z_k = [x_k; d_k]
```

and block structure

```text
A_aug = [[A, Bd],
         [0, I]]

B_aug = [[B],
         [0]]

C_aug = [C, Dd]
```

The observer updates `z_hat_k = [x_hat_k; d_hat_k]` using the innovation.

### 2. Target selector output

The target selector returns the steady-state equilibrium:

```text
x_s(k), u_s(k), d_s(k)
```

with

```text
d_s(k) = d_hat_k
```

and

```text
z_s(k) = [x_s(k); d_s(k)]
```

The steady-state relation is

```text
(I - A) x_s - B u_s - Bd d_hat = 0
```

and the steady-state output is

```text
y_s = C x_s + Dd d_hat
```

### 3. Error coordinates

Define the physical-state tracking error

```text
e_x(k) = x_hat_k - x_s(k)
```

For the first implementation, use the **physical-state-only Lyapunov function**. Do not build the initial Lyapunov function directly on the full augmented state unless there is a strong reason and numerical justification.

### 4. Local Lyapunov function

Construct a quadratic Lyapunov candidate

```text
V_k = e_x(k)^T P_x e_x(k)
```

where `P_x` is computed from a discrete Riccati equation using the physical subsystem `(A, B)` and chosen local weights `Q_lyap` and `R_lyap`.

Codex should implement a helper that solves the discrete algebraic Riccati equation and returns `P_x`.

### 5. Equilibrium-centered predicted error

For any candidate control input `u`, define

```text
Delta_u_s = u - u_s(k)
```

and the one-step predicted error around the **frozen equilibrium at time k** as

```text
e_x_next_pred(u) = A e_x(k) + B Delta_u_s
```

Then define

```text
V_next_pred(u) = e_x_next_pred(u)^T P_x e_x_next_pred(u)
```

### 6. Acceptance condition

Use the one-step contraction-style Lyapunov condition

```text
V_next_pred(u_cand) <= rho * V_k + eps_lyap
```

with

- `0 < rho < 1`
- `eps_lyap >= 0`

The term `eps_lyap` is needed because the target equilibrium may move from one time step to the next as the disturbance estimate changes.

### 7. Input admissibility checks

The candidate action should also satisfy all hard input and move constraints:

```text
u_min <= u_cand <= u_max
Delta_u_min <= u_cand - u_prev <= Delta_u_max
```

The final acceptance logic should therefore require both:

- hard bounds satisfied,
- Lyapunov decrease satisfied.

## Safety-filter optimization

If the candidate action is rejected, solve a minimally invasive correction problem.

### Version 1: one-step QP / convex correction

The first version should solve a one-step correction problem of the form

```text
min_{u, s_v, s_u} (u - u_cand)^T W_u (u - u_cand) + lambda_v * s_v^2 + lambda_u * s_u^2
```

subject to

```text
u_min <= u <= u_max
Delta_u_min <= u - u_prev <= Delta_u_max
V_next_pred(u) <= rho * V_k + eps_lyap + s_v
|u - u_cand| <= delta_trust + s_u
s_v >= 0
s_u >= 0
```

Notes for Codex:

- If the quadratic Lyapunov inequality leads to a convex quadratic constraint under the chosen representation, use a convex solver.
- If needed, convert the quadratic Lyapunov expression explicitly into a quadratic form in `u`.
- Keep the implementation numerically robust.
- Prefer a formulation that is easy to debug and log.

### Version 2: predictive backup optimization

Do **not** implement this unless Version 1 works well first.

Later, a short-horizon backup optimization can be added that searches for a safe backup trajectory instead of only a one-step safe correction.

## Required code tasks

### Task A: inspect current codebase and identify insertion points

Codex must inspect the current codebase and identify:

- where the observer state is updated,
- where the refined target selector is called,
- where candidate control actions are generated,
- and where a safety filter can be inserted without breaking existing flows.

### Task B: implement reusable Lyapunov helpers

Create reusable helper functions for:

1. extracting `(A, B, C, Bd, Dd)` from the augmented model,
2. computing `P_x` from the physical subsystem,
3. constructing the equilibrium-centered error,
4. evaluating `V_k`,
5. evaluating `V_next_pred(u)`,
6. checking the Lyapunov decrease condition,
7. checking hard input/move bounds.

### Task C: implement the general safety filter API

Create a generic function such as

```python
def apply_lyapunov_safety_filter(
    u_cand,
    xhat_aug,
    target_info,
    model_info,
    lyap_config,
    u_prev=None,
    bounds_info=None,
    return_debug=False,
):
    ...
```

This function must:

- compute the current equilibrium-centered error,
- compute `V_k`,
- test the candidate action,
- if accepted, return the original candidate action,
- if rejected, solve the correction optimization,
- return rich debug information.

### Task D: rich debug output

The debug dictionary must include at least:

- `accepted`,
- `accept_reason`,
- `reject_reason`,
- `V_k`,
- `V_next_cand`,
- `V_bound`,
- `rho`,
- `eps_lyap`,
- `u_cand`,
- `u_safe`,
- `u_s`,
- `x_s`,
- `e_x`,
- `candidate_bounds_ok`,
- `candidate_lyap_ok`,
- `solver_status`,
- `solver_residuals`,
- `trust_region_violation`,
- `slack_v`,
- `slack_u`.

### Task E: logging and integration hooks

Add logging so the surrounding notebook or run loop can store per-step Lyapunov diagnostics. The output format should be consistent and easy to plot.

### Task F: preserve generality

Do not hard-code this filter to RL only.

The caller should be able to pass a candidate input from any upstream controller. The report should explicitly state that the filter is controller-agnostic.

## Implementation guidance for equilibrium usage

Codex must preserve the following logic.

1. The target selector computes the moving equilibrium.
2. The Lyapunov function is centered at that equilibrium.
3. The candidate action is tested relative to `u_s(k)` and `x_s(k)`.
4. The correction optimization uses the same equilibrium frozen at time `k`.
5. The target must not be recomputed inside the filter optimization itself unless explicitly designed later.

## Recommended staged implementation order

### Stage 1

- Confirm the refined target selector is working.
- Confirm it returns `x_s`, `u_s`, `d_s`, and a useful debug structure.

### Stage 2

- Implement the Lyapunov matrix helper from `(A, B)`.
- Build a small standalone test that computes `V_k` and `V_next_pred(u)`.

### Stage 3

- Implement the acceptance check only.
- Do not yet optimize corrections.
- Log accepted and rejected actions.

### Stage 4

- Implement the one-step correction optimization.
- Verify numerical stability.

### Stage 5

- Integrate the filter into the main control loop.
- Keep the upstream controller unchanged.

### Stage 6

- Add plots and diagnostics.
- Compare accepted vs corrected actions, correction magnitudes, and Lyapunov trajectories.

## Acceptance criteria for Codex

The implementation is acceptable only if all of the following are true.

1. The code is modular and reusable.
2. The filter can wrap around any candidate control input.
3. The target selector remains the source of equilibrium information.
4. The Lyapunov helper is based on the physical subsystem first.
5. The acceptance condition is clearly implemented and logged.
6. The correction optimization is minimally invasive.
7. The code does not silently fail when the candidate action is rejected.
8. All solver statuses and numerical residuals are stored.
9. The output is easy to inspect in notebooks.
10. The LaTeX report is extensive, organized, and mathematically clear.

## Required LaTeX report content

Codex must create a standalone LaTeX report, for example:

```text
lyapunov_safety_filter_report.tex
```

The report should include the following sections.

1. Introduction and motivation.
2. Relation to offset-free MPC.
3. Disturbance-augmented model and observer.
4. Steady-state target selector.
5. Equilibrium-centered Lyapunov function.
6. Candidate-action acceptance rule.
7. Optimization-based correction step.
8. Connection to predictive safety filters and CLF filters.
9. Code structure and modified files.
10. Logging and diagnostics.
11. Limitations and open design questions.
12. Recommended next experiments.
13. References.

The report should include equations for:

- augmented dynamics,
- steady-state target equations,
- Lyapunov function,
- acceptance inequality,
- correction optimization problem.

The report should use clear notation and explain why this framework is **general** and not tied to RL only.

## Extra instructions for Codex

1. Do not make the implementation RL-specific.
2. Do not assume that the candidate action source is trustworthy.
3. Keep numerical robustness checks explicit.
4. Prefer clarity and debuggability over premature abstraction.
5. Add comments that explain the scientific role of each block.
6. If multiple possible insertion points exist, document the alternatives in the report.
7. If any existing file names differ from expectations, adapt while preserving the architecture described here.

## Suggested reference list for the LaTeX report

Use the following references in the report bibliography.

- Rawlings, J. B., Mayne, D. Q., and Diehl, M. *Model Predictive Control: Theory, Computation, and Design*, 2nd edition.
- Pannocchia, G., and Rawlings, J. B. "Disturbance models for offset-free model-predictive control," AIChE Journal, 2003.
- Pannocchia, G., and Bemporad, A. "Combined design of disturbance model and observer for offset-free model predictive control," IEEE TAC, 2007.
- Muske, K. R., and Badgwell, T. A. "Disturbance modeling for offset-free linear model predictive control," Journal of Process Control, 2002.
- Maeder, U., and Morari, M. "Offset-free reference tracking with model predictive control," Automatica, 2010.
- Limon, D., Alvarado, I., Alamo, T., and Camacho, E. F. "MPC for tracking piecewise constant references for constrained linear systems," Automatica, 2008.
- Wabersich, K. P., and Zeilinger, M. N. "A predictive safety filter for learning-based control of constrained nonlinear dynamical systems," Automatica, 2021.
- Ames, A. D., Xu, X., Grizzle, J. W., and Tabuada, P. "Control barrier function based quadratic programs for safety critical systems," IEEE TAC, 2017.
- Ames, A. D., Galloway, K., Sreenath, K., and Grizzle, J. W. "Rapidly exponentially stabilizing control Lyapunov functions and hybrid zero dynamics," IEEE TAC, 2014.
- Liu, J., Munoz de la Pena, D., and Christofides, P. D. "Lyapunov-based model predictive control of nonlinear systems subject to data losses," Automatica, 2009.

## Final instruction to Codex

Implement the framework in a way that is scientifically defensible, modular, well logged, and easy to test. The main goal is to create a **general Lyapunov safety filter around a moving offset-free equilibrium**, not merely to prove nominal MPC stability.
