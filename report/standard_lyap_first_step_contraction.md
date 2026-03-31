# Standard Lyapunov MPC With First-Step Contraction

## Purpose
This note describes the new standard Lyapunov MPC variant implemented in the maintained `Lyapunov/` path. The design goal is:

- keep the **existing standard tracking MPC objective unchanged**,
- keep the **current refined target selector**,
- add **one hard Lyapunov contraction constraint only on the first predicted step**,
- avoid the safety-filter architecture entirely.

This controller is therefore different from the safety-filter path. The contraction property is enforced **inside the MPC solve**, not by post-processing a candidate input afterward.

## 1. Model Split and Coordinates

The maintained controller uses the augmented linear model

\[
z_{k+1} = A_{\mathrm{aug}} z_k + B_{\mathrm{aug}} u_k,
\qquad
y_k = C_{\mathrm{aug}} z_k,
\]

with augmented state

\[
z_k = \begin{bmatrix} x_k \\ d_k \end{bmatrix},
\]

where:

- \(x_k \in \mathbb{R}^{n_x}\) is the physical-state estimate,
- \(d_k \in \mathbb{R}^{n_y}\) is the disturbance/output-offset estimate.

The maintained code splits the augmented matrices as

\[
A_{\mathrm{aug}} =
\begin{bmatrix}
A & B_d \\
0 & I
\end{bmatrix},
\qquad
B_{\mathrm{aug}} =
\begin{bmatrix}
B \\ 0
\end{bmatrix},
\qquad
C_{\mathrm{aug}} =
\begin{bmatrix}
C & C_d
\end{bmatrix}.
\]

In the implementation, the Lyapunov contraction is defined on the **physical-state error only**, not on the full augmented state.

## 2. Target Selector Step

At each control step \(k\), the refined target selector computes a steady package

\[
(x_s, u_s, d_s, y_s, r_s),
\]

using the current observer estimate \(z_k = [x_k; d_k]\), the current setpoint request, and the previous target for smoothing.

The selector solves a steady-state optimization subject to:

\[
x_s = A x_s + B u_s + B_d d_s,
\]
\[
y_s = C x_s + C_d d_s,
\]

plus input and optional output bounds/tightening.

The objective remains the refined Step A objective already implemented in the selector:

\[
\|r_s - y_{sp}\|_{Q_r}^2
+
\|u_s - u_{\mathrm{applied},k}\|_{R_{u,\mathrm{ref}}}^2
+
\|u_s - u_{s,\mathrm{prev}}\|_{R_{\Delta u,\mathrm{sel}}}^2
+
\|x_s - x_{s,\mathrm{prev}}\|_{Q_{\Delta x}}^2
+
\|x_s - \hat x_k\|_{Q_{x,\mathrm{ref}}}^2.
\]

This controller does **not** change the selector. It reuses it exactly as the center generator.

## 3. Existing Standard Tracking MPC Objective

After the target selector returns \((x_s,u_s,d_s)\), the standard tracking MPC solves over the future input sequence

\[
U = \{u_{0|k}, u_{1|k}, \dots, u_{N_C-1|k}\}.
\]

The augmented prediction model is

\[
z_{j+1|k} = A_{\mathrm{aug}} z_{j|k} + B_{\mathrm{aug}} u_{c(j)|k},
\]

where

\[
c(j) = \min(j, N_C-1).
\]

The predicted output is

\[
y_{j|k} = C_{\mathrm{aug}} z_{j|k}.
\]

The maintained standard objective is **kept exactly unchanged**:

\[
J(U) =
\sum_{j=1}^{N_P} \| y_{j|k} - y_{\mathrm{target}} \|_{Q_y}^2
+
\sum_{j=0}^{N_C-1} \| u_{j|k} - u_s \|_{S_u}^2
+
\sum_{j=0}^{N_C-1} \| \Delta u_{j|k} \|_{R_{\Delta u}}^2
+
\| x_{N_P|k} - x_s \|_{P_x}^2 \cdot \gamma_{\mathrm{terminal}}.
\]

Here:

- \(y_{\mathrm{target}}\) is whatever the standard rollout already uses for tracking,
- \(u_s\) is the steady target input,
- \(P_x\) is the existing terminal/Lyapunov matrix,
- \(\gamma_{\mathrm{terminal}}\) is the existing terminal cost scale.

So this new controller is **not** a new objective design. It is the old standard objective plus one extra constraint.

## 4. Physical-State Lyapunov Function

Define the physical-state tracking error relative to the current selected steady target:

\[
e_{x,k} = x_k - x_s.
\]

The Lyapunov function is

\[
V(e_x) = e_x^\top P_x e_x,
\]

where \(P_x \succ 0\) is the same matrix already used by the standard Lyapunov MPC terminal ingredients.

At the current time step, the current Lyapunov value is

\[
V_k = V(e_{x,k}) = (x_k - x_s)^\top P_x (x_k - x_s).
\]

The contraction bound is

\[
V_{\mathrm{bound}} = \rho \, V_k + \varepsilon_{\mathrm{lyap}},
\qquad 0 < \rho < 1,\quad \varepsilon_{\mathrm{lyap}} \ge 0.
\]

In the maintained codebase, \(\varepsilon_{\mathrm{lyap}}\) is treated as a numerical tolerance term, not as a negative contraction margin.

## 5. New First-Step Contraction Constraint

Only the **first predicted physical-state step** is constrained.

Let the first predicted physical state be

\[
x_{1|k}.
\]

Then the first predicted physical-state error relative to the selected steady target is

\[
e_{x,k+1|k} = x_{1|k} - x_s.
\]

The new hard constraint is

\[
e_{x,k+1|k}^\top P_x e_{x,k+1|k}
\le
\rho \, e_{x,k}^\top P_x e_{x,k}
+
\varepsilon_{\mathrm{lyap}}.
\]

Equivalently,

\[
V(e_{x,k+1|k}) \le \rho V(e_{x,k}) + \varepsilon_{\mathrm{lyap}}.
\]

This is enforced **once only**, using the first predicted state.

There are no analogous constraints on:

- \(x_{2|k}\),
- \(x_{3|k}\),
- \(\dots\),
- \(x_{N_P|k}\).

So this controller is not a full horizon-contraction MPC. It is a standard tracking MPC with one one-step Lyapunov contraction gate built into the solve.

## 6. Full Optimization Problem

At each control step, after the target selector succeeds, the controller solves

\[
\min_{U,\;Z}
J(U)
\]

subject to:

\[
z_{0|k} = z_k,
\]

\[
z_{j+1|k} = A_{\mathrm{aug}} z_{j|k} + B_{\mathrm{aug}} u_{c(j)|k},
\qquad j = 0,\dots,N_P-1,
\]

\[
u_{\min} \le u_{j|k} \le u_{\max},
\qquad j = 0,\dots,N_C-1,
\]

plus the existing terminal set constraint **if the standard configuration already turns it on**, and in addition:

\[
(x_{1|k} - x_s)^\top P_x (x_{1|k} - x_s)
\le
\rho (x_k - x_s)^\top P_x (x_k - x_s)
+
\varepsilon_{\mathrm{lyap}}.
\]

Nothing else about the objective is changed.

## 7. Why Only the First Step?

This design is intentionally different from both:

- a safety filter, and
- a full horizon Lyapunov-constrained MPC.

Why only the first step:

1. It forces the **applied move** to respect a Lyapunov decrease condition.
2. It keeps the optimization problem close to the existing standard controller.
3. It avoids over-constraining the whole horizon.
4. It is the most direct MPC-side analogue of a one-step Lyapunov filter, but enforced inside the solve.

So the philosophy is:

- the target selector defines the local center,
- the MPC objective still handles performance,
- the first applied move is forced to be contractive.

## 8. Difference From Other Controllers

### Baseline MPC
- no target selector steady package used as a Lyapunov center,
- no Lyapunov constraint,
- pure performance objective.

### Current standard Lyapunov MPC
- uses the target selector,
- uses standard tracking objective and terminal ingredients,
- may use terminal cost and optional terminal set,
- does **not** require first-step contraction.

### Safety-filter MPC
- computes a candidate controller action first,
- then applies a separate Lyapunov acceptance/correction/fallback layer after the candidate is proposed.

### This new controller
- uses the target selector,
- keeps the standard tracking objective,
- adds one hard first-step contraction constraint **inside** the MPC solve,
- does not use the safety filter.

## 9. Implementation Mapping

The implementation is split into three layers.

### Solver layer
`Lyapunov/lyapunov_core.py`

- existing standard solver stays untouched,
- new sibling solver adds:
  - `rho_lyap`
  - `eps_lyap`
  - `first_step_contraction_on`
- the first-step quadratic constraint is formed using the first predicted physical-state block.

### Rollout layer
`Lyapunov/run_lyap_mpc.py`

- existing standard rollout stays untouched,
- new sibling rollout:
  - calls the same refined target selector each step,
  - calls the new solver,
  - logs contraction quantities per step.

### Notebook layer
`StandardLyapMPCFirstStepContraction.ipynb`

- import-driven experiment entrypoint,
- uses the maintained `Lyapunov/` modules,
- exposes the contraction parameters and the current standard MPC settings.

## 10. Step Diagnostics

The new path should record:

- \(V_k\)
- \(V_{k+1|k}\) for the first predicted step
- \(V_{\mathrm{bound}}\)
- contraction margin
  \[
  V_{k+1|k} - V_{\mathrm{bound}}
  \]
- whether the first-step contraction was satisfied
- solver status and rejection reason

These are the right quantities to inspect when the new constraint makes the problem infeasible or degrades performance.

## 11. Expected Failure Mode

The new controller can become infeasible even when the old standard controller was feasible, because:

- the old controller only optimized performance and terminal structure,
- the new controller also demands an immediate one-step contraction.

So the key tradeoff is:

- better local Lyapunov discipline on the applied move,
- at the cost of potentially more infeasible steps or more conservative inputs.

That tradeoff is the central experimental question for this controller.
