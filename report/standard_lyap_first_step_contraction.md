# MPC Upstream With First-Step Lyapunov Contraction

## Purpose

This note describes the maintained first-step-contraction controller after the architecture correction.

The controller is **not** the older standard-Lyapunov MPC stack. It now follows the same high-level path as the MPC safety-filter notebook:

1. compute the refined Step A target selector at every step,
2. build the same effective target using current target or last valid target,
3. solve the same upstream offset-free MPC problem,
4. add only one extra hard Lyapunov inequality on the **first predicted physical-state step**,
5. do **not** run any QCQP projection,
6. if the constrained MPC solve fails, fall back to ordinary offset-free MPC.

So the change is:

- same selector and target-handling logic as the safety path,
- same MPC objective as baseline MPC,
- no post-projection safety correction,
- one hard first-step Lyapunov gate inside the upstream MPC solve.

## 1. Augmented MPC Model

The controller uses the same augmented offset-free linear model as the MPC safety notebook:

\[
z_{k+1} = A_{\mathrm{aug}} z_k + B_{\mathrm{aug}} u_k,
\qquad
y_k = C_{\mathrm{aug}} z_k,
\]

with augmented state

\[
z_k =
\begin{bmatrix}
x_k \\
d_k
\end{bmatrix},
\]

where:

- \(x_k \in \mathbb{R}^{n_x}\) is the physical-state estimate,
- \(d_k \in \mathbb{R}^{n_d}\) is the disturbance / output-offset estimate.

The matrix split is

\[
A_{\mathrm{aug}} =
\begin{bmatrix}
A & B_d \\
0 & I
\end{bmatrix},
\qquad
B_{\mathrm{aug}} =
\begin{bmatrix}
B \\
0
\end{bmatrix},
\qquad
C_{\mathrm{aug}} =
\begin{bmatrix}
C & C_d
\end{bmatrix}.
\]

All control computations are done in scaled deviation coordinates, exactly as in the baseline MPC and safety-MPC paths.

## 2. Refined Step A Target Selector

At time step \(k\), the same refined selector as the safety notebook computes a steady package

\[
(x_s, u_s, d_s, y_s, r_s).
\]

It solves the steady-state equations

\[
x_s = A x_s + B u_s + B_d d_s,
\]
\[
y_s = C x_s + C_d d_s,
\]

with the same tightened input / optional output bounds and the same refined objective:

\[
\|r_s - y_{sp,k}\|_{Q_r}^2
+
\|u_s - u_{\mathrm{applied},k}\|_{R_{u,\mathrm{ref}}}^2
+
\|u_s - u_{s,\mathrm{prev}}\|_{R_{\Delta u,\mathrm{sel}}}^2
+
\|x_s - x_{s,\mathrm{prev}}\|_{Q_{\Delta x}}^2
+
\|x_s - \hat{x}_k\|_{Q_{x,\mathrm{ref}}}^2.
\]

The selector itself is unchanged.

### Effective target reuse

The controller uses the same effective-target logic as the safety path:

- if the current selector solve succeeds, use the current target;
- otherwise, if `target_backup_policy="last_valid"` and a previous successful target exists, reuse that last valid target;
- otherwise, there is no effective Lyapunov center available.

So the controller center is

\[
(x_s^{\mathrm{eff}}, u_s^{\mathrm{eff}}, d_s^{\mathrm{eff}}, y_s^{\mathrm{eff}}, r_s^{\mathrm{eff}}),
\]

coming from either the current selector result or the last valid selector result.

## 3. Tracking target for the MPC objective

The upstream MPC still uses the same tracking-target policy surface as the safety notebook:

- `raw_setpoint`
- `selector_reference`
- `admissible_if_available`
- `admissible_on_fallback`

So the stage target supplied to the ordinary MPC objective is

\[
y_{\mathrm{track},k},
\]

selected by the same policy used in the safety notebook.

By default, the notebook keeps

\[
y_{\mathrm{track},k} = y_{sp,k}.
\]

## 4. Baseline MPC Objective Remains Unchanged

The decision variable is the future move sequence

\[
U =
\{u_{0|k}, u_{1|k}, \dots, u_{N_C-1|k}\}.
\]

Predictions use the same offset-free model:

\[
z_{j+1|k} = A_{\mathrm{aug}} z_{j|k} + B_{\mathrm{aug}} u_{c(j)|k},
\qquad
c(j)=\min(j,N_C-1).
\]

The predicted output is

\[
y_{j|k} = C_{\mathrm{aug}} z_{j|k}.
\]

The objective is exactly the baseline offset-free MPC objective already used by `MpcSolver.mpc_opt_fun(...)`:

\[
J(U)
=
\sum_{j=1}^{N_P} \|y_{j|k} - y_{\mathrm{track},k}\|_{Q_y}^2
+
\sum_{j=0}^{N_C-1} \|\Delta u_{j|k}\|_{R_{\Delta u}}^2.
\]

Nothing is added to this objective:

- no \(u_s\)-centering term,
- no terminal quadratic term,
- no terminal set term,
- no QCQP correction cost.

The only change is one extra hard constraint.

## 5. Lyapunov Function and First-Step Contraction Bound

Using the same physical-state Lyapunov matrix \(P_x\) as the safety path, define the physical-state error relative to the effective selector center:

\[
e_{x,k} = x_k - x_s^{\mathrm{eff}}.
\]

The Lyapunov function is

\[
V(e_x) = e_x^\top P_x e_x.
\]

So the current Lyapunov value is

\[
V_k = (x_k - x_s^{\mathrm{eff}})^\top P_x (x_k - x_s^{\mathrm{eff}}).
\]

The bound is

\[
V_{\mathrm{bound},k} = \rho_{\mathrm{lyap}} V_k + \varepsilon_{\mathrm{lyap}},
\]

with

\[
0 < \rho_{\mathrm{lyap}} < 1,
\qquad
\varepsilon_{\mathrm{lyap}} \ge 0.
\]

In this codebase, \(\varepsilon_{\mathrm{lyap}}\) is a numerical tolerance term, so the implemented inequality is

\[
V_{\mathrm{next}} \le \rho_{\mathrm{lyap}} V_k + \varepsilon_{\mathrm{lyap}}.
\]

## 6. The Only New Constraint

Let the first predicted augmented state be

\[
z_{1|k} =
\begin{bmatrix}
x_{1|k} \\
d_{1|k}
\end{bmatrix}.
\]

Only the physical part is used in the Lyapunov gate:

\[
e_{x,k+1|k} = x_{1|k} - x_s^{\mathrm{eff}}.
\]

The single additional hard inequality is

\[
(x_{1|k} - x_s^{\mathrm{eff}})^\top P_x (x_{1|k} - x_s^{\mathrm{eff}})
\le
\rho_{\mathrm{lyap}}
(x_k - x_s^{\mathrm{eff}})^\top P_x (x_k - x_s^{\mathrm{eff}})
+
\varepsilon_{\mathrm{lyap}}.
\]

Equivalently,

\[
V(e_{x,k+1|k}) \le \rho_{\mathrm{lyap}} V(e_{x,k}) + \varepsilon_{\mathrm{lyap}}.
\]

This constraint is enforced **only for the first predicted step**.

There are no analogous constraints on:

- \(x_{2|k}\),
- \(x_{3|k}\),
- terminal state \(x_{N_P|k}\).

So this controller is:

- not a QCQP safety filter,
- not a full-horizon Lyapunov-constrained MPC,
- not the older terminal-set standard Lyapunov MPC.

It is simply baseline MPC with one first-step Lyapunov contraction inequality.

## 7. Full constrained upstream-MPC problem

When an effective target exists and `first_step_contraction_on=True`, the controller solves

\[
\min_{U} J(U)
\]

subject to

\[
z_{0|k} = \hat z_k,
\]
\[
z_{j+1|k} = A_{\mathrm{aug}} z_{j|k} + B_{\mathrm{aug}} u_{c(j)|k},
\qquad j=0,\dots,N_P-1,
\]
\[
u_{\min} \le u_{j|k} \le u_{\max},
\qquad j=0,\dots,N_C-1,
\]
plus any ordinary MPC constraints already present in `cons`, and the additional first-step Lyapunov inequality

\[
(x_{1|k} - x_s^{\mathrm{eff}})^\top P_x (x_{1|k} - x_s^{\mathrm{eff}})
\le
\rho_{\mathrm{lyap}}
(x_k - x_s^{\mathrm{eff}})^\top P_x (x_k - x_s^{\mathrm{eff}})
+
\varepsilon_{\mathrm{lyap}}.
\]

The numerical optimization is still an ordinary `scipy.optimize.minimize(...)` solve on the same MPC decision vector, but with the above nonlinear inequality appended.

## 8. No projection / no QCQP

This path intentionally removes the post-candidate correction stage:

- no QCQP correction,
- no trust region,
- no Lyapunov slack variable,
- no projection-active step.

The accepted action is either:

1. the constrained upstream MPC candidate, or
2. an ordinary fallback MPC input.

## 9. Fallback behavior

If the constrained MPC solve cannot be used:

- because there is no effective target,
- or because the constrained optimization is infeasible / unsuccessful,

the controller falls back to ordinary offset-free MPC with the same tracking target

\[
y_{\mathrm{track},k}.
\]

That fallback MPC is **not** re-solved with another Lyapunov constraint.

It is then evaluated for diagnostics:

- if it happens to satisfy the Lyapunov test, it is labeled as verified fallback;
- otherwise it is labeled as unverified fallback.

So fallback is ordinary MPC behavior, not a second constrained optimization and not a QCQP projection.

## 10. Relationship to the safety MPC notebook

This controller shares with the safety MPC notebook:

- the same refined selector,
- the same selector config surface,
- the same effective-target reuse logic,
- the same tracking-target policy logic,
- the same warm-start handling,
- the same debug/export bundle structure.

It differs from the safety MPC notebook in exactly one architectural point:

- the safety notebook solves ordinary MPC first and then may correct it with QCQP projection,
- this notebook solves constrained MPC directly and never runs QCQP projection.

## 11. What the debug/export means in this path

Because the same safety debug/export pipeline is reused:

- target-selector plots mean the same thing as in the safety notebook,
- output/input/fallback plots mean the same thing as in the safety notebook,
- QCQP/projection plots should remain inactive,
- the dedicated contraction diagnostics plot is the key new figure.

The important first-step contraction diagnostics are:

- \(V_k\)
- \(V_{1|k}\) stored as `V_next_first`
- \(V_{\mathrm{bound}}\)
- `contraction_margin = V_next_first - V_bound`
- `first_step_contraction_satisfied`

## 12. Implementation mapping

The maintained implementation is split as:

- constrained upstream MPC solve helper:
  `Lyapunov/upstream_controllers.py`
- MPC-upstream rollout with refined selector and fallback logic:
  `Simulation/run_mpc_first_step_contraction.py`
- notebook entrypoint:
  `StandardLyapMPCFirstStepContraction.ipynb`
- debug/export:
  `Lyapunov/safety_debug.py`

This is now the intended first-step-contraction implementation path.
