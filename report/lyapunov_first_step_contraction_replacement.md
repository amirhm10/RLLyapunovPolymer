# Lyapunov First-Step-Contraction Replacement

## Purpose

This note documents the controller used in:

- `LyapunovFirstStepContractionMPC.ipynb`
- `LyapunovFirstStepContractionRL.ipynb`

These notebooks reuse the same refined selector, effective-target reuse, tracking-target policy, and debug/export pipeline as the safety-filter notebooks, but they do **not** use QCQP projection.

Instead, when the candidate input violates the hard Lyapunov first-step test, the controller activates a constrained MPC solve with the **same baseline MPC objective** and only **one extra hard constraint** on the first predicted physical-state step.

There is no fallback controller in this path. If the constrained MPC replacement solve is infeasible, the original candidate is still applied and the event is logged explicitly.

## Model Split and Coordinates

The controller works in the same scaled deviation coordinates as the existing MPC and safety-filter code.

The augmented model is

\[
x_{k+1}^{\text{aug}} = A_{\text{aug}} x_k^{\text{aug}} + B_{\text{aug}} u_k,
\qquad
y_k = C_{\text{aug}} x_k^{\text{aug}}.
\]

The augmented state is partitioned as

\[
x_k^{\text{aug}} =
\begin{bmatrix}
x_k \\
d_k
\end{bmatrix},
\]

where:

- \(x_k \in \mathbb{R}^{n_x}\) is the physical-state estimate
- \(d_k \in \mathbb{R}^{n_d}\) is the disturbance or offset part

The observer estimate stored in the code is `xhatdhat` or `xhat_aug`.

## Step A: Refined Target Selector

At each control step \(k\), the refined selector computes an admissible steady target package

\[
(x_s, u_s, d_s, y_s, r_s).
\]

This is the same selector used in the safety-filter notebooks. It solves a soft steady-state problem centered by:

- the raw setpoint \(y_{sp,k}\)
- the currently applied input \(u_{k-1}\)
- the previous steady target \((x_{s,\text{prev}}, u_{s,\text{prev}})\)
- the current observer state \(xhat_k\)

with disturbance fixed at the current estimate:

\[
d_s = \hat d_k.
\]

The selector returns:

- `target_info`: current selector result
- `effective_target_info`: current target if valid, otherwise the last valid target if `target_backup_policy="last_valid"`

The Lyapunov logic below is always centered on the **effective** target.

## Baseline Candidate Input

The candidate action depends on the notebook:

### MPC notebook

The candidate is the ordinary offset-free MPC action:

\[
u_k^{\text{cand}} = u_{0|k}^{\text{MPC}}.
\]

This is produced by the same upstream MPC objective already used in the safety notebook. The objective is **unchanged**.

### RL notebook

The candidate is the TD3 policy action:

\[
u_k^{\text{cand}} = \pi_\theta(s_k).
\]

The RL state construction and selector/effective-target logic remain aligned with the safety RL path.

## Physical-State Lyapunov Function

Let the physical-state tracking error relative to the effective steady target be

\[
e_{x,k} = x_k - x_s.
\]

The Lyapunov function is

\[
V(e_x) = e_x^\top P_x e_x,
\]

where \(P_x \succ 0\) is the physical-state Lyapunov matrix built by the same Lyapunov-ingredient design used elsewhere in the repository.

The hard first-step bound is

\[
V(e_{x,k+1}) \le \rho \, V(e_{x,k}) + \varepsilon_{\text{lyap}},
\]

with:

- \(0 < \rho < 1\)
- \(\varepsilon_{\text{lyap}} \ge 0\)

The implementation keeps the same tolerance semantics already used in the project:

\[
V_{\text{bound},k} = \rho V_k + \varepsilon_{\text{lyap}}.
\]

## Candidate Hard Check

For the candidate input \(u_k^{\text{cand}}\), the controller predicts the next physical-state error using the steady target input \(u_s\):

\[
e_{x,k+1}^{\text{cand}}
= A_x e_{x,k} + B_x (u_k^{\text{cand}} - u_s).
\]

Then it computes

\[
V_k = e_{x,k}^\top P_x e_{x,k},
\qquad
V_{k+1}^{\text{cand}}
= (e_{x,k+1}^{\text{cand}})^\top P_x e_{x,k+1}^{\text{cand}},
\]

and the candidate contraction margin

\[
\Delta V_k^{\text{cand}} = V_{k+1}^{\text{cand}} - (\rho V_k + \varepsilon_{\text{lyap}}).
\]

The candidate passes the hard test when

\[
\Delta V_k^{\text{cand}} \le \text{tol}_{\text{lyap}}.
\]

If the candidate passes, it is applied directly.

## Constrained MPC Replacement

If the candidate fails the hard first-step Lyapunov test, the controller solves a new MPC problem with:

- the **same objective** as the baseline MPC
- the **same input bounds**
- the **same prediction model**
- one additional hard nonlinear inequality on the **first** predicted physical-state step only

### Optimization problem

Let \(U_k\) denote the full MPC decision vector. The constrained replacement problem is

\[
\min_{U_k} J_{\text{MPC}}(U_k; y_{sp,k}, u_{k-1}, x_k^{\text{aug}})
\]

subject to all ordinary baseline MPC constraints and:

\[
V\!\left(x_{1|k}^{\text{phys}} - x_s\right)
\le
\rho \, V\!\left(x_k^{\text{phys}} - x_s\right)
 \varepsilon_{\text{lyap}}.
\]

In expanded quadratic form:

\[
(x_{1|k}^{\text{phys}} - x_s)^\top P_x (x_{1|k}^{\text{phys}} - x_s)
\le
\rho (x_k^{\text{phys}} - x_s)^\top P_x (x_k^{\text{phys}} - x_s)
 \varepsilon_{\text{lyap}}.
\]

This constraint is imposed only on \(x_{1|k}^{\text{phys}}\), not on later predicted states and not on the terminal state.

## Applied Action Logic

The implemented control law is:

1. Compute `target_info`.
2. Resolve `effective_target_info`.
3. Compute the candidate input.
4. Hard-check the candidate against the first-step Lyapunov inequality.
5. If it passes, apply the candidate.
6. If it fails, solve the constrained replacement MPC.
7. If that solve succeeds, apply the constrained MPC action.
8. If that solve fails, apply the original candidate anyway and log that the replacement failed.

So this controller is **not** a fallback-MPC controller. It is a **candidate-accept or constrained-replace** controller.

## Notebook-Specific Interpretation

### `LyapunovFirstStepContractionMPC.ipynb`

- candidate source: ordinary offset-free MPC
- replacement source: constrained MPC with first-step contraction
- if replacement fails: apply the original unconstrained MPC candidate

### `LyapunovFirstStepContractionRL.ipynb`

- candidate source: TD3 action
- replacement source: constrained MPC with first-step contraction
- if replacement fails: apply the original TD3 candidate

## Difference From the Old Safety-Filter Workflow

The old safety-filter notebooks used:

1. candidate input
2. hard Lyapunov verification
3. QCQP projection if needed
4. possible fallback MPC

The new notebooks use:

1. candidate input
2. hard Lyapunov verification
3. constrained MPC replacement if needed
4. no fallback controller
5. candidate still applied if replacement solve is infeasible

So the conceptual change is:

- old path: **verify, then project**
- new path: **verify, then re-solve MPC with one hard Lyapunov constraint**

## Logged Diagnostics

The debug exporter stores the usual safety-style artifacts plus first-step-replacement-specific signals:

- `candidate_first_step_lyap_ok`
- `first_step_contraction_triggered`
- `constrained_mpc_attempted`
- `constrained_mpc_solved`
- `constrained_mpc_applied`
- `constrained_mpc_failed_applied_candidate`
- `V_k`
- `V_next_first_candidate`
- `V_next_first_applied`
- `V_bound`
- `contraction_margin_candidate`
- `contraction_margin_applied`
- `first_step_contraction_satisfied_applied`

These are used in the plots to distinguish:

- candidate accepted directly
- candidate replaced by constrained MPC
- replacement failed and original candidate applied

## Implementation Mapping

The main implementation pieces are:

- `Simulation/run_mpc_first_step_contraction.py`
  - MPC-upstream candidate and replacement logic
- `Simulation/run_rl_lyapunov.py`
  - RL candidate path with `projection_backend="first_step_contraction_mpc"`
- `Lyapunov/upstream_controllers.py`
  - constrained replacement helper and constrained-MPC solve
- `Lyapunov/safety_debug.py`
  - shared export, tables, and plots for the new status signals

## Important Consequence

This controller does **not** guarantee that the applied action satisfies the hard Lyapunov inequality in every step, because when the constrained replacement solve is infeasible, the original violating candidate is still applied by design.

So the new notebooks should be interpreted as:

- **Lyapunov-guided replacement when possible**
- **candidate passthrough when replacement is infeasible**

not as a strict safety certificate in the same sense as a successful hard-feasibility filter.
