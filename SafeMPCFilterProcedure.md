# Safe MPC Filter Procedure

This note is a shareable research package for the current safe-MPC filter path in this repository. It is written to be handed directly to ChatGPT or another research assistant so the model can reason about the implementation, the mathematics, the optimization problems, the numerical methods, and the open questions.

The active implementation path is the MPC-upstream Lyapunov safety filter:

- [Simulation/run_mpc_lyapunov.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Simulation/run_mpc_lyapunov.py)
- [Lyapunov/target_selector.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/target_selector.py)
- [Lyapunov/lyapunov_core.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/lyapunov_core.py)
- [Lyapunov/safety_filter.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/safety_filter.py)
- [Lyapunov/upstream_controllers.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/upstream_controllers.py)
- [Lyapunov/safety_debug.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/safety_debug.py)
- [Simulation/mpc.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Simulation/mpc.py)
- [utils/lyapunov_utils.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/utils/lyapunov_utils.py)
- [utils/scaling_helpers.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/utils/scaling_helpers.py)
- [utils/helpers.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/utils/helpers.py)

## 1. Scope

The controller architecture is:

$$
\text{baseline MPC candidate}
\;\rightarrow\;
\text{target selector}
\;\rightarrow\;
\text{Lyapunov acceptance test}
\;\rightarrow\;
\text{QCQP safety correction if needed}
\;\rightarrow\;
\text{fallback MPC if correction fails}
\;\rightarrow\;
\text{plant}.
$$

The current research questions are not just about stability. They are about the interaction between:

- the target-selector equilibrium package,
- the Lyapunov function centered on that package,
- the QCQP correction layer,
- and the baseline MPC candidate.

## 2. Coordinate Convention

The implementation is built in scaled deviation coordinates.

### 2.1 Input scaling

For any physical input vector $u^{\text{phys}}$, the code computes:

$$
u^{\text{scaled}} = \operatorname{apply\_min\_max}(u^{\text{phys}}, u_{\min}^{\text{data}}, u_{\max}^{\text{data}})
$$

and then uses deviation-from-steady-state coordinates:

$$
u^{\text{dev}} = u^{\text{scaled}} - u_{ss}^{\text{scaled}}.
$$

The same idea is used for outputs:

$$
y^{\text{dev}} = y^{\text{scaled}} - y_{ss}^{\text{scaled}}.
$$

This is one of the most important facts in the repository. Many controller bugs have come from mixing:

- physical units,
- min-max scaled values in $[0,1]$,
- RL-scaled values in $[-1,1]$,
- deviation coordinates,
- augmented coordinates.

## 3. Augmented Offset-Free Model

The safety filter uses an augmented linear model

$$
z_{k+1} = A_{\text{aug}} z_k + B_{\text{aug}} u_k,
\qquad
y_k = C_{\text{aug}} z_k,
$$

with

$$
z_k = \begin{bmatrix} x_k \\ d_k \end{bmatrix}.
$$

The repository assumes:

- $x_k \in \mathbb{R}^{n_x}$ is the physical state,
- $d_k \in \mathbb{R}^{n_y}$ is the disturbance state,
- $z_k = [x_k; d_k]$,
- and $\dim(d_k) = n_y$.

The estimated augmented state is stored as:

$$
\hat z_k = \begin{bmatrix} \hat x_k \\ \hat d_k \end{bmatrix}
$$

and is called `xhatdhat` in the code.

The physical submatrices extracted from the augmented model are:

$$
A = A_{\text{aug}}(1{:}n_x,1{:}n_x),
\qquad
B_d = A_{\text{aug}}(1{:}n_x,n_x+1{:}n_x+n_y),
$$

$$
B = B_{\text{aug}}(1{:}n_x,:),
\qquad
C = C_{\text{aug}}(:,1{:}n_x),
\qquad
C_d = C_{\text{aug}}(:,n_x+1{:}n_x+n_y).
$$

## 4. Baseline MPC Candidate

The upstream candidate comes from a baseline offset-free MPC rollout.

### 4.1 Candidate optimization problem

The MPC solver in [Simulation/mpc.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Simulation/mpc.py) evaluates the objective:

$$
J_{\text{MPC}}(U)
=
\sum_{j=1}^{N_P} \| y_{k+j|k} - y_{\text{sp}} \|_{Q_y}^2
\;+\;
\sum_{j=0}^{N_C-1} \| \Delta u_{k+j|k} \|_{R_u}^2
$$

where:

- $N_P$ is the prediction horizon,
- $N_C$ is the control horizon,
- $U = [u_k,\dots,u_{k+N_C-1}]$,
- $\Delta u_{k+j|k} = u_{k+j|k} - u_{k+j-1|k}$.

The candidate solve uses:

$$
x_{k+j+1|k} = A_{\text{aug}} x_{k+j|k} + B_{\text{aug}} u_{k+j|k}.
$$

### 4.2 Numerical method

The candidate is solved in [Lyapunov/upstream_controllers.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/upstream_controllers.py) with:

- `scipy.optimize.minimize(...)`,
- explicit bounds,
- optional nonlinear/linear constraints if supplied,
- and the current `IC_opt` as the optimizer initial guess.

The current notebook path uses a fixed zero `IC_opt` unless changed manually.

## 5. Refined Target Selector

The target selector is the key object that constructs the equilibrium package around which the Lyapunov function is defined.

### 5.1 What it returns

The target selector returns:

$$
\mathcal{T}_k =
\left(
x_s(k),\;
u_s(k),\;
d_s(k),\;
y_s(k)
\right)
$$

plus diagnostics:

- `success`
- `solve_stage`
- `target_error_inf`
- `target_slack_inf`
- `dyn_residual_inf`
- `bound_violation_inf`
- `selector_debug`

### 5.2 Disturbance freezing

The current target problem freezes:

$$
d_s(k) = \hat d_k.
$$

This means the target selector solves a steady-state problem around the current disturbance estimate, not around the nominal model.

### 5.3 Steady-state equations

The physical steady-state equations are:

$$
(I-A)x_s - Bu_s - B_d \hat d_k = 0
$$

and

$$
y_s = C x_s + C_d \hat d_k.
$$

If an output-selection matrix $H$ is supplied, the controlled output is:

$$
y_c = H y_s.
$$

In the current MPC safety-filter notebook, $H$ is not supplied, so effectively:

$$
H = I,
\qquad
y_c = y_s.
$$

### 5.4 Stage 1: exact target problem

The selector first attempts an exact target solve:

$$
\min_{x_s,u_s}
\;
(u_s-u_{\text{nom}})^\top R_u (u_s-u_{\text{nom}})
\;+\;
x_s^\top Q_x x_s
\;+\;
\Phi_y^{\text{soft}}(y_s)
$$

subject to

$$
(I-A)x_s - Bu_s - B_d \hat d_k = 0
$$

$$
u_{\ell} \le u_s \le u_u
$$

$$
y_c = y_{\text{sp}}
$$

and optional output bounds if they are enabled.

If output bounds are present and `soft_output_bounds=True`, the objective includes output-bound slacks:

$$
\Phi_y^{\text{soft}}(y_s)
=
(s_y^{\text{low}})^\top W_y^{\text{low}} s_y^{\text{low}}

+
(s_y^{\text{high}})^\top W_y^{\text{high}} s_y^{\text{high}}.
$$

In the current notebook path, explicit output bounds are not passed, so this soft-bound machinery is inactive even though the code supports it.

### 5.5 Stage 2: fallback target problem

If the exact target is not accepted, the selector solves a fallback target problem:

$$
\min_{x_s,u_s}
\;
(y_c-y_{\text{sp}})^\top T_y (y_c-y_{\text{sp}})
\;+\;
(u_s-u_{\text{nom}})^\top R_u (u_s-u_{\text{nom}})
\;+\;
x_s^\top Q_x x_s
$$

plus optional smoothing terms:

$$
\;+\;
(x_s-x_s^{\text{prev}})^\top Q_{\Delta x}(x_s-x_s^{\text{prev}})
\;+\;
(u_s-u_s^{\text{prev}})^\top R_{\Delta u}(u_s-u_s^{\text{prev}})
\;+\;
\Phi_y^{\text{soft}}(y_s)
$$

subject to

$$
(I-A)x_s - Bu_s - B_d \hat d_k = 0
$$

and

$$
u_{\ell} \le u_s \le u_u.
$$

This stage removes the hard equality $y_c = y_{\text{sp}}$ and instead minimizes the distance between the admissible equilibrium output and the requested setpoint.

### 5.6 Numerical method and acceptance

The selector is written in CVXPY and uses the solver preference sequence:

- `OSQP`
- `CLARABEL`
- `SCS`

unless overridden.

The selector only accepts:

- `optimal`
- `optimal_inaccurate`

and then additionally checks residuals and bound violations with internal tolerances.

This is important: a CVXPY solve can return `optimal` and still be rejected by the selector wrapper if its residual or bound checks are not accepted.

## 6. Lyapunov Function and Candidate Acceptance

Once the target package is available, the safety filter defines:

$$
e_x = \hat x_k - x_s,
\qquad
e_u = u - u_s.
$$

The quadratic Lyapunov function is:

$$
V(e_x) = e_x^\top P_x e_x
$$

where $P_x$ is produced from the linearized/augmented design in [Lyapunov/lyapunov_core.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/lyapunov_core.py).

For a candidate action $u_{\text{cand}}$, the filter predicts:

$$
e_x^+ = A e_x + B(u_{\text{cand}} - u_s)
$$

and checks the one-step bound:

$$
V(e_x^+) \le \rho V(e_x) + \varepsilon_{\text{lyap}}
$$

where $\varepsilon_{\text{lyap}} \ge 0$ is used as a numerical tolerance. The implemented helper uses

$$
V_{\text{next}} \le V_{\text{bound}},
\qquad
V_{\text{bound}} = \rho V_k + \varepsilon_{\text{lyap}}
$$

through the exact helper implementation in [Lyapunov/lyapunov_core.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/lyapunov_core.py).

The candidate is accepted only if all of the following hold:

- input bounds,
- move bounds,
- Lyapunov decrease condition.

## 7. QCQP Safety Correction

If the candidate fails the Lyapunov test and an effective target package is available, the filter solves a QCQP in [Lyapunov/safety_filter.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/safety_filter.py).

The effective target package is:

- the current selector target if `target_success=True`,
- otherwise the last valid target if `target_backup_policy="last_valid"`,
- otherwise unavailable.

### 7.1 Decision variable

The corrected input is:

$$
u \in \mathbb{R}^{n_u}.
$$

Optional slack variables are:

$$
s_V \ge 0
$$

for the Lyapunov constraint, and

$$
s_{\text{tr}} \ge 0
$$

for the trust region.

### 7.2 Predicted next state and output

The QCQP uses:

$$
e_x^+ = A e_x + B(u-u_s)
$$

and

$$
y^+ = C(x_s + e_x^+) + C_d d_s.
$$

### 7.3 QCQP objective

The correction objective is:

$$
J_{\text{QCQP}}(u)
=
\|u-u_{\text{cand}}\|_{W_{\text{cand}}}^2
\;+\;
\|u-u_{k-1}\|_{W_{\text{move}}}^2
\;+\;
\|u-u_s\|_{W_{\text{steady}}}^2
\;+\;
\|y^+ - y_{\text{track}}\|_{W_{\text{out}}}^2
$$

plus optional penalties:

$$
\;+\;
\lambda_V s_V^2
\;+\;
\lambda_{\text{tr}} s_{\text{tr}}^2.
$$

The tracking target $y_{\text{track}}$ depends on the normalized tracking-target policy used by the runners. In the current default notebook path:

$$
\texttt{tracking\_target\_policy} = \texttt{raw\_setpoint}
$$

so:

$$
y_{\text{track}} = y_{\text{sp}}
$$

for the upstream MPC objective, the fallback MPC objective, and the QCQP output term.

### 7.4 QCQP constraints

The QCQP enforces:

$$
u_{\min} \le u \le u_{\max}
$$

$$
\Delta u_{\min} \le u-u_{k-1} \le \Delta u_{\max}
$$

and the Lyapunov inequality:

$$
V(e_x^+) \le V_{\text{bound}}
$$

or, if Lyapunov slack is enabled,

$$
V(e_x^+) \le V_{\text{bound}} + s_V.
$$

In the default `lyap_acceptance_mode="hard_only"` path, a slacked QCQP solution is still post-checked against the hard Lyapunov test before it can be applied.

If a trust region is enabled, the filter also enforces:

$$
|u-u_{\text{cand}}| \le \delta_{\text{tr}}
$$

or, with trust-region slack,

$$
|u-u_{\text{cand}}| \le \delta_{\text{tr}} + s_{\text{tr}}.
$$

### 7.5 Numerical method

The QCQP uses CVXPY. Because the Lyapunov constraint is quadratic, the solver preference excludes OSQP and uses:

- `CLARABEL`
- `SCS`

unless overridden.

## 8. Important subtlety: slack does not automatically mean accepted correction

This is one of the most important research details in the current implementation.

Even if the QCQP uses a positive Lyapunov slack $s_V > 0$ and returns an `optimal` solution, the corrected action is **not** automatically accepted.

The code still post-checks the corrected action against the hard acceptance test. Therefore:

- QCQP can solve successfully,
- slack variables can be positive,
- but the corrected action can still be rejected,
- and the filter can still fall back to MPC.

This behavior is directly relevant to current debugging and is an important question for further research.

## 9. Fallback Logic

If the target is unavailable or the QCQP correction is not accepted, the filter falls back.

The default runtime fallback is a fresh baseline offset-free MPC candidate computed from:

- the current model state estimate,
- the current setpoint target used by policy,
- the same MPC object,
- the same bounds,
- the same constraint set.

Secondary fallbacks also exist in code, but the main path is the fresh MPC fallback.

## 10. Current file-on-disk hyperparameter snapshot

This is the current snapshot from the file on disk for [LyapunovSafetyFilterMPC.ipynb](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/LyapunovSafetyFilterMPC.ipynb).

### 10.1 Target-selector-related settings

- `Qs_tgt_diag = 1e8 * MPC_obj.Q_out`
- `Ru_tgt_diag = 1.0 * ones`
- `w_x_tgt = 1e-6`
- `Qdx_diag = None`
- `Rdu_diag = Rmove_diag`
- `u_nom_tgt = None` so the selector uses the zero vector
- `target_solver_pref = None` so the selector uses the default solver sequence

### 10.2 Safety-filter settings

- `rho_lyap = 0.98`
- `lyap_eps = 1e-9`
- `lyap_tol = 1e-10`
- `w_mpc = 1.0`
- `w_track = 1.0`
- `w_move = 0.2`
- `w_ss = 0.0`
- `allow_lyap_slack = True`
- `trust_region_delta = 0.15`
- `mpc_target_policy = "raw_setpoint"`
- `reuse_mpc_solution_as_ic = False`
- `reset_system_on_entry = True`

Important note: the notebook file on disk and the live notebook kernel can diverge if the notebook is open while files are being edited. In practice, any serious comparison must use a restarted kernel or explicit module reload.

## 11. Optimization methods actually used

This is the concise list of numerical methods used in the current implementation.

### 11.1 Upstream MPC candidate and MPC fallback

- Backend: `scipy.optimize.minimize`
- Objective: nonlinear Python objective assembled from MPC rollout
- Constraints: bounds and optional extra constraints

### 11.2 Target selector

- Backend: CVXPY
- Problem type: convex quadratic steady-state programs
- Solvers: `OSQP`, then `CLARABEL`, then `SCS`
- Additional acceptance logic: residual and bound checks after the solver returns

### 11.3 QCQP safety correction

- Backend: CVXPY
- Problem type: convex QCQP with quadratic Lyapunov constraint
- Solvers: `CLARABEL`, then `SCS`
- Additional acceptance logic: hard post-check after solve

## 12. Current open technical questions

These are the main questions to hand to ChatGPT.

### 12.1 Target-selector questions

1. Why does the selector spend so much time in `fallback` rather than `exact`, even when the plant can visibly reach and hold the setpoint?
2. When the selector fails after returning `optimal` from CVXPY, which acceptance check is most often responsible: dynamic residual, equality residual, or bound violation?
3. Should the selector use stronger smoothing on $x_s$ through an active $Q_{\Delta x}$ term?
4. Is the current choice `T_y = 1e8 Q_out` too aggressive and causing pathological steady-state targets near bounds?
5. Is the disturbance estimate $\hat d_k$ causing artificial infeasibility of the exact steady-state equations?

### 12.2 Safety-filter questions

1. Should the post-check continue to enforce the hard Lyapunov inequality after a slacked QCQP solve, or should slacked acceptance be treated as a separate certified mode?
2. Is `w_ss` best set to zero, small positive, or policy-dependent?
3. Should the trust region be active all the time, or only near setpoint changes?
4. Should `projection_active` distinguish between:
   - QCQP attempted,
   - QCQP solved but rejected,
   - QCQP solved and accepted?

### 12.3 Architecture questions

1. Should the QCQP objective track raw $y_{sp}$ while the Lyapunov certificate remains centered on $(x_s,u_s,d_s)$, or should both be centered on the admissible target?
2. Should the selector and safety filter share a stronger warm-start memory, such as the last valid target during temporary failures?
3. Should the target selector be replaced with a single robust feasibility-restoring formulation instead of the current exact/fallback split?

## 13. Paper-guided ideas for ChatGPT to explore

These are focused directions inspired by the offset-free MPC and safety-filter literature.

1. Compare the current target-selector design against the steady-state target formulations in Pannocchia and Rawlings (2003) and Pannocchia and Bemporad (2007). Ask whether the current acceptance and disturbance-freezing logic is too brittle.
2. Compare the repository’s fallback target problem against the offset-free tracking formulations in Limon et al. (2008) and Maeder and Morari (2010). Ask whether the current $T_y$ and $R_{\Delta u}$ choice biases the selector toward boundary solutions.
3. Compare the current QCQP with CLF-CBF-QP practice from Ames et al. Ask whether slacked CLF acceptance should remain hard-postchecked or whether a soft-certified mode should be introduced.
4. Compare the current one-step filter against predictive safety-filter ideas from Wabersich and Zeilinger (2021). Ask whether a short-horizon backup filter would reduce the sharpness of single-step corrections.
5. Ask whether the current design should move from a pure physical-state Lyapunov function to an augmented-state Lyapunov function that explicitly includes the disturbance estimate.
6. Ask whether there is a principled way to tune `trust_region_delta`, `w_move`, `w_ss`, and `w_track` based on reachable-set or tube-MPC arguments rather than trial and error.

## 14. References to give ChatGPT

Use these as the starting reference list.

1. G. Pannocchia and J. B. Rawlings, *Disturbance models for offset-free model-predictive control*, AIChE Journal, 49(2):426–437, 2003. DOI: `10.1002/aic.690490213`
2. G. Pannocchia and A. Bemporad, *Combined design of disturbance model and observer for offset-free model predictive control*, IEEE TAC, 52(6):1048–1053, 2007. DOI: `10.1109/TAC.2007.899096`
3. D. Limon, T. Alamo, I. Alvarado, and E. F. Camacho, *MPC for tracking piecewise constant references for constrained linear systems*, Automatica, 44(9), 2008. DOI: `10.1016/j.automatica.2008.01.023`
4. U. Maeder and M. Morari, *Offset-free reference tracking with model predictive control*, Automatica, 46(9):1469–1476, 2010. DOI: `10.1016/j.automatica.2010.05.023`
5. A. D. Ames et al., *Control barrier function based quadratic programs for safety critical systems*, IEEE TAC, 62(8):3861–3876, 2017. DOI: `10.1109/TAC.2016.2638961`
6. K. P. Wabersich and M. N. Zeilinger, *A predictive safety filter for learning-based control of constrained nonlinear dynamical systems*, Automatica, 129, 2021. DOI: `10.1016/j.automatica.2021.109597`
7. J. B. Rawlings, D. Q. Mayne, and M. M. Diehl, *Model Predictive Control: Theory, Computation, and Design*, Nob Hill Publishing.

## 15. Recommended prompt to give ChatGPT

You can give ChatGPT the following prompt together with this markdown file and the Python code bundle:

> This repository implements an MPC-upstream Lyapunov safety filter for a polymer CSTR in scaled deviation coordinates. Please analyze the target-selector formulation, the Lyapunov acceptance test, the QCQP correction layer, and the fallback logic. Focus on why the selector often remains in fallback or fails, why QCQP corrections sometimes degrade performance, and how the formulations could be redesigned using ideas from offset-free MPC, CLF-QP, and predictive safety-filter papers. Please be explicit about which changes are mathematical changes, which are numerical-solver changes, and which are debugging or instrumentation changes.
