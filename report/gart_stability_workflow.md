# Guarded Admissible Reachable Target Stability Workflow

This document defines the final Guarded Admissible Reachable Target Lyapunov MPC (GART-LMPC) stability workflow for the Polymer CSTR study up to the closed-loop MPC stability analysis. It is written in scaled-deviation coordinates unless stated otherwise.

## 1. Output-Disturbance Target Model

The GART target selector is formulated using the identified linear state-space model obtained from step-test response data, augmented with an output-disturbance state for offset-free correction. In steady state, an admissible target must satisfy

$$
x_s = A x_s + B u_s,
\qquad
y_s = C x_s + d_s .
$$

In the final workflow, the disturbance entering the target equation is not the raw observer estimate. The raw observer estimate is allowed to respond quickly for monitoring and diagnostics, but the Lyapunov certificate uses a bounded certified disturbance sequence.

Let $\hat d_k$ denote the raw observer disturbance estimate, and let $d^c_k$ denote the certified disturbance estimate. These quantities have different roles:

| Quantity | Meaning | Role |
| --- | --- | --- |
| $\hat d_k$ | Raw observer disturbance estimate | Monitoring, diagnostics, and later learning features |
| $d^c_k$ | Certified disturbance estimate | Target selector, terminal admissibility, and Lyapunov certification |

Therefore, the target equation used by GART is

$$
y_{s,k}=C x_{s,k}+d^c_k,
$$

rather than

$$
y_{s,k}=C x_{s,k}+\hat d_k.
$$

This separation is important because abrupt changes in $\hat d_k$ can otherwise move the selected steady target and therefore move the center of the Lyapunov function.

## 2. Certified Disturbance Projection

The implemented certificate uses a fixed symmetric bounded-rate projection for the certified disturbance. Given the previous certified value $d^c_{k-1}$ and the current raw observer estimate $\hat d_k$, the certified estimate is first moved toward $\hat d_k$ through a convex low-pass update. The update gain satisfies

$$
\alpha_d \in (0,1],
$$

so that $\alpha_d=1$ corresponds to the largest nominal correction toward the raw observer estimate, while smaller values of $\alpha_d$ produce a more gradual certified update.

Define the unprojected candidate update as

$$
d^{\mathrm{cand}}_k
=
d^c_{k-1}
+
\operatorname{clip}_{\Delta d_{\max}}
\left(
\alpha_d(\hat d_k-d^c_{k-1})
\right).
$$

Here, $\operatorname{clip}_{\Delta d_{\max}}(\cdot)$ denotes component-wise saturation of the disturbance increment:

$$
\left[
\operatorname{clip}_{\Delta d_{\max}}(v)
\right]_i
=
\min\{\max\{v_i,-\Delta d_{\max,i}\},\Delta d_{\max,i}\}.
$$

The certified disturbance is then projected onto the admissible disturbance set $\mathcal D$:

$$
d^c_k = \Pi_{\mathcal D}\left(d^{\mathrm{cand}}_k\right),
\qquad
\mathcal D = \{d: d_{\min}\le d\le d_{\max}\}.
$$

The projection is also applied component-wise:

$$
\left[\Pi_{\mathcal D}(v)\right]_i
=
\min\{\max\{v_i,d_{\min,i}\},d_{\max,i}\}.
$$

The resulting certified disturbance satisfies the component-wise rate bound

$$
\|d^c_k-d^c_{k-1}\|_\infty
\le
\|\Delta d_{\max}\|_\infty.
$$

Thus, the disturbance used by the target selector cannot move faster than the selected certified rate limit. In the reported experiments, the nominal symmetric disturbance-rate bound is used:

$$
\Delta d_{\max}=[0.1556478092,\;0.6892411884].
$$

This construction separates estimation from certification. The observer may update $\hat d_k$ according to its own estimation dynamics, but the target selector and Lyapunov certificate use only the bounded sequence $d^c_k$. Consequently, the target equation is driven by a disturbance signal with explicit amplitude and rate bounds.

## 3. Consequence For Target Motion

The certified disturbance projection is useful for the stability argument because the selected equilibrium target depends on the disturbance used in the target equation. Let the target selector be represented abstractly as the mapping

$$
(x_{s,k},u_{s,k},y_{s,k})
=
\mathcal S(y_{sp,k},d^c_k).
$$

For a fixed setpoint $y_{sp,k}$ and within a regular region of the target map, suppose that the steady-state target is locally Lipschitz with respect to the certified disturbance. Then there exists a constant $L_d>0$ such that

$$
\|x_{s,k}-x_{s,k-1}\|
\le
L_d\|d^c_k-d^c_{k-1}\|.
$$

Since the certified disturbance update enforces

$$
\|d^c_k-d^c_{k-1}\|
\le
\Delta d_{\max},
$$

the target displacement induced by the certified disturbance is bounded by

$$
\|x_{s,k}-x_{s,k-1}\|
\le
L_d\Delta d_{\max}.
$$

This argument shows why rate-limiting $d^c_k$ can, under regularity of the target map, imply bounded target motion. However, the controller used in this study does not estimate $L_d$ online and does not set the target-motion bound as $L_d\Delta d_{\max}$. Instead, it imposes target-motion limits directly as design constraints in the target selector:

$$
\|x_{s,k}-x_{s,k-1}\|_\infty \le 0.05,
\qquad
\|y_{s,k}-y_{s,k-1}\|_\infty \le 1.0.
$$

Equivalently, the implemented target selector uses

$$
d_{x_s}=0.05,
\qquad
d_{y_s}=1.0,
$$

rather than computing $d_{x_s}=L_d\Delta d_{\max}$. These constants are component-wise bounds in scaled-deviation coordinates. They were selected as design parameters from the closed-loop tuning study and provide a direct finite bound on the motion of the target sequence, independent of the local value of $L_d$.

The certified disturbance projection and the explicit target-motion constraints therefore play complementary roles. The certified disturbance projection bounds one source of target variation, while the direct $x_s$ and $y_s$ motion constraints impose the finite moving-center bound used in the practical-stability proof.

The target triple is denoted by

$$
(x_s,u_s,y_s).
$$

Here, $x_s$ is the steady state, $u_s$ is the steady input required to hold that state, and $y_s$ is the corresponding output under the certified disturbance correction.

## 4. Reachable Equilibrium Target Set

The target selector should not impose the raw setpoint as an equality constraint unless the corresponding steady-state target is reachable. Instead, it chooses a reachable steady target that is as close as possible to the requested setpoint.

For a fixed certified disturbance $d^c_k$, define the admissible equilibrium target set as

$$
\mathcal Z_k
=
\left\{
(x_s,u_s,y_s):
x_s=A x_s+B u_s,\;
y_s=Cx_s+d^c_k,\;
u_s\in\mathcal U_s
\right\}.
$$

The target selector is only allowed to choose $(x_s,u_s,y_s)\in\mathcal Z_k$. Therefore, the raw equality condition

$$
y_s = y_{sp,k}
$$

is not imposed directly. The appropriate target-selection question is instead:

$$
\text{choose }(x_s,u_s,y_s)\in\mathcal Z_k
\text{ such that }y_s\text{ is closest to }y_{sp,k}.
$$

This distinction is important when the requested setpoint is not exactly reachable under the steady-state equation, certified disturbance correction, and input constraints. In that case, forcing $y_s=y_{sp,k}$ would either make the target problem infeasible or push the selector toward a target that is not admissible for the MPC certificate.

All quantities in the GART target selector are expressed in the same scaled-deviation coordinates used by the identified linear state-space model and the LMPC optimization. Therefore, the equilibrium equations and input constraints above are applied to the scaled variables, not to the raw physical plant units.

## 5. Input Headroom And Tightened Steady Inputs

The steady input $u_s$ must remain inside a tightened input set rather than merely satisfying the physical input bounds. Let the scaled-deviation input bounds be

$$
u_{\min}\le u\le u_{\max}.
$$

The GART target selector defines a component-wise input headroom

$$
h_u
=
\delta_u (u_{\max}-u_{\min}),
$$

where $\delta_u$ is the input-headroom fraction. The tightened steady-input set is

$$
\mathcal U_s
=
\left\{
u_s:
u_{\min}+h_u+\kappa_u
\le
u_s
\le
u_{\max}-h_u-\kappa_u
\right\}.
$$

Here, $\kappa_u$ denotes the terminal-input tightening induced by the local terminal feedback law. In the present configuration, $\kappa_u$ is very small because the terminal radius lower bound is small, but it is retained in the formulation so that the steady target leaves room for the terminal controller.

The purpose of the tightened set is to avoid selecting a steady input that lies directly on the actuator boundary. If $u_s$ is already at a boundary, then a local stabilizing feedback move around the target can immediately violate the input constraint. The headroom condition therefore supports both recursive feasibility and Lyapunov certification by reserving actuator authority around the selected target.

In the reported experiments, the input-headroom fraction is

$$
\delta_u=0.01.
$$

With the current scaled-deviation input bounds, this gives

$$
h_u=[0.1996,\;0.1480],
$$

and the tightened steady-input bounds are approximately

$$
[-9.8003998,\;-7.3519996]
\le
u_s
\le
[9.7603998,\;7.1519996],
$$

where the inequalities are interpreted component by component.

## 6. Motivation For Lexicographic Target Optimization

The target selector should not trade setpoint mismatch against secondary regularization terms in a single primary weighted objective. A single weighted objective can allow the optimizer to make $y_s$ significantly worse only to obtain a smoother or more convenient steady input.

The desired ordering is hierarchical. The selector should first find the closest reachable output target. Only after the best reachable output mismatch has been determined should it use secondary criteria to choose among nearly equivalent steady targets.

This motivates a lexicographic target optimization. The term "lexicographic" means that objectives are ordered by priority: a lower-priority objective may improve the solution only inside a near-optimal shell of the higher-priority objective.

## 7. Stage 1: Closest Reachable Output

Let $r_k$ denote the command reference passed to the target selector. Before introducing the dynamic governor, $r_k$ can be interpreted as the raw setpoint $y_{sp,k}$. The first stage solves the closest-reachable-output problem:

$$
J_1^\star
=
\min_{(x_s,u_s,y_s)\in\mathcal Z_k}
\left\|
W_y(y_s-r_k)
\right\|_2^2 .
$$

Equivalently, using the target equations explicitly:

$$
\begin{aligned}
J_1^\star
=
\min_{x_s,u_s}
\quad&
\left\|
W_y(Cx_s+d^c_k-r_k)
\right\|_2^2 \\
\text{s.t.}\quad&
x_s=A x_s+B u_s,\\
&
u_s\in\mathcal U_s.
\end{aligned}
$$

When a previous accepted target is available, GART also imposes component-wise target-motion constraints:

$$
\|x_{s,k}-x_{s,k-1}\|_\infty \le d_{x_s},
\qquad
\|u_{s,k}-u_{s,k-1}\|_\infty \le d_{u_s},
\qquad
\|y_{s,k}-y_{s,k-1}\|_\infty \le d_{y_s}.
$$

These constraints are feasibility constraints, not objective penalties. They directly bound the motion of the steady target sequence. In the reported experiments,

$$
d_{x_s}=0.05,\qquad
d_{u_s}=[0.998,\;0.740],\qquad
d_{y_s}=1.0.
$$

The matrix $W_y$ weights the output channels in the primary target mismatch. In the current Polymer CSTR configuration,

$$
W_y=\operatorname{diag}(5,\;1),
$$

so the first output channel receives a larger target-selection weight than the second output channel.

If the raw command is reachable, then the Stage 1 optimum can satisfy $y_s=r_k$ and $J_1^\star=0$. If the raw command is not reachable under the equilibrium equations and constraints, Stage 1 returns the closest admissible steady output in the weighted norm.

## 8. Stage 2: Tie-Breaking Inside The Primary-Cost Shell

Stage 1 may admit multiple targets with the same or nearly the same output mismatch. Stage 2 chooses among these targets without materially degrading the primary objective. Define the near-optimal shell

$$
\left\|
W_y(Cx_s+d^c_k-r_k)
\right\|_2^2
\le
J_1^\star
+
\varepsilon_{\mathrm{abs}}
+
\varepsilon_{\mathrm{rel}}\max(1,J_1^\star).
$$

The numerical tolerances are

$$
\varepsilon_{\mathrm{abs}}=10^{-8},
\qquad
\varepsilon_{\mathrm{rel}}=10^{-6}.
$$

Within this shell, Stage 2 regularizes the steady input toward the previously applied input $u_{k-1}^{\mathrm{app}}$:

$$
\begin{aligned}
\min_{x_s,u_s}
\quad&
\left\|
W_u(u_s-u_{k-1}^{\mathrm{app}})
\right\|_2^2\\
\text{s.t.}\quad&
x_s=A x_s+B u_s,\\
&
u_s\in\mathcal U_s,\\
&
\left\|
W_y(Cx_s+d^c_k-r_k)
\right\|_2^2
\le
J_1^\star
+
\varepsilon_{\mathrm{abs}}
+
\varepsilon_{\mathrm{rel}}\max(1,J_1^\star),\\
&
\text{target-motion constraints hold.}
\end{aligned}
$$

In the reported experiments,

$$
W_u=\operatorname{diag}(1,\;1).
$$

Thus, after the closest reachable output is determined, the active secondary objective is consistency of $u_s$ with the previously applied input.

## 9. Tracking Error Decomposition

The selected target separates closed-loop tracking error into two interpretable parts:

$$
y_k-y_{sp,k}
=
(y_k-y_{s,k})
+
(y_{s,k}-y_{sp,k}).
$$

Consequently,

$$
\|y_k-y_{sp,k}\|
\le
\|y_k-y_{s,k}\|
+
\|y_{s,k}-y_{sp,k}\|.
$$

The first term is the controller-to-target tracking error. The second term is the target displacement error caused by reachability, input headroom, target-motion limits, and certification constraints. The final artifacts therefore report both the plant tracking error and the target mismatch, since good contraction around $y_s$ does not by itself imply exact tracking of an unreachable raw setpoint.

## 10. Contraction-Admissible Target Check

A target can satisfy the steady-state equations and input-headroom constraints but still be unsuitable for immediate Lyapunov contraction from the current state estimate. This can occur when the target is too far from the current observer state, when the feasible input range cannot produce the required decrease, or when the target lies in a region where the local stabilizing move has insufficient authority.

Let $\hat x_k$ denote the state component of the augmented observer estimate, and let $x_{s,k}$ be a candidate target state. The target-centered state error is

$$
e_k=\hat x_k-x_{s,k}.
$$

Define the quadratic Lyapunov function

$$
V(e)=e^\top P_x e,
$$

where $P_x\succ 0$ is the terminal Lyapunov matrix used by the direct Lyapunov MPC design. The current target-centered Lyapunov value is

$$
V_k
=
e_k^\top P_x e_k.
$$

For a hard first-step contraction certificate, the desired inequality is

$$
V(\hat x_{k+1}-x_{s,k})
\le
\rho V_k+\epsilon,
$$

where

$$
\hat x_{k+1}=A\hat x_k+B u_k.
$$

The target selector checks whether such a contraction is feasible before allowing the target to be used by LMPC. Given a candidate target, it solves the contraction probe

$$
V_{\min}
=
\min_{u\in\mathcal U}
\left(A\hat x_k+B u-x_{s,k}\right)^\top
P_x
\left(A\hat x_k+B u-x_{s,k}\right),
$$

where $\mathcal U=\{u:u_{\min}\le u\le u_{\max}\}$ is the full scaled-deviation input constraint set for the first applied move. The contraction bound is

$$
V_{\mathrm{bd}}
=
\rho V_k+\epsilon.
$$

The contraction-probe margin is defined with the sign convention

$$
m_{\mathrm{probe}}
=
V_{\mathrm{bd}}-V_{\min}.
$$

Thus, positive margin is favorable. The target is contraction-admissible when

$$
m_{\mathrm{probe}}\ge -\tau_c,
$$

where $\tau_c$ is a small numerical tolerance. In the reported experiments,

$$
\rho=0.98,
\qquad
\epsilon=10^{-3},
\qquad
\tau_c=10^{-8}.
$$

This check is a target-quality gate. If the target QP solves but the contraction probe fails, then the target is not accepted and is not passed to the LMPC solver. The implemented acceptance logic therefore distinguishes between a successful target solve and an accepted target:

$$
\text{solve success}
\;\not\Rightarrow\;
\text{accepted target}.
$$

The target is usable for LMPC only if it solves successfully, satisfies the terminal-input admissibility checks, and passes the contraction probe.

## 11. Constraint-Aware Dynamic Governor

The target selector first attempts to certify the raw setpoint directly. This corresponds to the command reference

$$
r_k=y_{sp,k}.
$$

If this candidate target is accepted, then the target selector uses the raw setpoint command. If it is not accepted and a previous accepted target exists, GART introduces a governed command that moves only partway from the previous accepted command toward the raw setpoint:

$$
r_k(\alpha)
=
r_{k-1}
+
\alpha\left(y_{sp,k}-r_{k-1}\right),
\qquad
\alpha\in[0,1].
$$

Here, $r_{k-1}$ is the previously accepted command reference stored in the target-selector state. The scalar $\alpha$ determines the fraction of the requested setpoint movement accepted at the current time step:

$$
\alpha=1 \Rightarrow r_k=y_{sp,k},
\qquad
\alpha=0 \Rightarrow r_k=r_{k-1}.
$$

For each candidate value of $\alpha$, the lexicographic target optimization is solved using $r_k(\alpha)$ in place of the raw setpoint. The resulting target must satisfy the same admissibility requirements as before:

$$
(x_{s,k},u_{s,k},y_{s,k})\in\mathcal Z_k,
$$

the target-motion constraints must hold, the terminal-input admissibility checks must hold, and the contraction probe must satisfy

$$
m_{\mathrm{probe}}\ge -\tau_c.
$$

The governor selects the largest admissible $\alpha$:

$$
\alpha_k^\star
=
\max_{\alpha\in[0,1]}
\alpha
\quad
\text{s.t. the target selected for }r_k(\alpha)
\text{ is accepted}.
$$

In practice, this one-dimensional search is performed by evaluating a descending grid of candidate values followed by bisection between the largest accepted value and the nearest larger rejected value. In the reported experiments, the grid is

$$
\mathcal A_G=\{1.0,\;0.75,\;0.5,\;0.25,\;0.0\},
$$

with eight bisection refinements.

If no positive governed move can be certified, GART evaluates the zero-move governor candidate:

$$
\alpha_k^\star=0,
\qquad
r_k=r_{k-1}.
$$

This action holds the command reference, but it does not reuse a stale target package. The target selector still solves the lexicographic target problem using the current certified disturbance $d^c_k$. Therefore, if the zero-move candidate is accepted, the returned target satisfies

$$
y_{s,k}=Cx_{s,k}+d^c_k.
$$

If even the zero-move candidate cannot be re-certified, then no target is declared usable for LMPC at that sampling instant. In that fallback case, the LMPC layer holds the previously applied input rather than optimizing around an inconsistent target. As the plant state moves closer to the held command reference, or as the certified disturbance evolves, a positive value of $\alpha$ may become admissible at a subsequent time step.

The governor also provides a useful decomposition of target displacement relative to the raw setpoint:

$$
y_{s,k}-y_{sp,k}
=
(y_{s,k}-r_k)
+
(r_k-y_{sp,k}).
$$

The first term is the residual target mismatch relative to the governed command. The second term is the intentional command-governor displacement from the raw setpoint. This decomposition separates physical reachability limitations from deliberate slowing of the command sequence.

## 12. Final GART-LMPC Optimization

After a target has been accepted, the closed-loop controller solves a finite-horizon Lyapunov MPC problem. The target selector and the MPC have different roles. The target selector constructs a certified equilibrium center

$$
(x_{s,k},u_{s,k},y_{s,k}),
$$

whereas the MPC objective in the final selected method tracks the raw requested setpoint $y_{sp,k}$. Thus the target is used for certification and terminal ingredients, while the performance objective remains tied to the external tracking task.

Let

$$
z_{i|k}
=
\begin{bmatrix}
x_{i|k} \\
d_{i|k}
\end{bmatrix}
$$

denote the augmented prediction state at prediction index $i$ from time $k$. The initial prediction state is the current observer estimate,

$$
z_{0|k}=\hat z_k
=
\begin{bmatrix}
\hat x_k \\
\hat d_k
\end{bmatrix}.
$$

The prediction model is the identified augmented output-disturbance state-space model:

$$
z_{i+1|k}=A_a z_{i|k}+B_a u_{i|k},
\qquad
y_{i+1|k}=C_a z_{i+1|k}.
$$

Here $A_a$, $B_a$, and $C_a$ denote the augmented matrices used by the closed-loop MPC prediction. The target selector uses the certified disturbance $d^c_k$ to define the admissible target. The MPC prediction is initialized from the observer state, so the raw observer disturbance remains part of the output prediction, while the Lyapunov center is the certified target state $x_{s,k}$.

The final raw GART-LMPC objective is

$$
\begin{aligned}
J_k
=
&\sum_{i=0}^{N_p-1}
\left\|y_{i+1|k}-y_{sp,k}\right\|_{Q_{\mathrm{raw}}}^{2}
\\
&+
\sum_{i=0}^{N_c-1}
\left\|u_{i|k}-u_{i-1|k}\right\|_{R_{\Delta}}^{2},
\end{aligned}
$$

with

$$
u_{-1|k}=u_{k-1}^{\mathrm{app}}.
$$

The first term penalizes predicted output error relative to the raw setpoint, not relative to $y_{s,k}$. The second term penalizes input movement from the previously applied input and between subsequent planned moves. For prediction indices beyond the control horizon, the last optimized control move is held in the standard receding-horizon manner.

The MPC constraints are

$$
\begin{aligned}
z_{i+1|k} &= A_a z_{i|k}+B_a u_{i|k}, \\
y_{i+1|k} &= C_a z_{i+1|k}, \\
u_{\min} &\le u_{i|k}\le u_{\max}.
\end{aligned}
$$

The first optimized input must also satisfy the hard Lyapunov contraction constraint centered at the accepted GART target:

$$
V(x_{1|k}-x_{s,k})
\le
\rho V(x_{0|k}-x_{s,k})+\epsilon.
$$

This is the same contraction form used in the target probe, but now it is enforced directly inside the MPC optimization for the actual applied move.

In the final runner, the terminal set is enabled. The terminal predicted state is constrained by

$$
V(x_{N_p|k}-x_{s,k})\le \alpha_s,
$$

where $\alpha_s$ is the input-admissible terminal level computed for the accepted target. This terminal constraint is passed to the optimizer when $\alpha_s$ is finite and larger than the numerical floor $\alpha_{\min}$.

The resulting final GART-LMPC problem is

$$
\begin{aligned}
\min_{\{z_{i|k},u_{i|k}\}}
\quad
&
\sum_{i=0}^{N_p-1}
\left\|y_{i+1|k}-y_{sp,k}\right\|_{Q_{\mathrm{raw}}}^{2}
+
\sum_{i=0}^{N_c-1}
\left\|u_{i|k}-u_{i-1|k}\right\|_{R_{\Delta}}^{2}
\\
\text{s.t.}\quad
&
z_{0|k}=\hat z_k,
\\
&
z_{i+1|k}=A_a z_{i|k}+B_a u_{i|k},
\\
&
y_{i+1|k}=C_a z_{i+1|k},
\\
&
u_{\min}\le u_{i|k}\le u_{\max},
\\
&
V(x_{1|k}-x_{s,k})
\le
\rho V(x_{0|k}-x_{s,k})+\epsilon,
\\
&
V(x_{N_p|k}-x_{s,k})\le \alpha_s
\quad \text{when the terminal set is active.}
\end{aligned}
$$

This formulation separates tracking and certification. The raw setpoint appears in the objective, while the accepted GART target appears in the Lyapunov contraction and terminal constraints. Consequently, the proof path is target-centered, but the measured performance remains raw-setpoint tracking.

## 13. Moving-Target Practical Stability Bound

In this section, $x_k$ denotes the state component used by the controller in the Lyapunov calculation. In the implemented output-feedback controller, this is the observer estimate $\hat x_k$.

For a fixed target, the Lyapunov contraction condition has the standard interpretation

$$
V(e_{k+1})\le \rho V(e_k)+\epsilon,
\qquad
0<\rho<1,
$$

where $e_k=x_k-x_s$. In GART-LMPC, however, the certified target is allowed to move from one sampling time to the next. Therefore the relevant target-centered error is

$$
e_k = x_k-x_{s,k}.
$$

After the first input is applied, the controller contracts toward the target used at time $k$:

$$
V(x_{k+1}-x_{s,k})
\le
\rho V(x_k-x_{s,k})+\epsilon.
$$

The next measured target-centered error is instead defined relative to the next accepted target:

$$
e_{k+1}=x_{k+1}-x_{s,k+1}.
$$

Let the target displacement be

$$
\delta x_{s,k}=x_{s,k+1}-x_{s,k}.
$$

Then

$$
\begin{aligned}
e_{k+1}
&=x_{k+1}-x_{s,k+1} \\
&=(x_{k+1}-x_{s,k})-(x_{s,k+1}-x_{s,k}) \\
&=(x_{k+1}-x_{s,k})-\delta x_{s,k}.
\end{aligned}
$$

This identity is the reason target-motion bounds are necessary. The controller may contract toward the current target, but the error at the next step also depends on how far the target center moves.

For the quadratic Lyapunov function $V(e)=e^\top P_x e$, the inequality

$$
V(a-b)
\le
(1+\mu)V(a)
+
\left(1+\frac{1}{\mu}\right)b^\top P_x b,
\qquad
\mu>0,
$$

gives

$$
\begin{aligned}
V(e_{k+1})
&\le
(1+\mu)V(x_{k+1}-x_{s,k})
+
\left(1+\frac{1}{\mu}\right)
\delta x_{s,k}^\top P_x \delta x_{s,k} \\
&\le
(1+\mu)\rho V(e_k)
+
(1+\mu)\epsilon
+
\left(1+\frac{1}{\mu}\right)
\delta x_{s,k}^\top P_x \delta x_{s,k}.
\end{aligned}
$$

Define

$$
\bar\rho=(1+\mu)\rho.
$$

The proof parameter $\mu$ must be chosen so that

$$
0<\bar\rho<1.
$$

Because the target selector enforces

$$
\|\delta x_{s,k}\|_\infty
\le
d_{x_s},
$$

we have

$$
\delta x_{s,k}^\top P_x \delta x_{s,k}
\le
\lambda_{\max}(P_x)\|\delta x_{s,k}\|_2^2
\le
\lambda_{\max}(P_x)n_x d_{x_s}^2.
$$

Thus

$$
V(e_{k+1})
\le
\bar\rho V(e_k)+w,
$$

where

$$
w
=
(1+\mu)\epsilon
+
\left(1+\frac{1}{\mu}\right)
\lambda_{\max}(P_x)n_x d_{x_s}^2.
$$

This is a practical-stability recursion. Iterating the recursion gives

$$
V(e_k)
\le
\bar\rho^k V(e_0)
+
\frac{1-\bar\rho^k}{1-\bar\rho}w.
$$

Therefore,

$$
\limsup_{k\to\infty}V(e_k)
\le
\frac{w}{1-\bar\rho}.
$$

The final method does not claim exact asymptotic convergence to a moving target. Instead, it gives a certified tube around the moving reachable target. The tube size is determined by the practical contraction tolerance $\epsilon$, the contraction rate $\rho$, and the target-motion bound $d_{x_s}$.

## 14. Raw Setpoint Tracking Bound

The Lyapunov argument is target-centered, while the performance goal is raw setpoint tracking. The output error decomposition from Section 9 gives

$$
y_k-y_{sp,k}
=
(y_k-y_{s,k})+(y_{s,k}-y_{sp,k}).
$$

The first term is the closed-loop error relative to the selected reachable target. The second term is the target displacement relative to the raw setpoint, caused by reachability limits or by the dynamic governor.

Using the output map associated with the identified model, the target-centered output error can be bounded in the form

$$
\|y_k-y_{s,k}\|
\le
\|C\|\,\|x_k-x_{s,k}\|+\varepsilon_{\mathrm{est},k},
$$

where $\varepsilon_{\mathrm{est},k}$ collects estimator mismatch, plant-model mismatch, and the difference between the certified disturbance used for target selection and the disturbance component active in the measured output. Therefore

$$
\|y_k-y_{sp,k}\|
\le
\|C\|\,\|e_k\|
+
\|y_{s,k}-y_{sp,k}\|
+
\varepsilon_{\mathrm{est},k}.
$$

This bound clarifies the interpretation of GART-LMPC results. Good raw setpoint tracking requires both a small target-centered closed-loop error and a small target-to-setpoint displacement. The target selector and governor reduce certification risk, while the raw LMPC objective keeps the closed-loop controller focused on the requested setpoint whenever the certified constraints allow it.

## 15. Interpretation Of The Stability Claim

The preceding bounds do not imply exact asymptotic convergence to the raw setpoint. Such a statement would generally be too strong for this setting because the raw setpoint may be unreachable under the steady-input constraints, the certified disturbance can vary over time, and the target sequence is intentionally governed when certification requires it.

The stability claim is instead a practical target-centered statement:

$$
\limsup_{k\to\infty}V(e_k)
\le
\frac{w}{1-\bar\rho}.
$$

Using

$$
\lambda_{\min}(P_x)\|e_k\|_2^2
\le
V(e_k),
$$

the corresponding state-error tube satisfies

$$
\limsup_{k\to\infty}\|e_k\|_2
\le
\sqrt{
\frac{w}{(1-\bar\rho)\lambda_{\min}(P_x)}
}.
$$

Combining this with the raw setpoint tracking decomposition gives the external tracking interpretation

$$
\limsup_{k\to\infty}\|y_k-y_{sp,k}\|
\le
\|C\|
\sqrt{
\frac{w}{(1-\bar\rho)\lambda_{\min}(P_x)}
}
+
\limsup_{k\to\infty}\|y_{s,k}-y_{sp,k}\|
+
\limsup_{k\to\infty}\varepsilon_{\mathrm{est},k}.
$$

Thus the raw tracking error is bounded by three terms:

- the controller-to-target stability tube;
- the target-to-setpoint displacement caused by reachability and governor action;
- the residual estimator/model mismatch.

This is the appropriate conclusion for the final GART-LMPC method. The method certifies contraction around an admissible reachable target and uses the raw setpoint in the MPC objective. Therefore, when the target selector can keep $y_{s,k}$ close to $y_{sp,k}$ and the estimator residual remains small, the measured output tracks the raw setpoint closely. When the raw setpoint is temporarily unreachable or the governor slows the command, the same bound separates that target displacement from the controller's target-centered stability behavior.

## 16. GART-Aware Reinforcement Learning Step

The GART-LMPC certificate can also be used as the safety layer for reinforcement learning. The main principle is that the actor may propose an exploratory input, but the input applied to the plant must satisfy the same hard Lyapunov contraction condition used by the certified MPC path.

The actor receives a GART-aware observation containing the plant estimate, the certified disturbance, the raw setpoint, the governed command, the selected reachable target, the previous input, and the current contraction margin. A compact representation is

$$
o_k^{\mathrm{GART}}
=
\left[
\hat z_k,\;
d^c_k,\;
y_{sp,k},\;
r_k,\;
y_{s,k},\;
u_{s,k},\;
u_{k-1}^{\mathrm{app}},\;
m_{\mathrm{probe},k}
\right].
$$

In implementation, these components are scaled consistently with the existing TD3 observation convention. The certified disturbance $d^c_k$ is included separately because it is the disturbance used by the target selector and Lyapunov certificate, while $\hat z_k$ contains the raw augmented observer state.

The actor proposes a candidate input

$$
u_k^{\mathrm{RL}}
=
\pi_\theta(o_k^{\mathrm{GART}})+\zeta_k,
$$

where $\pi_\theta$ is the actor policy and $\zeta_k$ is the exploration signal. The applied input is obtained by a hard GART action projection:

$$
\begin{aligned}
u_k^{\mathrm{safe}}
=
\arg\min_{u\in\mathcal U}
\quad
&
\left\|u-u_k^{\mathrm{RL}}\right\|_{W_c}^{2}
\\
\text{s.t.}\quad
&
V(x_{k+1}(u)-x_{s,k})
\le
\rho V(\hat x_k-x_{s,k})+\epsilon,
\end{aligned}
$$

where $x_{k+1}(u)$ is the one-step state prediction under input $u$. If the actor proposal already satisfies the input constraints and the Lyapunov contraction constraint, then

$$
u_k^{\mathrm{safe}}=u_k^{\mathrm{RL}}.
$$

Otherwise, the projection returns the closest certified input in the metric $W_c$. The correction

$$
\Delta u_k^{\mathrm{safe}}
=
u_k^{\mathrm{safe}}-u_k^{\mathrm{RL}}
$$

is a useful learning signal. A small correction means that the actor proposed an input consistent with the certificate. A large correction means that the actor attempted to move outside the certified action set.

For replay storage, the transition should retain both the actor proposal and the executed input:

$$
\mathcal T_k
=
\left(
o_k^{\mathrm{GART}},
u_k^{\mathrm{RL}},
u_k^{\mathrm{safe}},
r_k,
x_{s,k},
u_{s,k},
y_{s,k},
d^c_k,
m_{\mathrm{probe},k},
\ell_k,
o_{k+1}^{\mathrm{GART}}
\right),
$$

where $\ell_k$ denotes the scalar reward or stage loss used by the RL update. Storing both $u_k^{\mathrm{RL}}$ and $u_k^{\mathrm{safe}}$ prevents a hidden-action problem: the actor learns what it proposed, what was actually applied, and how much correction was required for certification.

Because the GART target selector has memory, pretraining data should also be generated sequentially. The target at time $k$ depends on the previous accepted command and target state:

$$
(r_{k-1},x_{s,k-1},u_{s,k-1},y_{s,k-1},d^c_{k-1}).
$$

Therefore, expert labels should be generated from rollouts that carry this target memory forward. A memoryless dataset of independent random states can be useful for coverage, but it does not fully represent the deployed target-selection map. The preferred pretraining label is

$$
u_k^{\star}
=
\kappa_{\mathrm{GART}}
\left(
o_k^{\mathrm{GART}},
r_{k-1},
x_{s,k-1},
u_{s,k-1},
y_{s,k-1},
d^c_{k-1}
\right),
$$

where $\kappa_{\mathrm{GART}}$ denotes the certified expert action produced by the GART-LMPC or by the hard GART action projection.

The exploration amplitude should also be certificate-aware. One practical form is

$$
\zeta_k
=
\sigma(m_{\mathrm{probe},k})\,\varepsilon_k,
\qquad
\varepsilon_k\sim\mathcal N(0,I),
$$

with

$$
\sigma(m)
=
\sigma_{\max}
\operatorname{clip}
\left(
\frac{m}{m_{\mathrm{scale}}},
0,
1
\right).
$$

This keeps exploration small near the certificate boundary and allows larger exploratory moves when the contraction margin is comfortably positive.

The RL layer therefore does not alter the stability certificate as long as the plant receives $u_k^{\mathrm{safe}}$ rather than the uncertified proposal $u_k^{\mathrm{RL}}$. The actor can change the candidate action distribution, exploration policy, and replay data distribution, but the proof path remains tied to the executed input satisfying

$$
V(x_{k+1}(u_k^{\mathrm{safe}})-x_{s,k})
\le
\rho V(\hat x_k-x_{s,k})+\epsilon.
$$

Thus, RL is treated as a performance-improvement layer around the certified controller, not as a replacement for the Lyapunov certificate. Exploration is permitted only through actions that pass the GART hard-contraction projection.

## Parameter And Notation Table

This table collects the notation introduced so far. It will be extended as the workflow document is expanded. The main text uses mathematical notation; implementation identifiers are included only when they help connect the notation to the code.

| Symbol / parameter | Meaning | Current value / setting | Notes |
| --- | --- | ---: | --- |
| $x_k$ | Identified state-space model state at time $k$ | Model-dependent | Scaled-deviation coordinate |
| $\hat x_k$ | Observer state estimate used by GART | Computed online | State component of augmented observer state |
| $u_k$ | Manipulated input at time $k$ | 2 inputs | Scaled-deviation coordinate |
| $u_{k-1}^{\mathrm{app}}$ | Previously applied manipulated input | Measured from closed-loop rollout | Scaled-deviation coordinate; Stage 2 input-smoothing reference |
| $y_k$ | Controlled output at time $k$ | 2 outputs | Scaled-deviation coordinate |
| $A$ | Identified discrete-time state matrix | From step-test model identification | Used in target and LMPC predictions |
| $B$ | Identified discrete-time input matrix | From step-test model identification | Used in target and LMPC predictions |
| $C$ | Identified output matrix | From step-test model identification | Maps state to output before disturbance correction |
| $z_{i|k}$ | Augmented MPC prediction state | $[x_{i|k};d_{i|k}]$ | Used by the closed-loop prediction model |
| $A_a$ | Augmented prediction state matrix | From output-disturbance state-space model | Used in GART-LMPC rollout |
| $B_a$ | Augmented prediction input matrix | From output-disturbance state-space model | Used in GART-LMPC rollout |
| $C_a$ | Augmented prediction output matrix | From output-disturbance state-space model | Maps augmented state to predicted output |
| $N_p$ | MPC prediction horizon | $9$ | Final GART-LMPC runner setting |
| $N_c$ | MPC control horizon | $3$ | Final GART-LMPC runner setting |
| $\mathcal U$ | Full feasible input set for the applied move | $u_{\min}\le u\le u_{\max}$ | Used in contraction probe and LMPC input constraints |
| $\mathcal Z_k$ | Admissible equilibrium target set at time $k$ | Defined online by $d^c_k$ and $\mathcal U_s$ | Set from which GART selects the target |
| $\mathcal U_s$ | Tightened steady-input set | $u_{\min}+h_u+\kappa_u\le u_s\le u_{\max}-h_u-\kappa_u$ | Reserves actuator authority around $u_s$ |
| $r_k$ | Command reference passed to target selector | Raw setpoint before governor discussion | Later becomes governed reference candidate |
| $r_{k-1}$ | Previously accepted command reference | Stored in target state | Used by the dynamic governor |
| $r_k(\alpha)$ | Governed command candidate | $r_{k-1}+\alpha(y_{sp,k}-r_{k-1})$ | Replaces $y_{sp,k}$ inside target selection when raw command is not accepted |
| $\alpha$ | Candidate governor fraction | $[0,1]$ | Fraction of requested setpoint movement |
| $\alpha_k^\star$ | Selected governor fraction | Solved online | Largest certified admissible governor fraction |
| $\mathcal A_G$ | Governor grid | $\{1.0,\ 0.75,\ 0.5,\ 0.25,\ 0.0\}$ | Evaluated in descending order |
| $N_{\mathrm{bis}}$ | Governor bisection refinements | $8$ | Used after grid bracketing |
| $u_{\min}$ | Lower scaled-deviation input bound | $[-10.0,\ -7.5]$ | Component-wise |
| $u_{\max}$ | Upper scaled-deviation input bound | $[9.96,\ 7.30]$ | Component-wise |
| $\delta_u$ | Input-headroom fraction | $0.01$ | Used to compute $h_u$ |
| $h_u$ | Component-wise input headroom | $[0.1996,\ 0.1480]$ | $h_u=\delta_u(u_{\max}-u_{\min})$ |
| $\kappa_u$ | Terminal-input tightening | $[2.12{\times}10^{-7},\ 4.10{\times}10^{-7}]$ | Retained for terminal-feedback admissibility |
| $u_{s,\min}^{tight}$ | Tight lower steady-input bound | $[-9.8003998,\ -7.3519996]$ | Includes headroom and terminal tightening |
| $u_{s,\max}^{tight}$ | Tight upper steady-input bound | $[9.7603998,\ 7.1519996]$ | Includes headroom and terminal tightening |
| $d_{u_s}$ | Direct steady-input target-motion bound | $[0.998,\ 0.740]$ | Component-wise bound on $u_s$ changes between accepted targets |
| $W_y$ | Stage 1 output mismatch weight | $\operatorname{diag}(5,\ 1)$ | Primary target-selection weight |
| $W_u$ | Stage 2 input tie-breaker weight | $\operatorname{diag}(1,\ 1)$ | Penalizes deviation of $u_s$ from $u_{k-1}^{\mathrm{app}}$ |
| $Q_{\mathrm{raw}}$ | Raw setpoint tracking weight in LMPC | $\operatorname{diag}(5,\ 1)$ | Penalizes $y_{i|k}-y_{sp,k}$ |
| $R_{\Delta}$ | Input movement weight in LMPC | $\operatorname{diag}(1,\ 1)$ | Penalizes planned input increments |
| $J_1^\star$ | Optimal Stage 1 primary target mismatch | Solved online | Defines the Stage 2 near-optimal shell |
| $\varepsilon_{\mathrm{abs}}$ | Absolute primary-shell tolerance | $10^{-8}$ | Stage 2 may only move inside this shell |
| $\varepsilon_{\mathrm{rel}}$ | Relative primary-shell tolerance | $10^{-6}$ | Scales with $\max(1,J_1^\star)$ |
| $P_x$ | Terminal Lyapunov matrix | $7\times7$ matrix | Diagonal approximately $[0.00512,\ 0.02054,\ 0.02025,\ 0.01558,\ 5.0,\ 5.0,\ 5.0]$ |
| $e_k$ | Target-centered state error used by the controller | $\hat x_k-x_{s,k}$ in implementation | Argument of the Lyapunov function |
| $V(e)$ | Quadratic Lyapunov function | $e^\top P_x e$ | Used for target probe and hard LMPC contraction |
| $V_k$ | Current target-centered Lyapunov value | Computed online | $e_k^\top P_x e_k$ |
| $\delta x_{s,k}$ | Target-state displacement | $x_{s,k+1}-x_{s,k}$ | Bounded by $d_{x_s}$ |
| $\mu$ | Moving-target proof tuning parameter | Analysis choice | Must satisfy $(1+\mu)\rho<1$ |
| $\bar\rho$ | Effective moving-target contraction factor | $(1+\mu)\rho$ | Must be less than 1 |
| $w$ | Practical-stability residual bound | Analysis expression | Depends on $\epsilon$, $P_x$, $n_x$, and $d_{x_s}$ |
| $\lambda_{\max}(P_x)$ | Largest eigenvalue of $P_x$ | Computed from terminal matrix | Used in moving-target bound |
| $\lambda_{\min}(P_x)$ | Smallest eigenvalue of $P_x$ | Computed from terminal matrix | Converts Lyapunov value to a state-error norm bound |
| $\varepsilon_{\mathrm{est},k}$ | Output-estimation/model residual | Analysis residual | Accounts for estimator and plant-model mismatch |
| $\rho$ | Hard Lyapunov contraction factor | $0.98$ | Reported experiment setting |
| $\epsilon$ | Practical contraction tolerance | $10^{-3}$ | Reported experiment setting |
| $V_{\min}$ | Minimum achievable next-step Lyapunov value in target probe | Solved online | Optimized over $u\in\mathcal U$ |
| $V_{\mathrm{bd}}$ | First-step contraction bound | $\rho V_k+\epsilon$ | Target probe acceptance bound |
| $m_{\mathrm{probe}}$ | Contraction-probe margin | $V_{\mathrm{bd}}-V_{\min}$ | Positive is favorable |
| $\tau_c$ | Probe-margin numerical tolerance | $10^{-8}$ | Accept target when $m_{\mathrm{probe}}\ge-\tau_c$ |
| $\alpha_s$ | Terminal Lyapunov level | Computed online | Applied when above the terminal-level numerical floor |
| $\alpha_{\min}$ | Minimum terminal-level threshold | $10^{-8}$ | Terminal set is skipped below this value |
| `terminal_set_on` | Terminal-set enable flag | `True` | Final runner enables the terminal set conditionally through $\alpha_s>\alpha_{\min}$ |
| $x_s$ | Steady target state | Optimized online | Must satisfy $x_s=A x_s+B u_s$ |
| $u_s$ | Steady target input | Optimized online | Subject to input bounds and headroom |
| $y_s$ | Certified reachable target output | Optimized online | $y_s=Cx_s+d^c$ |
| $d_s$ | Generic steady output-disturbance term | Replaced by $d^c_k$ in final workflow | Appears in the general target model |
| $\hat d_k$ | Raw observer disturbance estimate | Computed online by observer | Used for monitoring, diagnostics, and later learning features |
| $d^c_k$ | Certified disturbance estimate | Computed online by bounded projection | Used by target selector and Lyapunov certification |
| $d^{\mathrm{cand}}_k$ | Candidate certified-disturbance update before projection | Computed online | Projected onto $\mathcal D$ |
| $\alpha_d$ | Low-pass update gain for certified disturbance | $0.2$ | Must satisfy $\alpha_d\in(0,1]$ |
| $\Delta d_{\max}$ | Certified disturbance rate bound | $[0.1556478092,\ 0.6892411884]$ | Component-wise; scaled-deviation disturbance coordinates |
| $L_d$ | Local Lipschitz constant from certified disturbance to target state | Analysis constant | Used to obtain $\|x_{s,k}-x_{s,k-1}\|\le L_d\Delta d_{\max}$ |
| $d_{\min}$ | Lower disturbance bound | $[-4.06547624,\ -28.25906582]$ | Component-wise; scaled-deviation disturbance coordinates |
| $d_{\max}$ | Upper disturbance bound | $[3.71691422,\ 6.20299360]$ | Component-wise; scaled-deviation disturbance coordinates |
| $\mathcal D$ | Admissible certified-disturbance set | $\{d:d_{\min}\le d\le d_{\max}\}$ | Projection set for $d^c_k$ |
| $\Pi_{\mathcal D}(\cdot)$ | Projection onto admissible disturbance bounds | Component-wise clipping | Keeps $d^c_k$ inside $\mathcal D$ |
| $y_{sp,k}$ | Raw setpoint at time $k$ | Generated by the direct-runner setpoint schedule | Performance objective tracks this raw setpoint |
| $\mathcal S(\cdot)$ | GART target-selection map | Solved online | Maps $(y_{sp,k},d^c_k)$ to $(x_{s,k},u_{s,k},y_{s,k})$ |
| $d_{x_s}$ | Direct target-state motion bound | $0.05$ | Component-wise scaled-deviation bound; chosen directly, not computed as $L_d\Delta d_{\max}$ |
| $d_{y_s}$ | Direct target-output motion bound | $1.0$ | Component-wise scaled-deviation bound |
| $\sigma_d$ | Disturbance-rate scaling factor | $1.0$ | Fixed symmetric setting; implementation parameter `d_rate_scale` |
| $o_k^{\mathrm{GART}}$ | GART-aware RL observation | Constructed online | Contains observer state, $d^c_k$, setpoint, governed command, target, previous input, and margin |
| $\pi_\theta$ | RL actor policy | TD3 actor | Maps $o_k^{\mathrm{GART}}$ to a candidate input |
| $\zeta_k$ | Exploration signal | Training parameter | May be scaled by contraction margin |
| $u_k^{\mathrm{RL}}$ | Actor-proposed input | Computed online | Candidate input before certification |
| $u_k^{\mathrm{safe}}$ | Certified executed input | Solved online | Closest hard-certified input to $u_k^{\mathrm{RL}}$ |
| $W_c$ | RL action-projection correction weight | Identity by default | Implementation parameter `candidate_weight_diag` |
| $\Delta u_k^{\mathrm{safe}}$ | Safety correction | $u_k^{\mathrm{safe}}-u_k^{\mathrm{RL}}$ | Stored for learning diagnostics |
| $\mathcal T_k$ | Replay transition with GART diagnostics | Stored per RL step | Includes proposed and executed actions |
| $\ell_k$ | RL reward or stage loss | Reward-function dependent | Scalar signal used by TD3 update |
| $u_k^\star$ | Expert pretraining label | Generated by GART expert | Should be generated with sequential target memory |
| $\kappa_{\mathrm{GART}}$ | Certified GART expert map | GART-LMPC or hard action projection | Used for pretraining labels |
| $\sigma(m)$ | Margin-aware exploration scale | Design choice | Increases with positive contraction margin |
| $\sigma_{\max}$ | Maximum exploration scale | Training parameter | Upper bound on exploration amplitude |
| $m_{\mathrm{scale}}$ | Margin normalization scale | Training parameter | Sets margin at which exploration reaches $\sigma_{\max}$ |
