# Governed-Reference Direct Lyapunov MPC Methodology

Date: 2026-05-22

## Objective

This report documents Proposal 01, the governed artificial reference target layer for the polymer CSTR direct Lyapunov MPC study. The goal is to test whether making the internal admissible command explicit improves the scientific alignment between:

- the raw process setpoint, $y_{sp,k}$,
- the admissible command actually certified by the target layer, $r_k$,
- the steady Lyapunov target, $(x_{s,k},u_{s,k},y_{s,k})$,
- and the physical closed-loop output, $y_k$.

No result claim is made here. This report defines the new methodology and the diagnostics that should be used after the first run.

## Motivation

The current direct Lyapunov MPC target selector can produce a feasible target $y_s$ that differs from the raw setpoint $y_{sp}$. The raw tracking error decomposes as:

$$
y_k-y_{sp,k} = (y_k-y_{s,k}) + (y_{s,k}-y_{sp,k}).
$$

The Lyapunov contraction condition only certifies behavior around $x_s$ and therefore around $y_s$. It does not by itself guarantee that $y_s$ remains close to the raw command $y_{sp}$. This creates the hidden target-command mismatch observed in the previous direct LMPC and safety-gated RL studies.

Proposal 01 makes the target modification explicit:

$$
y_{sp,k} \rightarrow r_k \rightarrow (x_{s,k},u_{s,k},y_{s,k}).
$$

The new logged command $r_k$ is the closest admissible command found by the governor layer. This separates infeasible-command handling from Lyapunov tracking.

## Coordinates And Plant Setup

The polymer CSTR uses the same setup as `DirectLyapunovMPC.py`.

The manipulated inputs are:

$$
u =
\begin{bmatrix}
Q_c \\
Q_m
\end{bmatrix}.
$$

The controlled outputs are:

$$
y =
\begin{bmatrix}
\eta \\
T
\end{bmatrix}.
$$

The controller operates in scaled deviation coordinates. If $u^{phys}$ and $y^{phys}$ are physical variables, the control variables are formed by min-max scaling and subtracting the steady-state anchors:

$$
u_k = S_u(u^{phys}_k)-S_u(u^{phys}_{ss}),
$$

$$
y_k = S_y(y^{phys}_k)-S_y(y^{phys}_{ss}).
$$

The augmented offset-free model is:

$$
x^a_k =
\begin{bmatrix}
x_k \\
\hat d_k
\end{bmatrix},
$$

with output prediction:

$$
\hat y_k = Cx_k + \hat d_k.
$$

The disturbance estimate $\hat d_k$ is treated as frozen inside the steady target calculation.

## Step 1: Governed Command

The governed command layer computes an admissible reference $r_k$ before computing the steady Lyapunov target. In the first implementation, the decision variables are $x_g$ and $u_g$, with:

$$
r_k = Cx_g + \hat d_k.
$$

The optimization is:

$$
\begin{aligned}
\min_{x_g,u_g}\quad
& \|Cx_g+\hat d_k-y_{sp,k}\|_{W_r}^2
+ \lambda_r\|Cx_g+\hat d_k-r_{k-1}\|_2^2 \\
\text{s.t.}\quad
& x_g = Ax_g + Bu_g, \\
& u_{\min}^{h} \le u_g \le u_{\max}^{h}.
\end{aligned}
$$

The tightened bounds are:

$$
u_{\min}^{h}=u_{\min}+\alpha_h(u_{\max}-u_{\min}),
$$

$$
u_{\max}^{h}=u_{\max}-\alpha_h(u_{\max}-u_{\min}),
$$

where $\alpha_h=0.03$ by default. This preserves a small input headroom for the later Lyapunov controller.

The default command movement penalty is:

$$
\lambda_r = 1.0.
$$

The output weight $W_r$ uses the same direct LMPC output weights by default:

$$
W_r = \operatorname{diag}(5.0,1.0).
$$

## Step 2: Steady Target Around The Governed Command

After $r_k$ is computed, the steady Lyapunov target is solved around $r_k$, not directly around the raw setpoint:

$$
\begin{aligned}
\min_{x_s,u_s}\quad
& \|Cx_s+\hat d_k-r_k\|_{Q_r}^2
+ \lambda_u\|u_s-u_{k-1}\|_2^2
+ \lambda_x\|x_s-x_{s,k-1}\|_2^2 \\
\text{s.t.}\quad
& x_s = Ax_s + Bu_s, \\
& u_{\min}^{h} \le u_s \le u_{\max}^{h}.
\end{aligned}
$$

The initial defaults are:

$$
Q_r=\operatorname{diag}(5.0,1.0),
$$

$$
\lambda_u=0.1,
\qquad
\lambda_x=0.1.
$$

This preserves the previous target anchoring idea, but it applies it after the command has been explicitly governed.

## Step 3: Direct Lyapunov MPC Around The New Target

The direct LMPC solver is unchanged. The target from Step 2 defines the Lyapunov center. The tracking MPC solves for a control sequence while enforcing the first-step Lyapunov contraction condition:

$$
V_{k+1} \le \rho V_k + \epsilon.
$$

The Lyapunov function is:

$$
V_k=(\hat x_k-x_{s,k})^T P_x(\hat x_k-x_{s,k}).
$$

The current experiment keeps:

$$
\rho=0.99,
\qquad
\epsilon=10^{-2}.
$$

The stage tracking objective still uses the raw setpoint by default:

$$
y_{\mathrm{track},k}=y_{sp,k}.
$$

This is deliberate. It lets the run test whether the governed target improves the certificate center without hiding raw setpoint error.

## Optional One-Step Probe

The governor also computes an optional one-step feasibility probe. For the target $(x_s,u_s)$, it solves:

$$
\min_{u_0\in U}\;
\left(A(\hat x_k-x_s)+B(u_0-u_s)\right)^T
P_x
\left(A(\hat x_k-x_s)+B(u_0-u_s)\right).
$$

The probe margin is:

$$
m_{\mathrm{probe},k}
=
V_{\min,k+1}
-
(\rho V_k+\epsilon).
$$

A negative margin means there exists at least one first-step input that can satisfy the Lyapunov decrease test around the governed target.

## New Diagnostics

The new run stores the following command-layer diagnostics at each step:

- `r_cmd`: governed admissible command.
- `r_cmd_minus_y_sp`: how much the raw setpoint was modified.
- `y_s_minus_r_cmd`: target consistency error after the second-stage target solve.
- `y_s_minus_y_sp`: total target mismatch relative to the raw setpoint.
- `governor_active`: whether $r_k$ differs from $y_{sp,k}$ beyond tolerance.
- `governor_probe_margin`: optional one-step Lyapunov feasibility probe margin.
- `input_headroom_min`: minimum distance from the selected target input to the original input bounds.
- `command_move_inf`: $\|r_k-r_{k-1}\|_\infty$.

The key new plot is:

```text
06_governed_reference_diagnostics.png
```

It overlays $y$, $y_{sp}$, $r_k$, and $y_s$, then shows command mismatch, target-command mismatch, and governor activity.

## Experiment Runner

The new root runner is:

```text
DirectLyapunovMPC_GovernedReference.py
```

The run saves under:

```text
results/LyapMPC_GovernedReference/<timestamp>/
```

The case name is:

```text
lyap_mpc_governed_reference
```

The existing direct LMPC runner and existing bounded/unbounded selector behavior remain unchanged.

## Expected Success Criteria

After the first full run, Proposal 01 should be judged by these criteria:

$$
\|y_s-y_{sp}\|_\infty \downarrow,
\qquad
\|r_k-y_{sp}\|_\infty \text{ explicit and explainable},
\qquad
\|y_s-r_k\|_\infty \approx 0.
$$

Closed-loop criteria:

- Raw output RMSE should improve or remain scientifically explainable.
- Final-tail offset should not worsen.
- Lyapunov contraction margins should not become systematically worse.
- Input headroom should not collapse to zero for long periods.
- Governor activation should concentrate around setpoint or disturbance transitions, not remain permanently active.

## What To Add After Results Exist

After running `DirectLyapunovMPC_GovernedReference.py`, extend this report with:

- output tracking plots for $y$, $y_{sp}$, $r_k$, and $y_s$,
- input and target-input plots,
- Lyapunov margin summaries,
- raw RMSE and final-tail offset,
- target mismatch statistics,
- command-governor activation rate,
- wall-clock seconds per step,
- and a comparison against the latest `directLyap` baseline.

The most important scientific question is whether the governed command makes target modification transparent and reduces the previously hidden gap between the Lyapunov certificate target and the raw process objective.
