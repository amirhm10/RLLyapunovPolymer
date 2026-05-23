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

The initial Proposal 01 configuration was designed around:

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

## First Two-Scenario Result

Two short governed-reference tests were run after the implementation. Each test used two episodes with `set_points_len = 400`, so each run contains $1600$ closed-loop control steps. Both valid runs solved the governed target layer and the direct LMPC problem at every step. This means the poor case is not a numerical failure. It is a control-objective mismatch.

The local runner used for these short tests had a stricter Lyapunov tolerance than the initial proposal:

$$
\rho=0.99,
\qquad
\epsilon=10^{-6}.
$$

The two scenarios were:

- `raw-ysp tracking`: `use_target_output_for_tracking = False`, so the MPC stage objective tracks the original requested setpoint $y_{sp,k}$.
- `target-output tracking`: `use_target_output_for_tracking = True`, so the MPC stage objective tracks the selected Lyapunov target output $y_{s,k}$.

An earlier wrong-interpreter run failed because `cvxpy` was unavailable and the target selector could not solve. That run should be discarded. The valid runs below used the `rl-env` interpreter and produced feasible target and solver diagnostics.

## Result Plots

Raw-setpoint tracking keeps the controller objective aligned with the process objective. The governed command and target are visible, but they are not allowed to redefine the reported tracking goal.

![Governed-reference raw setpoint tracking outputs](../results/LyapMPC_GovernedReference/20260523_010658/lyap_mpc_governed_reference/plots/01_outputs_vs_targets.png)

![Governed-reference raw setpoint diagnostics](../results/LyapMPC_GovernedReference/20260523_010658/lyap_mpc_governed_reference/plots/06_governed_reference_diagnostics.png)

Target-output tracking makes the plant track $y_s$ very tightly, but $y_s$ itself is far from the requested raw setpoint for long intervals.

![Governed-reference target-output tracking outputs](../results/LyapMPC_GovernedReference/20260523_010838/lyap_mpc_governed_reference/plots/01_outputs_vs_targets.png)

![Governed-reference target-output diagnostics](../results/LyapMPC_GovernedReference/20260523_010838/lyap_mpc_governed_reference/plots/06_governed_reference_diagnostics.png)

## Performance Summary

The raw-setpoint objective is much better for the reported process-control objective. The target-output objective receives a much worse reward and much larger raw-output RMSE, even though it tracks its own internal target very well.

| Case | Stage target | Reward mean | $\eta$ RMSE | $T$ RMSE | Mean RMSE |
|---|---|---:|---:|---:|---:|
| Raw-ysp tracking | $y_{sp}$ | -3.925 | 0.181 | 0.537 | 0.359 |
| Target-output tracking | $y_s$ | -53.200 | 0.557 | 3.540 | 2.048 |

The same conclusion appears when separating raw process error from internal tracking error.

| Case | Mean raw $|\eta-y_{sp,\eta}|$ | Mean raw $|T-y_{sp,T}|$ | Mean tracking $|\eta-y_{track,\eta}|$ | Mean tracking $|T-y_{track,T}|$ |
|---|---:|---:|---:|---:|
| Raw-ysp tracking | 0.071 | 0.284 | 0.071 | 0.284 |
| Target-output tracking | 0.556 | 3.045 | 0.0007 | 0.0415 |

This is the key evidence from the two-scenario run. If the MPC tracks $y_s$, the controller becomes excellent at following the modified Lyapunov target, but the modified target is not sufficiently aligned with $y_{sp}$. Therefore, target-output tracking can hide the original tracking problem instead of solving it.

## Feasibility And Lyapunov Diagnostics

Both scenarios were reliable from the optimization and Lyapunov-certification point of view.

| Case | Target success | Solver success | Hard contraction | Probe success | Seconds per step |
|---|---:|---:|---:|---:|---:|
| Raw-ysp tracking | 1.000 | 1.000 | 1.000 | 1.000 | 0.0341 |
| Target-output tracking | 1.000 | 1.000 | 1.000 | 1.000 | 0.0338 |

The Lyapunov layer is therefore not the immediate failure point. The governor found feasible commands, the target solve succeeded, and the direct LMPC found an optimal action at every step. The scientific question is whether the feasible command is the right command.

## Target-Layer Diagnosis

The target-output case is more conservative. The governed command is active at every step and remains far from the raw setpoint.

| Case | Governor active | Mean $\|r-y_{sp}\|_\infty$ | Max $\|r-y_{sp}\|_\infty$ | Mean $\|y_s-r\|_\infty$ | Mean $\|r_k-r_{k-1}\|_\infty$ |
|---|---:|---:|---:|---:|---:|
| Raw-ysp tracking | 0.903 | 0.485 | 4.999 | 0.065 | 0.031 |
| Target-output tracking | 1.000 | 4.244 | 6.893 | 0.225 | 0.0039 |

This table explains the poor raw tracking for the target-output run:

- The plant tracks $y_s$ well, so the direct LMPC can stabilize around the chosen target.
- The chosen target is far from $y_{sp}$, especially for temperature.
- The command movement is very small in the target-output case, which suggests the governed command can become too sticky or conservative.
- The target-command mismatch $\|y_s-r\|_\infty$ is also larger in the target-output case, so the second target solve does not perfectly realize the governed command.

## Segment-Level Behavior

The last 20 samples of each 400-step segment are useful because they show whether the controller eventually settles near the requested setpoint.

| Case | Segment | Tail mean raw $\eta$ error | Tail mean raw $T$ error |
|---|---:|---:|---:|
| Raw-ysp tracking | 1 | 0.014 | 0.074 |
| Raw-ysp tracking | 2 | 0.013 | 0.062 |
| Raw-ysp tracking | 3 | 0.0037 | 0.014 |
| Raw-ysp tracking | 4 | 0.0043 | 0.021 |
| Target-output tracking | 1 | 0.575 | 0.759 |
| Target-output tracking | 2 | 0.556 | 5.486 |
| Target-output tracking | 3 | 0.540 | 2.184 |
| Target-output tracking | 4 | 0.548 | 4.280 |

Raw-ysp tracking settles close to the requested setpoint by the tail of each segment. Target-output tracking does not, because it is settling close to $y_s$ rather than $y_{sp}$.

## Interpretation

Proposal 01 is useful, but the first result says it should currently be treated as a diagnostic architecture rather than a solved target-selector fix.

The positive result is that the governed-reference implementation is numerically stable:

- The governed command layer solved at every step.
- The target layer solved at every step.
- The direct LMPC solver solved at every step.
- The first-step Lyapunov contraction condition was satisfied at every step.
- The new diagnostics clearly separate $y_{sp}$, $r_k$, $y_s$, and $y$.

The negative result is that tracking $y_s$ does not solve the raw process objective. It makes the controller look excellent in the internal target coordinates, while raw setpoint tracking becomes much worse. This confirms the central issue from the RL and direct LMPC work: the target selector can produce a Lyapunov-admissible target that is not a good raw-setpoint target.

## Same-Setup DirectLyap Comparison

A matching two-episode `directLyap` run was also completed:

```text
results/directLyap/20260523_011436/
```

The main direct Lyapunov case is:

```text
lyap_mix_u0p1_x0p1_lex
```

The direct run also includes `mpc_only`. This is useful as a tracking reference, but it should not be interpreted as a safety-gated Lyapunov controller because it does not enforce fallback. Its Lyapunov contraction failures are diagnostic "would-be activation" events.

![DirectLyap bounded output tracking](../results/directLyap/20260523_011436/lyap_mix_u0p1_x0p1_lex/plots/01_outputs_vs_targets.png)

![DirectLyap bounded target diagnostics](../results/directLyap/20260523_011436/lyap_mix_u0p1_x0p1_lex/plots/05_target_diagnostics.png)

For the same two-episode setup, governed-reference with raw setpoint tracking is better than the bounded direct Lyapunov run on the raw process objective. It is also very close to `mpc_only` on RMSE and reward, but slower because it adds the governed command optimization before the direct LMPC solve.

| Case | Target mode | Reward mean | $\eta$ RMSE | $T$ RMSE | Mean RMSE |
|---|---|---:|---:|---:|---:|
| `mpc_only` | bounded diagnostic | -3.878 | 0.180 | 0.534 | 0.357 |
| Governed raw-ysp | governed reference | -3.925 | 0.181 | 0.537 | 0.359 |
| DirectLyap bounded | bounded | -5.163 | 0.203 | 0.655 | 0.429 |
| Governed target-ys | governed reference | -53.200 | 0.557 | 3.540 | 2.048 |

The reliability metrics show why the governed-reference raw-ysp result is promising. It matched or improved the direct bounded Lyapunov reliability metrics while keeping raw tracking close to `mpc_only`.

| Case | Solver success | Hard contraction | Target success | Sec/step |
|---|---:|---:|---:|---:|
| `mpc_only` | 1.000 | 0.641 | 1.000 | 0.0046 |
| Governed raw-ysp | 1.000 | 1.000 | 1.000 | 0.0341 |
| DirectLyap bounded | 0.999 | 0.999 | 1.000 | 0.0234 |
| Governed target-ys | 1.000 | 1.000 | 1.000 | 0.0338 |

The target diagnostics also improve under governed-reference raw-ysp tracking. The bounded direct target has a larger target residual maximum, while the governed-reference target is much more consistent with its command.

| Case | Mean $\|y_s-y_{sp}\|_\infty$ | Max target residual | Mean raw output error | Tail behavior |
|---|---:|---:|---:|---|
| Governed raw-ysp | 0.539 | 0.415 | 0.453 | settles close |
| DirectLyap bounded | 0.813 | 8.161 | 0.615 | larger tail errors |
| `mpc_only` | 0.494 | 6.100 | 0.450 | settles close |
| Governed target-ys | 4.469 | 0.303 | 4.463 | tracks wrong target |

This comparison changes the interpretation of Proposal 01. The governed reference layer does help when the stage objective remains the raw setpoint. It does not help when the stage objective is changed to the governed target output.

## Why $y_s$ Is Close To $y_{sp}$ In One Run But Not The Other

This is not a contradiction. The selected target $y_s$ is not a fixed precomputed signal. It is recalculated at every control step from the current observer state, disturbance estimate, previous command, previous target, previous input, and current raw setpoint:

$$
y_{s,k}
=
g\left(
\hat x_k,\hat d_k,u_{k-1},x_{s,k-1},r_{k-1},y_{sp,k}
\right).
$$

Changing the tracking objective changes the closed-loop trajectory. That changes $\hat x_k$, $\hat d_k$, $u_{k-1}$, and the next target optimization. Therefore, the two runs do not generate the same sequence of $r_k$ and $y_s$.

In the raw-ysp run, the MPC stage cost still pushes the plant toward the requested process setpoint:

$$
y_{\mathrm{track},k}=y_{sp,k}.
$$

The governed command and $y_s$ are used to define a Lyapunov-admissible certificate center, but they do not replace the process objective. Since the plant is being pushed toward $y_{sp}$, the observer and target selector remain in a region where an admissible $y_s$ can stay relatively close to $y_{sp}$.

In the target-output run, the MPC stage cost changes to:

$$
y_{\mathrm{track},k}=y_{s,k}.
$$

Once $y_s$ is offset from the raw setpoint, the controller has no stage-cost reason to close the remaining gap to $y_{sp}$. It becomes good at tracking its internal target. This makes the internal tracking error small, but it also reinforces a closed-loop trajectory where the next governed command remains conservative and far from the requested setpoint.

The practical reading is:

- Close $y_s$ in the raw-ysp plot is an outcome of the closed-loop objective staying aligned with $y_{sp}$.
- Far $y_s$ in the target-output plot shows that $y_s$ should not be allowed to redefine the process objective unless target quality is already trusted.
- The governed target is a certificate and diagnostic object. It is not yet a reliable replacement for the operator setpoint.

## Adopted Default For Next Runs

The governed-reference selector is now promoted from a test runner into the default target selector for the active direct MPC, cold-start RL, and pretrained RL runners. The adopted default keeps:

$$
y_{\mathrm{track},k}=y_{sp,k}
$$

as the main reported objective. The target output $y_s$ remains a Lyapunov certificate center and diagnostic signal. It does not replace the operator-requested raw setpoint in the MPC stage cost.

The active target configuration is:

```python
target_mode = "governed_reference"
lambda_cmd_move = 1.0
Qr_diag = Qy_diag
W_r_diag = Qy_diag
u_ref_weight = 0.1
x_ref_weight = 0.1
input_headroom_frac = 0.03
one_step_probe = True
```

This is the raw-setpoint governed-reference version, not the target-output tracking version.

## Default Configuration Rationale

`target_mode = "governed_reference"` is used because the two-episode comparison showed better raw-setpoint performance than the bounded direct Lyapunov target while maintaining full target-solve, solver, and hard-contraction success. It also makes the target modification visible through $r_k$, $y_s-r_k$, and $y_s-y_{sp}$ diagnostics.

`lambda_cmd_move = 1.0` penalizes large jumps in the governed command:

$$
\lambda_r\|r_k-r_{k-1}\|^2.
$$

The purpose is not to freeze the command. The purpose is to prevent aggressive target jumps that can create unnecessary input movement or abrupt Lyapunov-center changes, while still allowing the command to move toward the feasible setpoint.

`Qr_diag = Qy_diag` gives the steady target solve the same output priorities as the controller. The target solve therefore uses the same relative weighting between viscosity-like $\eta$ and temperature $T$ that the direct LMPC uses in its stage objective.

`W_r_diag = Qy_diag` gives the governed command projection the same output priorities:

$$
\|r_k-y_{sp,k}\|_{W_r}^2.
$$

This prevents the command layer from treating the two outputs with a different importance than the controller itself.

`u_ref_weight = 0.1` anchors the steady input target to the previous applied input:

$$
\lambda_u\|u_s-u_{k-1}\|^2.
$$

This reduces unnecessary target-input chatter. The value is intentionally moderate, so it smooths $u_s$ without dominating the output target objective.

`x_ref_weight = 0.1` anchors the steady state target to the previous successful target:

$$
\lambda_x\|x_s-x_{s,k-1}\|^2.
$$

This smooths the Lyapunov center and avoids discontinuous target jumps. It is especially useful because $V_k$ is computed around $x_s$, so sudden target movement can change the certificate geometry even if the raw setpoint has not changed much.

`input_headroom_frac = 0.03` shrinks the steady target input box slightly before solving the target problem. If the physical input bounds are $u_{\min}$ and $u_{\max}$, the governed target solve uses:

$$
u_{\min}^{h}=u_{\min}+0.03(u_{\max}-u_{\min}),
\qquad
u_{\max}^{h}=u_{\max}-0.03(u_{\max}-u_{\min}).
$$

This leaves room for the later MPC correction step. Without headroom, the target can sit directly on an input bound, leaving the controller little authority to enforce contraction and tracking.

`one_step_probe = True` keeps the feasibility probe active. The probe checks whether there exists a first-step input that can satisfy:

$$
V_{k+1}\le \rho V_k+\epsilon
$$

around the governed target. This probe is diagnostic, not a replacement for the actual direct LMPC solve. It helps explain whether a target is Lyapunov-compatible before the full controller action is interpreted.

## Remaining Diagnostic Use Of $y_s$ Tracking

Use target-output tracking only as a diagnostic. It answers the question: "Can the controller track the target it was given?" The two-episode result says yes. The remaining question is: "Is the target selector giving the controller the right target?" For the adopted default, this is tested by keeping raw $y_{sp}$ tracking active while logging $r_k$, $y_s$, and target mismatch diagnostics.

Future ablations should focus on the command layer only after the default governed-reference runs are complete:

- Compare governed-reference targets against the existing bounded selector on the same two-episode test.
- Report both raw-setpoint RMSE and internal-target RMSE, so target quality and controller tracking are never mixed together.

The most important scientific conclusion is that Proposal 01 made the hidden target mismatch visible and improved the direct Lyapunov raw-tracking comparison when the controller continued tracking $y_{sp}$. The next full runs will test whether this advantage holds for longer direct MPC, cold-start RL, and pretrained RL experiments.
