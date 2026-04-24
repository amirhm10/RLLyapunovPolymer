# Four-Scenario Direct Lyapunov MPC With Frozen Output Disturbance

Prepared from the latest available four-scenario export:

`Data/debug_exports/direct_lyapunov_mpc_four_scenario/20260423_195957`

The notebook behind this report is `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`. It runs the same plant, observer, setpoint schedule, horizons, weights, and failure policy across four cases:

| Case | Target selector | Lyapunov constraint |
| --- | --- | --- |
| `unbounded_hard` | exact unbounded steady target | strict first-step contraction |
| `bounded_hard` | input-bounded steady target projection | strict first-step contraction |
| `unbounded_soft` | exact unbounded steady target | contraction relaxed by slack |
| `bounded_soft` | input-bounded steady target projection | contraction relaxed by slack |

The central conclusion from this run is that target admissibility is the main issue. The unbounded target solve tracks the requested setpoint almost exactly, but its required steady input is usually far outside the admissible input box. The hard MPC then becomes infeasible, while the soft MPC can solve by using Lyapunov slack and aggressive moves. The bounded target projection makes the steady target admissible and gives the most useful controller behavior in this run. Among the four cases, `bounded_hard` gives the best mean reward, while `bounded_soft` gives the most robust solver success rate and almost always satisfies the original hard contraction without needing much slack.

## 1. Method

### 1.1 Coordinates And Offset-Free Output-Disturbance Model

The controller works in scaled deviation coordinates. The plant model used by the direct Lyapunov MPC is a linear prediction model with a constant output-disturbance augmentation:

```math
x_{k+1}=Ax_k+Bu_k,
\qquad
y_k=Cx_k+d_k,
\qquad
d_{k+1}=d_k .
```

The observer state is

```math
\hat{z}_k =
\begin{bmatrix}
\hat{x}_k\\
\hat{d}_k
\end{bmatrix},
```

where `xhat` is the estimated plant state in controller coordinates and `dhat` is the estimated output disturbance. This is the offset-free MPC idea: persistent plant-model mismatch is represented by a disturbance state, and the target calculation uses the disturbance estimate so that the predicted steady output can match the requested setpoint when an admissible target exists. This is consistent with offset-free MPC disturbance-model literature such as Muske and Badgwell (2002) and Pannocchia and Rawlings (2003).

The direct notebook intentionally uses the frozen output-disturbance form. At each control step:

```math
d_s = \hat{d}_k .
```

The target selector solves only for `x_s` and `u_s`. The disturbance target is not optimized separately.

### 1.2 Steady Target Equations

For a requested output setpoint `y_sp,k`, the steady target must satisfy

```math
x_s = A x_s + B u_s,
\qquad
y_{\mathrm{sp},k}=C x_s+d_s .
```

Equivalently,

```math
(I-A)x_s-Bu_s=0,
\qquad
Cx_s=y_{\mathrm{sp},k}-\hat{d}_k .
```

Stacking these equations gives the linear steady-target system

```math
\begin{bmatrix}
I-A & -B\\
C & 0
\end{bmatrix}
\begin{bmatrix}
x_s\\
u_s
\end{bmatrix}
=
\begin{bmatrix}
0\\
y_{\mathrm{sp},k}-\hat{d}_k
\end{bmatrix}.
```

Let

```math
M =
\begin{bmatrix}
I-A & -B\\
C & 0
\end{bmatrix},
\qquad
\xi_s =
\begin{bmatrix}
x_s\\
u_s
\end{bmatrix},
\qquad
b_k =
\begin{bmatrix}
0\\
y_{\mathrm{sp},k}-\hat{d}_k
\end{bmatrix}.
```

Then the target residual is

```math
r_s = M\xi_s-b_k .
```

The target residual norm reported in the tables below is based on this stacked residual. A small residual means the target solves the linear steady-state equations well. It does not by itself mean the target is admissible for the constrained MPC.

### 1.3 Unbounded Target Mode

In `unbounded` target mode, the target selector solves the steady equations without enforcing the input box:

```math
\min_{x_s,u_s}\ \|M\xi_s-b_k\|_2^2 .
```

When the exact equations are solvable, this gives

```math
y_s = Cx_s+\hat{d}_k \approx y_{\mathrm{sp},k}.
```

If `I-A` is invertible, the same calculation can be viewed through the reduced steady gain

```math
G = C(I-A)^{-1}B,
\qquad
G u_s = y_{\mathrm{sp},k}-\hat{d}_k,
\qquad
x_s=(I-A)^{-1}Bu_s .
```

This is mathematically clean, but it can ask for a steady input that violates

```math
u_{\min}\le u_s\le u_{\max}.
```

That is exactly what happens in this run: the unbounded target has nearly zero output residual, but the implied `u_s` is far outside the admissible input region for almost all steps.

### 1.4 Bounded Target Mode

In `bounded` target mode, the exact target is first computed and checked against the input box. If it is inside the bounds, the exact target is used. If not, the selector solves an input-bounded least-squares projection.

In reduced form, the projection is

```math
\min_{u_{\min}\le u_s\le u_{\max}}
\left\|G u_s-(y_{\mathrm{sp},k}-\hat{d}_k)\right\|_2^2,
\qquad
x_s=(I-A)^{-1}Bu_s .
```

This changes the meaning of the target output. In unbounded exact mode, the
target equations enforce `y_s = y_sp` when they are solvable. In bounded mode,
that equality is guaranteed only if the exact target input is also admissible:

```math
\begin{aligned}
y_s=y_{\mathrm{sp},k}
\quad\Longleftrightarrow\quad
&\exists (x_s,u_s)\ \text{such that}\\
&(I-A)x_s-Bu_s=0,\\
&Cx_s+\hat{d}_k=y_{\mathrm{sp},k},\\
&u_{\min}\le u_s\le u_{\max}.
\end{aligned}
```

If no such admissible steady target exists, the bounded solve keeps
`u_s` inside the input box and accepts a nonzero output-target residual:

```math
y_s=Cx_s+\hat{d}_k
=y_{\mathrm{sp},k}+r_{\mathrm{out},k},
\qquad
r_{\mathrm{out},k}=Cx_s-(y_{\mathrm{sp},k}-\hat{d}_k).
```

Therefore, after bounded projection, generally

```math
y_s\ne y_{\mathrm{sp},k}.
```

If the reduced form is unavailable, the implementation falls back to the full constrained least-squares problem:

```math
\min_{x_s,u_s}
\left\|
\begin{bmatrix}
I-A & -B\\
C & 0
\end{bmatrix}
\begin{bmatrix}
x_s\\
u_s
\end{bmatrix}
-
\begin{bmatrix}
0\\
y_{\mathrm{sp},k}-\hat{d}_k
\end{bmatrix}
\right\|_2^2
\quad
\text{s.t.}\quad
u_{\min}\le u_s\le u_{\max}.
```

The bounded target may no longer achieve `y_s = y_sp`. Instead, it returns the closest steady target that respects the input limits. This is why the bounded cases have nonzero target residual and nonzero `||y_s-y_sp||`, but much better MPC feasibility. This is closely related to admissible artificial-reference tracking MPC, where infeasible references are replaced by the nearest admissible steady target; see Limon et al. (2008).

### 1.5 Direct Tracking MPC Objective

After target selection, the controller solves one MPC problem directly. The prediction model freezes the output disturbance:

```math
x_{0|k}=\hat{x}_k,
\qquad
x_{i+1|k}=Ax_{i|k}+Bu_{i|k},
\qquad
y_{i|k}=Cx_{i|k}+\hat{d}_k .
```

The current notebook objective is the normal output-tracking MPC objective:

```math
J_k =
\sum_{i=0}^{N_p-1}
\left(y_{i|k}-y^{\mathrm{track}}_k\right)^T
Q_y
\left(y_{i|k}-y^{\mathrm{track}}_k\right)
+
\sum_{i=0}^{N_c-1}
\Delta u_{i|k}^T
R_{\Delta u}
\Delta u_{i|k}.
```

The input move is

```math
\Delta u_{0|k}=u_{0|k}-u_{k-1},
\qquad
\Delta u_{i|k}=u_{i|k}-u_{i-1|k}.
```

The default study setting is

```math
y^{\mathrm{track}}_k = y_{\mathrm{sp},k}.
```

This is important: the solver variable named `y_target` is the scheduled setpoint in this run, not the steady target output `y_s`. The objective no longer contains the extra terms

```math
\|u_{i|k}-u_s\|^2,
\qquad
\|x_{N|k}-x_s\|^2 .
```

The steady variables `x_s`, `u_s`, `y_s`, and `d_s` are used for Lyapunov contraction and terminal admissibility. They are not objective anchors in this four-scenario run.

### 1.6 Terminal Set And Terminal Cost

The terminal ingredients are centered on the selected steady state. A generic terminal set has the form

```math
\mathcal{X}_f(x_s)
=
\left\{
x:\ (x-x_s)^T P_f (x-x_s)\le \alpha_f
\right\}.
```

The terminal constraint, when active, requires

```math
x_{N_p|k}\in \mathcal{X}_f(x_s).
```

For this direct study, the terminal cost scale, denoted here by
`\lambda_f`, is set to zero:

```math
\lambda_f=0.
```

Therefore, the terminal ingredients act as admissibility and stability diagnostics, not as an added terminal penalty in the cost. This keeps the MPC objective focused on `y-y_sp` and `Delta u`.

### 1.7 Hard Lyapunov Contraction

Define the Lyapunov function around the selected target:

```math
V_k =
(\hat{x}_k-x_s)^TP_x(\hat{x}_k-x_s).
```

Hard mode enforces first-step contraction:

```math
V_{1|k}
\le
\rho V_k+\epsilon_{\mathrm{lyap}},
```

where

```math
V_{1|k}
=
(x_{1|k}-x_s)^TP_x(x_{1|k}-x_s),
\qquad
0<\rho<1.
```

The contraction margin recorded in the diagnostics is

```math
m_k = V_{1|k} - \rho V_k - \epsilon_{\mathrm{lyap}} .
```

Hard contraction is satisfied when `m_k <= 0`. Hard mode is the stricter scientific test: if the target is inadmissible or too aggressive, the MPC can become infeasible. That infeasibility is a diagnostic result, not a notebook failure.

### 1.8 Soft Lyapunov Contraction

Soft mode adds one nonnegative slack variable:

```math
V_{1|k}
\le
\rho V_k+\epsilon_{\mathrm{lyap}}+\sigma_k,
\qquad
\sigma_k\ge 0.
```

The soft objective is

```math
J_k^{\mathrm{soft}} = J_k+\lambda_\sigma\sigma_k .
```

The slack variable tells us how much contraction had to be relaxed to keep the controller feasible. A good soft result should have high solver success and low slack. A poor soft result can solve often but only by using large slack, meaning it is not behaving like the intended Lyapunov controller.

### 1.9 Online Algorithm

At each time step:

1. Estimate `xhat` and `dhat` from the augmented observer.
2. Freeze the output disturbance: `d_s = dhat`.
3. Solve the steady target using either unbounded exact target mode or bounded least-squares projection.
4. Solve the MPC with output tracking, input move penalty, input constraints, and either hard or soft Lyapunov contraction.
5. Apply the first input move if the MPC solve succeeds.
6. If the MPC solve fails, hold the previous input and log `solver_fail_hold_prev`.
7. Save the target residuals, bound activity, solver status, Lyapunov margins, slack, outputs, inputs, and target comparison plots.

This direct controller is simpler than the earlier safety-filter/target-selector ablation notebook. The ablation path used additional target-selector terms to keep `u_s`, `x_s`, and previous targets smooth and close to the current operating point. Here those extra selector objectives are deliberately absent so the four cases isolate the effect of target admissibility and Lyapunov hard/soft relaxation.

## 2. Latest Run Results

### 2.1 Run Source And Artifacts

The latest four-scenario run found in the repository is:

`Data/debug_exports/direct_lyapunov_mpc_four_scenario/20260423_195957`

The comparison summary was created at `2026-04-23T20:06:47`. Each case has 1600 logged control steps. Runtime artifacts remain under `Data/debug_exports/`, while the report-ready figures are copied into:

`report/figures/direct_lyapunov_mpc_frozen_output_disturbance/`

The comparison table is:

| Case | Mean reward | Output 0 RMSE | Output 1 RMSE | Solver success | Hard contraction | Relaxed contraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `unbounded_hard` | -40.135 | 0.557 | 2.291 | 0.00% | 0.00% | 0.00% |
| `bounded_hard` | -23.758 | 0.422 | 1.660 | 95.75% | 95.75% | 95.75% |
| `unbounded_soft` | -91.229 | 0.218 | 0.935 | 97.25% | 30.50% | 97.25% |
| `bounded_soft` | -42.432 | 0.535 | 2.211 | 99.88% | 97.75% | 99.88% |

The output RMSE values are post-step physical output RMSE values computed from `y_system[1:]` against the physical setpoint schedule.

The Lyapunov slack and target-selector table is:

| Case | Slack mean | Slack max | Slack active | Max target residual | Exact target in bounds | Bounded target used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `unbounded_hard` | 0.000 | 0.000 | 0 / 1600 | `4.58e-15` | 0 / 1600 | 0 / 1600 |
| `bounded_hard` | 0.000 | 0.000 | 0 / 1600 | 15.946 | 8 / 1600 | 1592 / 1600 |
| `unbounded_soft` | 4.236 | 70.784 | 1068 / 1600 | `6.48e-15` | 1 / 1600 | 0 / 1600 |
| `bounded_soft` | 0.00870 | 1.708 | 34 / 1600 | 17.244 | 0 / 1600 | 1600 / 1600 |

The target-comparison diagnostics in controller coordinates are:

| Case | Mean `||u-u_s||_inf` | Mean `||y_s-y_sp||_inf` | Mean `||d_s||_inf` | Bounded lower active | Bounded upper active |
| --- | ---: | ---: | ---: | ---: | ---: |
| `unbounded_hard` | 585.151 | `8.70e-16` | 0.703 | 0 / 1600 | 0 / 1600 |
| `bounded_hard` | 10.158 | 2.629 | 3.851 | 1117 / 1600 | 1405 / 1600 |
| `unbounded_soft` | 476.225 | `5.86e-16` | 6.974 | 0 / 1600 | 0 / 1600 |
| `bounded_soft` | 13.591 | 3.335 | 4.570 | 1242 / 1600 | 1380 / 1600 |

The unbounded cases have almost zero `y_s-y_sp` because the target solve is exact, but this is misleading by itself. The `u-u_s` gap is enormous because `u_s` is outside the admissible input region. The bounded cases accept a nonzero `y_s-y_sp` gap so that the steady input is admissible.

### 2.2 Comparison Plots

![Mean reward by case.](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_reward_mean.png)

`bounded_hard` gives the least negative mean reward in this run. `unbounded_soft` has the best output RMSE but the worst reward, which is consistent with solving by using aggressive target/input behavior and frequent Lyapunov relaxation rather than producing a balanced constrained-control response.

![Output RMSE by case.](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_output_rmse.png)

The output RMSE plot highlights the tradeoff: unbounded-soft tracks the scheduled output best, but it does so while its target remains inadmissible and while the Lyapunov slack is active on most steps.

![Solver success and contraction rates.](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_solver_contraction_rates.png)

`unbounded_hard` is infeasible for every MPC step. `bounded_hard` recovers strict Lyapunov behavior on all successful solves. `bounded_soft` almost never needs meaningful relaxation: its hard contraction rate is 97.75% and its relaxed contraction rate is 99.88%.

The next diagnostic looks directly at the Lyapunov function decrease. Two
differences are useful:

```math
\Delta V_{\mathrm{pred},k}=V_{1|k}-V_k,
\qquad
\Delta V_{\mathrm{logged},k}=V_k-V_{k-1}.
```

The first-step predicted delta is the cleaner contraction diagnostic because it
uses the current target `x_s(k)` and the first predicted state from the current
MPC solve. The logged step-to-step delta is still useful, but it can jump when
the target selector moves `x_s` between time steps.

| Case | First-step finite | First-step `Delta V <= 0` | First-step mean | Logged finite | Logged `Delta V <= 0` | Logged mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `unbounded_hard` | 0 | n/a | n/a | 0 | n/a | n/a |
| `bounded_hard` | 1532 | 1532 / 1532 (100.00%) | -10.053 | 1510 | 1118 / 1510 (74.04%) | -0.112 |
| `unbounded_soft` | 1556 | 611 / 1556 (39.27%) | 1.491 | 1514 | 813 / 1514 (53.70%) | -0.715 |
| `bounded_soft` | 1598 | 1598 / 1598 (100.00%) | -13.767 | 1595 | 1121 / 1595 (70.28%) | -0.062 |

![Lyapunov delta comparison across cases.](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_lyapunov_delta.svg)

This plot reinforces the main conclusion. The bounded cases decrease the
Lyapunov function on every accepted first-step prediction. The unbounded-soft
case solves often, but its first-step `Delta V` is positive more often than it
is negative, which is why the slack-based result should be interpreted as a
diagnostic rather than as the preferred stabilizing controller.

![Lyapunov slack by case.](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_slack.png)

The slack plot separates the two soft cases clearly. `unbounded_soft` uses large slack often, with a maximum of 70.784. `bounded_soft` uses small slack rarely, with a maximum of 1.708 and only 34 active slack steps.

![Target residual and bounded-target activity.](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_target_residual_bounded_activity.png)

The target residual plot shows why bounded projection should be interpreted carefully. A nonzero residual in bounded mode is expected: it is the price of respecting input admissibility. The relevant question is whether the resulting target makes the MPC feasible and stabilizing.

![Output overlay across all four cases.](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_outputs_overlay.png)

![Input overlay across all four cases.](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_inputs_overlay.png)

The overlay plots show the practical effect of each target/Lyapunov combination. The unbounded-hard case holds the previous input because every solve fails. The bounded cases produce feasible constrained moves. The unbounded-soft case can move, but the target it is trying to contract toward remains physically incompatible with the input bounds.

### 2.3 Case Study: `unbounded_hard`

Summary:

| Metric | Value |
| --- | ---: |
| Solver success | 0 / 1600 |
| Solver statuses | `infeasible`: 1600 |
| Target success | 1600 / 1600 |
| Exact target in bounds | 0 / 1600 |
| Target residual max | `4.58e-15` |
| Mean `||u-u_s||_inf` | 585.151 |
| Mean `||y_s-y_sp||_inf` | `8.70e-16` |

The unbounded target selector solves the steady equations exactly, so `y_s` is essentially equal to `y_sp`. However, the exact target is outside the input bounds at every step. The hard Lyapunov MPC then has no accepted feasible trajectory and falls back to holding the previous input at every step.

![unbounded_hard outputs and target outputs.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_hard_01_outputs_vs_targets.png)

![unbounded_hard inputs and steady input targets.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_hard_02_inputs_vs_targets.png)

The input-target plot is the key figure for this case: the requested `u_s` is not physically admissible. This explains why a perfect target residual does not translate into an executable controller.

![unbounded_hard state-target error.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_hard_03_state_target_error.png)

![unbounded_hard Lyapunov diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_hard_04_lyapunov_diagnostics.png)

Because the MPC solve is infeasible at all steps, the Lyapunov contraction trace has no accepted successful contraction behavior to evaluate.

![unbounded_hard Lyapunov delta diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_hard_06_lyapunov_delta.svg)

The delta diagnostic has no finite accepted values for this case. This is
consistent with the all-step hard-MPC infeasibility result.

![unbounded_hard target diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_hard_05_target_diagnostics.png)

The target diagnostics show the exact target residual near numerical zero while simultaneously showing persistent bound inadmissibility. This is the cleanest evidence that unbounded target selection is the wrong target selector for hard direct Lyapunov MPC in this experiment.

### 2.4 Case Study: `bounded_hard`

Summary:

| Metric | Value |
| --- | ---: |
| Solver success | 1532 / 1600 |
| Solver statuses | `optimal`: 1532, `optimal_inaccurate`: 18, `infeasible`: 50 |
| Target success | 1600 / 1600 |
| Hard contraction | 1532 / 1600 |
| Exact target in bounds | 8 / 1600 |
| Bounded target used | 1592 / 1600 |
| Slack active | 0 / 1600 |
| Mean reward | -23.758 |

This is the best reward case. The bounded target projection is active on almost every step, which means the exact target is normally outside the admissible input region. Unlike unbounded-hard, the projected target gives the hard Lyapunov MPC a feasible steady center for most of the run.

![bounded_hard outputs and target outputs.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_hard_01_outputs_vs_targets.png)

![bounded_hard inputs and steady input targets.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_hard_02_inputs_vs_targets.png)

The input plot shows `u_s` being clipped/projected into admissible regions, often with active lower or upper bounds. The controller can then solve without Lyapunov slack. The price is that `y_s` is not always equal to `y_sp`, because the admissible steady input cannot exactly realize the requested output target.

![bounded_hard state-target error.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_hard_03_state_target_error.png)

The state-target error is much smaller on average than in the unbounded cases, although it still has large spikes. This tells us that target projection helps, but the selected target can still move far from the current estimated state.

![bounded_hard Lyapunov diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_hard_04_lyapunov_diagnostics.png)

The Lyapunov diagnostics are strong when the solver succeeds: the hard contraction rate equals the solver success rate. The maximum contraction margin is essentially zero (`8.31e-11`), so accepted trajectories respect the contraction inequality up to numerical tolerance.

![bounded_hard Lyapunov delta diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_hard_06_lyapunov_delta.svg)

The first-step predicted Lyapunov delta is negative for all 1532 finite
accepted solves. The logged `V_k-V_{k-1}` value is negative on 74.04% of finite
step-to-step comparisons; the positive spikes are expected because the target
center `x_s(k)` can move between setpoint segments.

![bounded_hard target diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_hard_05_target_diagnostics.png)

The target diagnostics show that bounded projection is not a rare backup; it is the normal target-selection path in this experiment. This supports keeping bounded target selection as the default for the direct controller.

### 2.5 Case Study: `unbounded_soft`

Summary:

| Metric | Value |
| --- | ---: |
| Solver success | 1556 / 1600 |
| Solver statuses | `optimal`: 892, `optimal_inaccurate`: 708 |
| Target success | 1600 / 1600 |
| Hard contraction | 488 / 1600 |
| Relaxed contraction | 1556 / 1600 |
| Exact target in bounds | 1 / 1600 |
| Slack active | 1068 / 1600 |
| Slack max | 70.784 |
| Mean reward | -91.229 |

Softening the Lyapunov constraint rescues feasibility for the unbounded target, but the result is not a clean Lyapunov controller. The unbounded target is still inadmissible almost everywhere. The optimizer can solve by using slack, but hard contraction holds on only 30.5% of steps and slack is active on 66.75% of steps.

![unbounded_soft outputs and target outputs.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_soft_01_outputs_vs_targets.png)

![unbounded_soft inputs and steady input targets.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_soft_02_inputs_vs_targets.png)

The output RMSE is the best among the four cases, but the target/input diagnostics explain why this is not the preferred result. The exact `u_s` remains far from the applied input. The controller is tracking outputs while relaxing the Lyapunov requirement, not solving the original hard stabilizing problem.

![unbounded_soft state-target error.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_soft_03_state_target_error.png)

The state-target error remains large on average. This is consistent with an unbounded target that is mathematically exact but not close to the feasible operating region.

![unbounded_soft Lyapunov diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_soft_04_lyapunov_diagnostics.png)

The Lyapunov plot is the main caution. The relaxed inequality is often feasible, but the hard contraction inequality is frequently violated. The slack magnitude is not just numerical noise.

![unbounded_soft Lyapunov delta diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_soft_06_lyapunov_delta.svg)

Only 611 of 1556 finite first-step predicted deltas are nonpositive, and the
mean first-step delta is positive. This makes the soft unbounded case the clearest
example that solver feasibility alone is not enough: the controller often solves
by relaxing the Lyapunov decrease.

![unbounded_soft target diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/unbounded_soft_05_target_diagnostics.png)

The target diagnostics show near-zero residual and almost no exact bounded admissibility. This confirms that the soft formulation masks target inadmissibility rather than resolving it.

### 2.6 Case Study: `bounded_soft`

Summary:

| Metric | Value |
| --- | ---: |
| Solver success | 1598 / 1600 |
| Solver statuses | `optimal`: 1565, `optimal_inaccurate`: 35 |
| Target success | 1600 / 1600 |
| Hard contraction | 1564 / 1600 |
| Relaxed contraction | 1598 / 1600 |
| Bounded target used | 1600 / 1600 |
| Slack active | 34 / 1600 |
| Slack max | 1.708 |
| Mean reward | -42.432 |

This is the most robust case by solver success. The bounded target is used at every step, and the soft slack is active only rarely. The result suggests that bounded projection plus soft contraction is a good diagnostic and fallback mode: it preserves nearly all of the hard Lyapunov behavior while avoiding the occasional hard infeasibility seen in `bounded_hard`.

![bounded_soft outputs and target outputs.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_soft_01_outputs_vs_targets.png)

![bounded_soft inputs and steady input targets.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_soft_02_inputs_vs_targets.png)

The bounded target output does not exactly equal the requested setpoint, but the selected input target remains admissible. This is the intended compromise.

![bounded_soft state-target error.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_soft_03_state_target_error.png)

The state-target error remains larger than `bounded_hard` on average in this run, which helps explain why the reward is not as good as the hard bounded case. The soft slack keeps the controller feasible through difficult steps, but it does not automatically improve performance.

![bounded_soft Lyapunov diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_soft_04_lyapunov_diagnostics.png)

The Lyapunov diagnostics are encouraging. Hard contraction already holds on 97.75% of all logged steps, and relaxed contraction holds on 99.88%. Slack is small and sparse compared with `unbounded_soft`.

![bounded_soft Lyapunov delta diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_soft_06_lyapunov_delta.svg)

The bounded-soft first-step delta is negative for all 1598 finite accepted
solves, with a more negative mean first-step delta than bounded-hard in this
run. The logged step-to-step delta is negative on 70.28% of finite comparisons,
again reflecting both controller action and target-center movement.

![bounded_soft target diagnostics.](figures/direct_lyapunov_mpc_frozen_output_disturbance/bounded_soft_05_target_diagnostics.png)

The target diagnostics confirm that bounded target projection is continuously active and that this is beneficial rather than pathological. It is the mechanism that turns an inadmissible exact setpoint target into a usable steady center for the Lyapunov MPC.

## 3. Cross-Case Interpretation

### 3.1 What The Four Cases Tell Us

`unbounded_hard` is a negative control. It proves that exact output target matching is not sufficient. A target can have near-zero residual and still be unusable because the implied steady input violates constraints.

`bounded_hard` is the strongest strict Lyapunov result. It gives the best reward and satisfies hard contraction whenever the MPC solve is accepted. Its weakness is the 68 held-input steps caused by infeasible or not-accepted solves.

`unbounded_soft` is useful as a warning. It has the best output RMSE, but it achieves this while using large Lyapunov slack and while contracting toward an inadmissible target. This case should not be presented as the best controller just because it tracks output aggressively.

`bounded_soft` is the safest operational variant in this run. It almost always solves, usually satisfies the original hard contraction anyway, and uses only small slack. Its reward is worse than `bounded_hard`, so it is better interpreted as a robust fallback or as a candidate for tuning rather than the final best-performance controller.

### 3.2 Target Selector Conclusions

The target selector must be judged using both target residual and admissibility:

```math
\|r_s\| \approx 0
\quad\not\Rightarrow\quad
u_{\min}\le u_s\le u_{\max}.
```

The unbounded target selector makes `y_s` match `y_sp`, but it frequently selects an unusable `u_s`. The bounded target selector accepts a nonzero output residual so the target becomes compatible with input constraints. This is the right tradeoff for a constrained direct Lyapunov MPC.

The extra focus plots for `x_s`, `u_s`, `y_s`, and `d_s` support this:

- `u_s` is far from actual applied input in unbounded cases.
- `y_s` and `y_sp` are nearly identical in unbounded cases, but this is not enough.
- Bounded cases keep `u_s` admissible and make the MPC feasible.
- `x_s` can still move far from `xhat`, especially during setpoint changes, which motivates adding a small target-smoothing or current-state proximity term later.
- `d_s` is frozen at the observer disturbance estimate and is not the cause of hard infeasibility; the main issue is target input admissibility.

### 3.3 Lyapunov Conclusions

The best Lyapunov story is not the case with the lowest output RMSE. It is the case where solver feasibility, contraction, and slack are all consistent.

For this run:

- `bounded_hard` is the best strict-contraction controller when it solves.
- `bounded_soft` is the best robustness controller because it solves 1598 of 1600 steps and only needs slack 34 times.
- `unbounded_soft` should be treated as a diagnostic showing that soft slack can hide an inadmissible target.
- `unbounded_hard` should be kept as a reproducible failure baseline.

## 4. Literature Context

The method connects four ideas from the MPC literature.

Offset-free MPC uses disturbance augmentation and steady-state target calculation to remove steady-state output offset. The frozen output-disturbance model used here is a simple version of this idea, and the broader theory is discussed by Muske and Badgwell, "Disturbance modeling for offset-free linear model predictive control," Journal of Process Control, 2002, DOI: [10.1016/S0959-1524(01)00051-8](https://doi.org/10.1016/S0959-1524(01)00051-8), and Pannocchia and Rawlings, "Disturbance models for offset-free model-predictive control," AIChE Journal, 2003, DOI: [10.1002/aic.690490213](https://doi.org/10.1002/aic.690490213).

Constrained MPC stability and terminal ingredients are standard tools for proving recursive feasibility and stability; see Mayne, Rawlings, Rao, and Scokaert, "Constrained model predictive control: Stability and optimality," Automatica, 2000, DOI: [10.1016/S0005-1098(99)00214-9](https://doi.org/10.1016/S0005-1098(99)00214-9).

Admissible target selection is central when setpoints are not exactly reachable under constraints. The bounded target projection used here is aligned with tracking MPC ideas in Limon, Alvarado, Alamo, and Camacho, "MPC for tracking piecewise constant references for constrained linear systems," Automatica, 2008, DOI: [10.1016/j.automatica.2008.01.023](https://doi.org/10.1016/j.automatica.2008.01.023).

Lyapunov MPC uses a Lyapunov function or Lyapunov-based constraint to enforce stabilizing behavior under constraints. The hard/soft interpretation here is related to Lyapunov-based predictive control for constrained nonlinear systems; see Mhaskar, El-Farra, and Christofides, "Stabilization of nonlinear systems with state and control constraints using Lyapunov-based predictive control," Systems & Control Letters, 2006, DOI: [10.1016/j.sysconle.2005.09.014](https://doi.org/10.1016/j.sysconle.2005.09.014).

The older safety-filter notebook is also conceptually related to predictive safety filters, where an optimization layer modifies or rejects an unsafe input while maintaining constraints; see Wabersich and Zeilinger, "A predictive safety filter for learning-based control of constrained nonlinear dynamical systems," Automatica, 2021, DOI: [10.1016/j.automatica.2021.109597](https://doi.org/10.1016/j.automatica.2021.109597).

## 5. Recommended Next Steps

1. Make `bounded_hard` the primary strict Lyapunov baseline and `bounded_soft` the robustness baseline. The unbounded modes should remain diagnostic comparisons.
2. Add a small target-selector regularization study, but do it one term at a time. The first candidates are `u_s` proximity to the previous or current applied input and `x_s` proximity to the current `xhat`. This directly addresses the large `xhat-x_s` and `u-u_s` gaps without immediately returning to the full five-term ablation objective.
3. Tune the bounded-soft slack penalty and contraction factor. The goal is to preserve the 99.88% solve rate while reducing the performance gap relative to bounded-hard.
4. Add a baseline normal offset-free MPC without Lyapunov contraction using the same bounded target selector. This will clarify the marginal value of the direct Lyapunov constraint.
5. Repeat the four-scenario run over multiple random setpoint schedules or disturbances. This single run is informative, but supervisor-level conclusions should eventually include variability across seeds.
6. Consider an admissible-reference formulation where the artificial steady target is an MPC decision variable, following the spirit of Limon et al. (2008), if the current separate target-then-MPC architecture continues to create large target jumps.

## 6. Final Takeaway

The direct Lyapunov MPC method is scientifically useful now because it exposes the exact failure mechanism. The problem is not the output-disturbance target equations: those solve cleanly. The problem is that exact output targets are often not admissible under the input constraints. Bounded target projection fixes the main issue. Hard Lyapunov contraction then gives the best reward when feasible, and soft Lyapunov contraction gives a robust fallback with very little slack when paired with bounded target selection.
