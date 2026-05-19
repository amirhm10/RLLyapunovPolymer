# Direct Lyapunov Target Quality Gate And RL Guard

Date: 2026-05-18

## Purpose

This report explains the latest three direct-notebook results and documents the method changes that were implemented after the diagnosis. The main goal is to make the controller logic visible enough that the method can be reviewed, tuned, or rolled back before another long notebook run.

The active case is the polymer CSTR direct Lyapunov workflow in scaled deviation coordinates. Manipulated inputs are the coolant and monomer-related flows, usually `Qc` and `Qm`. Outputs are viscosity-like `eta` and reactor temperature `T`. The disturbed tests change process variables such as `Qi`, `Qs`, and `hA`.

The key conclusion is unchanged:

`mpc_only` is often better because it tracks the raw setpoint directly, while the Lyapunov-gated controller can enforce contraction around a target that is a poor certificate anchor under the disturbed plant.

## Figure Summary

The figures below are copied into this report from the reviewed result bundles, with one compact summary figure generated from the three comparison tables. They are intentionally placed near the claims they support so the report can be read without opening the raw `results/` folders.

![Three-bundle metric summary](figures/2026-05-18_direct_lyapunov_target_quality_gate/three_bundle_metric_summary.png)

The summary figure shows the same pattern as the tables: the safe/Lyapunov path has higher mean output RMSE and lower reward than `mpc_only` in all three reviewed bundles. In the RL cases, the actor is accepted most of the time, so the remaining gap is not only a fallback-frequency issue. It is more consistent with a target/certificate alignment issue.

## Result Bundles Reviewed

### No-RL Direct MPC

Bundle:

`results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260518_150230/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Lyap | 0.436 | -5.70 |
| mpc_only | 0.357 | -3.88 |

Tail tracking is the strongest evidence:

| Case | Final physical error |
|---|---:|
| Lyap | `[0.125, -0.598]` |
| mpc_only | `[0.004, -0.020]` |

Interpretation:

The `mpc_only` run nearly removes the tail offset, while the Lyapunov path keeps a clear residual error, especially in the temperature-like output. This is not an RL reward issue because this notebook does not use an RL policy. The failure has to be in the direct target, disturbance model, finite-horizon constraint interaction, or Lyapunov certificate anchor.

![No-RL direct output overlay](figures/2026-05-18_direct_lyapunov_target_quality_gate/direct_outputs_overlay.png)

The output overlay is the most direct visual evidence. Both controllers see the same disturbed two-setpoint schedule, but the Lyapunov case shows larger transients and a visible late deviation, while `mpc_only` stays closer to the dashed raw setpoint.

![No-RL direct reference errors](figures/2026-05-18_direct_lyapunov_target_quality_gate/direct_reference_errors.png)

The reference-error bars separate target quality from tracking quality. The Lyapunov case has a larger mean target-setpoint gap and a larger mean output-setpoint gap than `mpc_only`, which supports the diagnosis that the certificate target can be a worse anchor than the raw setpoint.

### RL Cold-Start

Bundle:

`results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260518_165924/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Safe gate | 0.265 | -3.209 |
| mpc_only | 0.239 | -2.225 |

Interpretation:

The safe gate improves over the worse no-RL Lyapunov behavior, but it still underperforms `mpc_only`. The likely reason is the same: actions are judged against the Lyapunov target, not directly against raw setpoint performance. A safe action can be poor if the target used for safety is poor.

![Cold-start RL last-episode output overlay](figures/2026-05-18_direct_lyapunov_target_quality_gate/rl_cold_outputs_last_episode.png)

![Cold-start RL safety-gate rates](figures/2026-05-18_direct_lyapunov_target_quality_gate/rl_cold_rates.png)

In the cold-start run, the last-episode output overlay shows the safe-gate policy is not catastrophically wrong, but it still leaves more tracking structure than `mpc_only`. The gate-rate plot shows that most steps are verified and accepted, so the issue is not simply excessive fallback. The accepted safe action can still be inferior for raw setpoint tracking.

### RL Pretrained

Bundle:

`results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260518_165928/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Safe gate | 0.255 | -3.036 |
| mpc_only | 0.245 | -2.249 |

Interpretation:

Pretraining narrows the gap, but it does not remove it. That means the bottleneck is not only policy quality. A stronger policy still passes through a target/certificate layer whose reference may be misaligned with the raw setpoint under disturbance.

![Pretrained RL last-episode output overlay](figures/2026-05-18_direct_lyapunov_target_quality_gate/rl_pretrained_outputs_last_episode.png)

![Pretrained RL safety-gate rates](figures/2026-05-18_direct_lyapunov_target_quality_gate/rl_pretrained_rates.png)

The pretrained actor has slightly better reward and RMSE than cold-start safe-gate RL, but the gap to `mpc_only` remains. This figure pair supports the interpretation that better policy initialization helps locally but does not remove the target/certificate bottleneck.

## What The Numbers Mean

The three bundles form a consistent picture:

| Evidence | Meaning |
|---|---|
| No-RL Lyap worse than `mpc_only` | The problem exists before RL enters. |
| `mpc_only` tail error is near zero | The plant and baseline MPC can still track the disturbed two-setpoint schedule. |
| Lyap tail error remains large | The Lyapunov target/certificate path can anchor the controller away from the raw setpoint. |
| RL safe gate close but worse | RL is not the first bottleneck; the gate and target are. |
| Pretraining helps only slightly | Better policy initialization does not fix a poor target selector. |

The important separation is:

- Raw tracking quality: how well the closed loop follows the requested setpoint.
- Target quality: how close the steady target is to the requested setpoint and how small its residual is.
- Certificate quality: whether a one-step Lyapunov decrease is meaningful around that target.

The old method mixed these. It could certify contraction even when the target itself was the wrong object to contract around.

## Full Mathematical Logic

This section writes the controller in rendered mathematical form. Symbols are defined in prose so the equations can be read as method logic rather than as code snippets.

### Physical Plant And Scaled Deviation Coordinates

The physical polymer CSTR can be viewed as:

$$
\begin{aligned}
x_{\mathrm{phys}}(k+1) &= f_{\mathrm{phys}}\!\left(x_{\mathrm{phys}}(k), u_{\mathrm{phys}}(k), p_{\mathrm{dist}}(k)\right),\\
y_{\mathrm{phys}}(k) &= h_{\mathrm{phys}}\!\left(x_{\mathrm{phys}}(k), u_{\mathrm{phys}}(k), p_{\mathrm{dist}}(k)\right).
\end{aligned}
$$

Here $u_{\mathrm{phys}} = [Q_c, Q_m]$, $y_{\mathrm{phys}} = [\eta, T]$, and $p_{\mathrm{dist}}$ collects plant disturbances such as $Q_i$, $Q_s$, and $hA$.

The direct Lyapunov notebooks do not optimize directly in physical units. They mostly use scaled deviation variables around the nominal steady state:

$$
\begin{aligned}
x(k) &= S_x\!\left(x_{\mathrm{phys}}(k)-x_{\mathrm{ss}}\right),\\
u(k) &= S_u\!\left(u_{\mathrm{phys}}(k)-u_{\mathrm{ss}}\right),\\
y(k) &= S_y\!\left(y_{\mathrm{phys}}(k)-y_{\mathrm{ss}}\right),\\
y_{\mathrm{sp}}(k) &= S_y\!\left(y_{\mathrm{sp,phys}}(k)-y_{\mathrm{ss}}\right).
\end{aligned}
$$

The local prediction model is then written as:

$$
\begin{aligned}
x(k+1) &= A x(k) + B u(k),\\
y(k) &= C x(k) + d_y(k).
\end{aligned}
$$

This coordinate choice matters. A target that looks close in scaled deviation coordinates may still produce a noticeable physical error in `eta` or `T`, and a physical disturbance such as a heat-transfer change is not guaranteed to look like a pure additive output disturbance.

### Offset-Free Observer

The observer state is augmented:

$$
\hat z(k) =
\begin{bmatrix}
\hat x(k)\\
\hat d(k)
\end{bmatrix}.
$$

The generic observer update is:

$$
\begin{aligned}
e_y(k) &= y_{\mathrm{meas}}(k)-C_{\mathrm{aug}}\hat z(k),\\
\hat z(k+1) &= A_{\mathrm{aug}}\hat z(k)+B_{\mathrm{aug}}u(k)+L e_y(k),\\
\hat x(k+1) &= \mathrm{state\ block}\!\left(\hat z(k+1)\right),\\
\hat d(k+1) &= \mathrm{disturbance\ block}\!\left(\hat z(k+1)\right).
\end{aligned}
$$

For the frozen output-disturbance target, the disturbance estimate is interpreted as an additive output correction:

$$
y_s = Cx_s + \hat d_y.
$$

This is the key modelling assumption behind the target selector. If the real disturbed plant changes the state dynamics instead of only shifting the output, the target can be feasible for the frozen model and still be a poor anchor for the physical plant.

### Steady Target Selector

At every control step, the target selector finds a steady target. The decision variables are $x_s$ and $u_s$. The data are the requested setpoint $y_{\mathrm{sp}}(k)$, the disturbance estimate $\hat d(k)$, the input smoothing reference $u_{\mathrm{ref}}$, and the state smoothing reference $x_{\mathrm{ref}}$. The input target must obey:

$$
u_{\min} \le u_s \le u_{\max}.
$$

The steady-state equations are:

$$
\begin{aligned}
x_s &= A x_s + B u_s,\\
y_s &= C x_s + \hat d_y.
\end{aligned}
$$

Equivalently, the residuals are:

$$
\begin{aligned}
r_{\mathrm{dyn}} &= x_s-Ax_s-Bu_s,\\
r_y &= Cx_s+\hat d_y-y_{\mathrm{sp}}(k).
\end{aligned}
$$

The primary target-quality objective is:

$$
J_{\mathrm{primary}}
=
r_{\mathrm{dyn}}^\top W_{\mathrm{dyn}} r_{\mathrm{dyn}}
+
r_y^\top W_y r_y.
$$

In compact notation, $\|v\|_W^2 = v^\top Wv$.

The old bounded least-squares target mixed output quality and smoothing in one problem:

$$
\begin{aligned}
\min_{x_s,u_s}\quad
&J_{\mathrm{primary}}
+ w_u\|u_s-u_{\mathrm{ref}}\|_2^2
+ w_x\|x_s-x_{\mathrm{ref}}\|_2^2\\
\mathrm{s.t.}\quad
&u_{\min}\le u_s\le u_{\max}.
\end{aligned}
$$

That objective can sacrifice output fit to stay close to the previous input or previous target state. In a constrained disturbed run, this can make the selected target smooth but wrong.

The lexicographic target solve separates the roles. Stage 1 finds the best reachable target:

$$
\begin{aligned}
J_1 =
\min_{x_s,u_s}\quad
&J_{\mathrm{primary}}\\
\mathrm{s.t.}\quad
&u_{\min}\le u_s\le u_{\max}.
\end{aligned}
$$

Stage 2 then smooths the target only while preserving Stage-1 quality:

$$
\begin{aligned}
\min_{x_s,u_s}\quad
&w_u\|u_s-u_{\mathrm{ref}}\|_2^2
+w_x\|x_s-x_{\mathrm{ref}}\|_2^2\\
\mathrm{s.t.}\quad
&u_{\min}\le u_s\le u_{\max},\\
&J_{\mathrm{primary}}\le J_1+\varepsilon_{\mathrm{lex}}.
\end{aligned}
$$

The logic is: first find the best reachable output target, then smooth only inside a small loss of primary target quality.

### Direct Lyapunov Tracking MPC

Given a target $x_s,u_s,y_s$, the direct MPC solves for $x_0,\ldots,x_N$, $u_0,\ldots,u_{N-1}$, and optional slack variables. The initial condition and prediction model are:

$$
\begin{aligned}
x_0 &= \hat x(k),\\
x_{i+1} &= A x_i + B u_i,\\
y_i &= Cx_i+\hat d_y.
\end{aligned}
$$

The tracking objective has the usual finite-horizon form:

$$
\begin{aligned}
\min_{\{x_i,u_i\}}\quad
&\sum_{i=0}^{N-1}\|y_i-y_{\mathrm{ref},i}\|_{Q_y}^2
+\sum_{i=0}^{N-1}\|u_i-u_s\|_{R_u}^2\\
&+\sum_{i=0}^{N-1}\|u_i-u_{i-1}\|_{R_{\Delta u}}^2
+\|x_N-x_s\|_P^2
+J_{\mathrm{slack}}.
\end{aligned}
$$

The controller has two possible references:

$$
\begin{aligned}
\text{target tracking:}\quad y_{\mathrm{ref},i} &= y_s,\\
\text{raw setpoint tracking:}\quad y_{\mathrm{ref},i} &= y_{\mathrm{sp}}(k+i).
\end{aligned}
$$

The failure mode diagnosed in this report appears when the optimizer is judged or constrained around `x_s` even when the raw setpoint `y_sp` is the real performance target.

The first-step Lyapunov decrease uses:

$$
\begin{aligned}
e_{\mathrm{now}} &= x_0-x_s,\\
e_{\mathrm{next}} &= x_1-x_s,\\
V(e) &= e^\top P e.
\end{aligned}
$$

The hard contraction condition is:

$$
V(e_{\mathrm{next}})
\le
\rho V(e_{\mathrm{now}})+\varepsilon_{\mathrm{lyap}}.
$$

The terminal condition has the same anchor:

$$
V(x_N-x_s)\le r_{\mathrm{terminal}}+\varepsilon_{\mathrm{terminal}}.
$$

These inequalities are only useful for raw setpoint tracking when `x_s` is a good representation of the requested setpoint under the current disturbance. If `x_s` is poor, the Lyapunov certificate can make the controller faithfully contract toward the wrong anchor.

### Target Quality Gate

The new target-quality gate computes three diagnostic quantities:

$$
\begin{aligned}
m_{\infty} &= \|y_s-y_{\mathrm{sp}}(k)\|_{\infty},\\
r_{\mathrm{target}} &= \text{target residual norm reported by the selector},\\
\Delta x_{s,\infty} &= \|x_s-x_{s,\mathrm{prev}}\|_{\infty}.
\end{aligned}
$$

The target is accepted only if:

$$
\begin{aligned}
m_{\infty} &\le m_{\infty,\max},\\
r_{\mathrm{target}} &\le r_{\max},\\
\Delta x_{s,\infty} &\le \Delta x_{s,\infty,\max}.
\end{aligned}
$$

When `target_quality.enabled=True` and the policy is `bypass_hard_lyap`, the logic is:

- If the target is acceptable, keep the hard first-step Lyapunov contraction, keep the terminal set constraint, and use the normal target/certificate path.
- If the target is poor, log `target_quality_bypass=True`, track the raw setpoint in the direct solver, disable hard first-step contraction, and skip the terminal set constraint.

This does not claim the poor target is safe. It says the target is not reliable enough to be used as a hard certificate anchor.

### RL Safety Gate

The direct RL safety gate starts from an actor action:

$$
a(k)=\pi_{\theta}(s_{\mathrm{RL}}(k)).
$$

In the original full-authority interpretation:

$$
u_{\mathrm{cand}}=\mathrm{map\_to\_bounds}(a(k)).
$$

In the residual-RL interpretation:

$$
\begin{aligned}
\tilde u_{\mathrm{cand}} &= u_{\mathrm{base}}+\alpha(k)a(k),\\
u_{\mathrm{cand}} &= \mathrm{clip}(\tilde u_{\mathrm{cand}},u_{\min},u_{\max}).
\end{aligned}
$$

The authority can shrink near the setpoint:

$$
e_{\infty}(k)=\|y(k)-y_{\mathrm{sp}}(k)\|_{\infty}.
$$

$$
\alpha(k)=
\begin{cases}
\alpha_{\min}\alpha_0, & e_{\infty}(k)\le e_{\mathrm{shrink}},\\
\alpha_0, & e_{\infty}(k)>e_{\mathrm{shrink}}.
\end{cases}
$$

The safety gate checks whether the candidate respects input bounds, move limits, solver status, and Lyapunov decrease around the selected target:

$$
\begin{aligned}
\mathrm{safe}(u_{\mathrm{cand}})=
&\ \mathrm{bounds\_ok}
\land \mathrm{move\_ok}
\land \mathrm{target\_success}\\
&\land
V(x_{\mathrm{next,cand}}-x_s)
\le
\rho V(x_{\mathrm{now}}-x_s)+\varepsilon.
\end{aligned}
$$

If `candidate_safe=False`, the controller applies the fallback direct MPC action. If `candidate_safe=True`, the old logic could accept the action even when its one-step raw tracking cost was worse than the fallback.

### Performance Guard

The performance guard adds the missing raw-tracking comparison. It computes:

$$
J_{\mathrm{perf}}(u)
=
\|y_{\mathrm{next}}(u)-y_{\mathrm{sp}}(k+1)\|_{Q_{\mathrm{perf}}}^2
+
\|u-u_{\mathrm{prev}}\|_{R_{\mathrm{perf}}}^2.
$$

Then it compares the safe RL candidate against a reference action. The strongest reference is the direct-MPC fallback action:

$$
u_{\mathrm{ref,perf}} = u_{\mathrm{direct\ MPC}}.
$$

In a cheaper ablation, the reference can instead be the previous input:

$$
u_{\mathrm{ref,perf}}=u_{\mathrm{prev}}.
$$

The safe RL action is accepted only if:

$$
J_{\mathrm{perf}}(u_{\mathrm{cand}})
\le
(1+\tau_{\mathrm{rel}})J_{\mathrm{perf}}(u_{\mathrm{ref,perf}})
\tau_{\mathrm{abs}}.
$$

This is the correct separation of roles for the current failure mode:

- The Lyapunov gate answers: "Is the action certifiable around the current target?"
- The performance guard answers: "Is the action at least competitive for raw setpoint tracking?"

### Reward Maintenance Terms

The baseline RL reward can be summarized as:

$$
r_{\mathrm{base}}(k)
=
-\|y(k)-y_{\mathrm{sp}}(k)\|_{Q_r}^2
-\|u(k)-u(k-1)\|_{R_r}^2
-r_{\mathrm{safety/fallback}}(k).
$$

The maintenance additions activate only inside a tracking band:

$$
\mathrm{inside\_band}(k)
\Longleftrightarrow
\|y(k)-y_{\mathrm{sp}}(k)\|_{\infty}
\le
b_{\mathrm{maint}}.
$$

When inside the band:

$$
\begin{aligned}
r(k)
=&\ r_{\mathrm{base}}(k)
-w_{\mathrm{move}}\|u(k)-u(k-1)\|_2^2\\
&-w_{\mathrm{jitter}}
\left\|
\left(y(k)-y_{\mathrm{sp}}(k)\right)
-
\left(y(k-1)-y_{\mathrm{sp}}(k-1)\right)
\right\|_2^2\\
&+b_{\mathrm{dwell}}\,n_{\mathrm{dwell}}(k).
\end{aligned}
$$

These terms are useful after the target and gate are fixed. They should not be used to explain the pure `mpc_only` offset, because `mpc_only` does not optimize the RL reward.

## Previous Direct Lyapunov Method

This section reconstructs the method before the new changes.

### Coordinates

Most direct-controller logic is in scaled deviation coordinates.

Important variables:

- `xhatdhat`: augmented observer state, made of physical-state estimate and disturbance estimate.
- `u_prev_dev`: previous input in scaled deviation coordinates.
- `y_sp_k`: setpoint at step `k`, also in scaled deviation coordinates.
- `u_dev_min`, `u_dev_max`: input bounds in scaled deviation coordinates.
- `x_s`, `u_s`, `d_s`, `y_s`: steady target selected for the current setpoint and disturbance estimate.

Observer update:

$$
\hat z(k+1)=A_{\mathrm{aug}}\hat z(k)+B_{\mathrm{aug}}u(k)+L\,e_y(k).
$$

where:

$$
e_y(k)=y_{\mathrm{measured,scaled}}(k)-C_{\mathrm{aug}}\hat z(k).
$$

### Old Target Solve

The direct target layer solved a frozen output-disturbance target. In the output-disturbance case, the disturbance estimate acts directly at the output:

$$
y_s=Cx_s+\hat d_y.
$$

The target equations were:

$$
x_s=Ax_s+Bu_s.
$$

The output target should also satisfy:

$$
y_s \approx y_{\mathrm{sp}}.
$$

When the exact target was outside input bounds, the old bounded fallback used a single least-squares problem. That problem combined:

- steady-state dynamic residual
- output-setpoint mismatch
- previous-input anchor, if `u_ref_weight` was nonzero
- previous-state smoothing, if `x_ref_weight` was nonzero

The problem is that these terms were all in one objective. A strong previous-input or previous-state regularizer could pull the target away from the best reachable output. That is useful for smoothness but dangerous if it redefines the output target.

### Old Direct Lyapunov MPC Step

Once the target was selected, the direct tracking MPC solved over an input sequence. The output objective tracked either raw `y_sp_k` or `y_s`, depending on config. In the direct notebooks here, the important behavior is that the Lyapunov certificate was still formed around `x_s`.

The hard first-step Lyapunov condition was effectively:

$$
V(x_{\mathrm{next}}-x_s)
\le
\rho V(x_{\mathrm{now}}-x_s)+\varepsilon.
$$

If `x_s` is a poor target, this condition can reject or reshape actions that would track the raw setpoint well.

### Old RL Direct Gate

For the direct RL safety gate:

1. The actor proposed an action.
2. The proposed input was mapped to input bounds.
3. The direct target was recomputed.
4. The candidate was checked for bounds, move limits, and Lyapunov decrease around the direct target.
5. If the candidate was unsafe, the direct tracking MPC fallback was applied.

The missing piece was a performance check. A candidate could be Lyapunov-safe but still worse than a fallback or direct MPC action for raw setpoint tracking.

## Root Cause Diagnosis

### Why `mpc_only` Wins In The No-RL Disturbed Run

`mpc_only` does not enforce the direct Lyapunov certificate. It solves the offset-free tracking problem against the raw setpoint. In the latest no-RL result, this gives a final physical error of `[0.004, -0.020]`, which is essentially zero at the tail.

The Lyapunov controller solves a more constrained problem. Even if the output objective uses the raw setpoint, the hard contraction and terminal ingredients are built around `x_s`. If `x_s` is produced by a target selector that is not consistent with the disturbed plant, the optimizer can be pushed toward the certificate target instead of the raw setpoint.

So the no-RL result says:

- The baseline MPC model and horizon are good enough to nearly remove tail offset.
- The Lyapunov target/certificate layer is introducing a competing objective.
- The reward function cannot be the root cause because no RL reward is optimized in this controller.

### Why RL Does Not Fully Fix It

The RL policy can only propose actions. The gate decides what is allowed. If the gate certifies safety around a poor target, then policy improvement does not guarantee raw tracking improvement.

Pretraining helps because the actor starts closer to useful actions. But the pretrained result still loses to `mpc_only`, so the gate logic and target selector remain the bottleneck.

### Why `mpc_only` Can Still Sometimes Have Offset

When `mpc_only` has offset, that is not caused by the RL reward. `mpc_only` does not optimize the RL reward. Offset in `mpc_only` points to:

- disturbance-model mismatch
- target calculation mismatch
- observer augmentation mismatch
- finite-horizon effects
- active input constraints
- move suppression or input saturation
- plant changes not represented by a frozen output disturbance

Reward tuning can improve RL maintenance behavior, but it cannot explain offset in a pure MPC-only run.

## Implemented Method Changes

This section explains exactly what changed and where.

### Change 1: Target Quality Gate

Main file:

`Lyapunov/direct_lyapunov_mpc.py`

Key functions:

- `DEFAULT_DIRECT_TARGET_CONFIG`
- `_target_quality_config`
- `_annotate_target_quality`
- `prepare_direct_output_disturbance_step`
- `solve_direct_tracking_from_target`

New config structure:

```python
direct_target_config = {
    "target_quality": {
        "enabled": True,
        "policy": "bypass_hard_lyap",
        "max_mismatch_inf": 0.03,
        "max_residual_norm": 0.10,
        "max_rate_inf": 0.20,
    }
}
```

All thresholds are in scaled deviation coordinates.

The gate computes:

| Quantity | Meaning |
|---|---|
| `target_quality_mismatch_inf` | Infinity norm of `y_s - y_sp`. |
| `target_quality_residual_norm` | Main target residual norm, using available target diagnostics. |
| `target_rate_inf` | Infinity norm of target-state jump from previous successful target. |
| `target_quality_ok` | True if all enabled checks pass. |
| `target_quality_reason` | Text label explaining why a target was poor. |
| `target_quality_bypass` | True when a poor target should not receive hard Lyapunov enforcement. |

Step-by-step behavior:

1. The target selector solves for `x_s`, `u_s`, `d_s`, and `y_s`.
2. The quality gate compares `y_s` to raw `y_sp_k`.
3. The gate checks the target residual.
4. The gate checks whether `x_s` jumps too far from the previous target.
5. If all active checks pass, the normal Lyapunov controller is used.
6. If a target is poor and `policy="bypass_hard_lyap"`, the controller logs a bypass.
7. During bypass, the direct solver tracks raw `y_sp_k`.
8. During bypass, hard first-step contraction is disabled.
9. During bypass, the terminal set constraint is skipped.

What did not change by default:

- `target_quality.enabled` defaults to `False`.
- Existing notebooks keep the old behavior unless the config is enabled.

Important review note:

The current implementation still lets a numerically successful but poor target update `x_target_prev_success_next`. If we want poor targets not to seed the next target-smoothing reference, change the update condition from `target_success` to `target_success and target_quality_ok`. This is a reasonable follow-up if the next run shows target-quality bypasses clustered after large target jumps.

![Direct target residual and bounded activity](figures/2026-05-18_direct_lyapunov_target_quality_gate/direct_target_residual_bounded_activity.png)

This diagnostic explains why the target-quality gate is needed. A target solve can be numerically successful while still having large target residual or bounded-solution activity. Such a target should not automatically receive hard Lyapunov authority.

![Direct solver and contraction rates](figures/2026-05-18_direct_lyapunov_target_quality_gate/direct_solver_contraction_rates.png)

The contraction-rate plot separates feasibility from usefulness. A high contraction or solver-success rate does not prove raw tracking quality if contraction is measured around a poor target.

### Change 2: Lexicographic Bounded Target Solve

Main files:

- `analysis/steady_state_debug_analysis.py`
- `Lyapunov/frozen_output_disturbance_target.py`

New config:

```python
direct_target_config = {
    "solve_strategy": "lexicographic",
    "lexicographic_primary_tol_abs": 1.0e-10,
    "lexicographic_primary_tol_rel": 1.0e-8,
    "lexicographic_maxiter": 200,
    "lexicographic_ftol": 1.0e-10,
}
```

Old strategy:

`solve_strategy="legacy_ls"`

This keeps the previous single-stage least-squares behavior.

New strategy:

`solve_strategy="lexicographic"`

Stage 1:

Minimize the primary steady-target quality. In reduced form this is the output mismatch:

$$
J_{\mathrm{primary}}=\|G u_s-b_y\|_2^2.
$$

In full form it minimizes the stacked steady-state residual:

$$
J_{\mathrm{primary}}=\left\|M
\begin{bmatrix}
x_s\\
u_s
\end{bmatrix}
-b\right\|_2^2.
$$

Stage 2:

Minimize smoothing only inside a small tolerance of the Stage 1 primary cost:

$$
J_{\mathrm{anchor}}
=
\|u_s-u_{\mathrm{ref}}\|_{W_u}^2
+
\|x_s-x_{\mathrm{ref}}\|_{W_x}^2.
$$

The Stage 2 constraint is:

$$
J_{\mathrm{primary}}\le J_1+\varepsilon_{\mathrm{lex}}.
$$

Why this matters:

The previous target solve could sacrifice output fit to satisfy `u_ref` or `x_ref` smoothing. The new solve says: first get the best reachable target, then smooth only if smoothing does not damage target quality.

Synthetic validation:

| Strategy | u target | output residual |
|---|---:|---:|
| legacy_ls | 0.009901 | 0.990099 |
| lexicographic | 0.200000 | 0.800000 |

The synthetic case had an unreachable setpoint and a strong previous-input anchor at zero. The old solve was pulled almost to zero input. The lexicographic solve stayed at the upper bound because that minimized output mismatch first.

What can be changed:

- Use `legacy_ls` if the old behavior is needed for comparison.
- Increase `lexicographic_primary_tol_abs` or `lexicographic_primary_tol_rel` if small target-quality sacrifices are acceptable for smoother inputs.
- Keep these tolerances small for the disturbed two-setpoint runs until we confirm the target selector is no longer the bottleneck.

### Change 3: Disturbance Model Mode

Main file:

`Lyapunov/frozen_output_disturbance_target.py`

New config:

```python
direct_target_config = {
    "disturbance_model_mode": "output"
}
```

Allowed values:

| Mode | Meaning |
|---|---|
| `output` | Existing frozen output-disturbance model. |
| `state_via_B` | Use generic augmented target selector when the augmented model exposes state disturbance channels. |
| `mixed` | Use generic augmented target selector when both state and output disturbance effects are represented. |

Important limitation:

This change exposes the target-side path. It does not magically redesign the observer, augmentation matrices, or disturbance estimator. For `state_via_B` and `mixed` to be meaningful, the notebooks must pass an augmented model and observer gain whose disturbance states actually represent the disturbance channel.

Why this matters:

The disturbed tests change `Qi`, `Qs`, and `hA`. These changes do not necessarily appear as a pure additive output bias. A frozen output disturbance can compensate some offset, but it can also produce a target that is internally inconsistent with how the plant actually moved.

What can be changed:

- Keep `output` for compatibility and baseline comparisons.
- Test `state_via_B` only after confirming the augmented matrices and observer gain are built for that disturbance structure.
- Log the mode in every bundle so target diagnostics can be compared mode-by-mode.

### Change 4: Direct RL Performance Guard

Main file:

`Simulation/run_rl_lyapunov.py`

New function/config:

- `_normalize_performance_guard_config`
- `performance_guard_config`

Example:

```python
performance_guard_config = {
    "enabled": True,
    "reference_policy": "direct_mpc",
    "abs_tol": 0.0,
    "rel_tol": 0.05,
}
```

Step-by-step behavior:

1. RL proposes `u_rl`.
2. The direct Lyapunov gate checks whether `u_rl` is safe.
3. If `u_rl` is not safe, fallback MPC is used as before.
4. If `u_rl` is safe and the performance guard is disabled, `u_rl` can be accepted as before.
5. If `u_rl` is safe and the performance guard is enabled, the code computes a one-step raw tracking cost.
6. The same cost is computed for a reference action, either direct MPC fallback or hold-previous input.
7. If the RL action is worse by more than tolerance, it is rejected with `reject_reason="performance_guard"`.
8. The controller then uses the direct fallback path.

The one-step raw tracking cost is:

$$
J_{\mathrm{perf}}=J_y+J_{\Delta u}.
$$

where:

$$
J_y=\sum_i Q_i\left(y_{\mathrm{next},i}-y_{\mathrm{sp},i}\right)^2.
$$

$$
J_{\Delta u}=\sum_j R_j\left(u_j-u_{\mathrm{prev},j}\right)^2.
$$

Why this matters:

Safety is necessary but not sufficient. A safe RL action can be worse than the fallback for raw tracking. This guard adds the missing performance logic.

What can be changed:

- `reference_policy="direct_mpc"` is stronger but more expensive because it can solve the direct fallback for comparison.
- `reference_policy="hold_prev"` is cheaper but weaker.
- Increase `rel_tol` if the guard rejects too many actions and collapses into MPC-only behavior.
- Use the guard first for short RL smoke tests, not full training immediately.

### Change 5: Residual RL Option

Main file:

`Simulation/run_rl_lyapunov.py`

New function/config:

- `_normalize_residual_rl_config`
- `residual_rl_config`

Example:

```python
residual_rl_config = {
    "enabled": True,
    "baseline_policy": "offset_free_mpc",
    "authority_scale": 0.20,
    "shrink_error_inf": 0.05,
    "min_authority_scale": 0.10,
}
```

Old RL action interpretation:

$$
u=\mathrm{map\_actor\_action\_to\_input\_bounds}(a).
$$

New residual option:

$$
u=u_{\mathrm{baseline}}+\alpha a.
$$

Then the input is clipped to bounds.

Step-by-step behavior:

1. Compute a baseline input.
2. The baseline is `offset_free_mpc` if available, otherwise previous input.
3. Compute residual authority.
4. Optionally shrink authority near the setpoint using `shrink_error_inf`.
5. Add the actor residual to the baseline.
6. Clip to input bounds.
7. Pass the resulting action to the safety/performance gate.

Why this matters:

The latest results show that `mpc_only` is already strong. Residual RL lets the policy improve or maintain MPC behavior instead of replacing it with a full-authority action.

What can be changed:

- Use `authority_scale=0.10` to `0.25` for conservative tests.
- Use a larger value only after the target-quality gate is working.
- Use `baseline_policy="previous_input"` only for ablations; `offset_free_mpc` is the more meaningful baseline.

### Change 6: RL Maintenance Reward Terms

Main file:

`TD3Agent/reward_functions.py`

New optional arguments:

```python
maintenance_band_scale=1.0
maintenance_move_weight=0.0
jitter_weight=0.0
dwell_bonus=0.0
```

Defaults preserve old behavior.

Step-by-step reward addition:

1. The original relative QR reward is computed.
2. The code checks whether all outputs are inside the maintenance band.
3. If inside the band, an additional move penalty can be applied.
4. If the previous output error exists, an output-jitter penalty can be applied.
5. If the output remains inside the band, the dwell counter increases.
6. A dwell bonus can be added proportional to the dwell count.

Why this matters:

The RL notebooks show maintenance and jitter concerns near the setpoint. A reward that only encourages entering the band may not sufficiently discourage small near-setpoint moves or output oscillation.

What can be changed:

- Keep all maintenance weights zero until target-quality and performance-gate fixes are tested.
- Add a small `maintenance_move_weight` first.
- Add `jitter_weight` only if the output trace still oscillates after the gate changes.
- Use `dwell_bonus` carefully, because too large a dwell bonus can encourage passive behavior.

## Export And Diagnostics Added

Direct debug exports now include:

- `target_quality_enabled`
- `target_quality_ok`
- `target_quality_reason`
- `target_quality_policy`
- `target_quality_bypass`
- `target_quality_mismatch_inf`
- `target_quality_residual_norm`
- `target_rate_inf`

RL safety debug exports now include:

- target-quality fields
- performance-guard fields
- residual-RL fields

Useful summary metrics:

| Metric | Why it matters |
|---|---|
| `target_quality_ok_rate` | Fraction of steps with acceptable target anchors. |
| `target_quality_bypass_rate` | Fraction of steps where hard Lyapunov was bypassed. |
| `target_rate_inf_max` | Largest target jump. |
| `performance_guard_ok_rate` | Fraction of checked RL actions that passed raw tracking comparison. |
| `target_quality_mismatch_inf_max` | Worst target-setpoint mismatch. |
| `target_quality_residual_norm_max` | Worst target residual. |

## Recommended Config For The Next No-RL Run

### Latest Change 2 Ablation: Lexicographic Target Solve With `y_sp` Versus `y_s` Tracking

After the initial diagnosis, two no-RL direct runs were made with the Change 2 lexicographic bounded target solve and stronger target smoothing weights:

- `results/direct_lyap_ch2_lex/20260518_204423/`: lexicographic target solve, dynamic MPC tracks raw `y_sp`.
- `results/direct_lyap_ch2_lex/20260518_204533/`: lexicographic target solve, dynamic MPC tracks selected target `y_s`.

Both runs used the disturbed two-setpoint plant case with `u_ref_weight=0.5`, `x_ref_weight=0.5`, `disturbance_model_mode="output"`, and `target_quality.enabled=False`. The `mpc_only` case is unchanged and remains the baseline.

![Change 2 summary](figures/2026-05-18_change2_lex_ysp_vs_ys/change2_ysp_vs_ys_summary.png)

The result is mixed. Tracking `y_s` is more internally consistent with target-selector MPC and it substantially reduces the error to the active target. However, it does not beat `mpc_only` on raw setpoint performance or reward.

| Run | Case | Tracks | RMSE mean | Reward mean | Mean target gap | Mean active-target error |
|---|---|---|---:|---:|---:|---:|
| 204423 | Lyap | raw `y_sp` | 0.853 | -14.52 | 1.895 | 1.408 |
| 204423 | mpc_only | raw `y_sp` | 0.357 | -3.88 | 0.494 | 0.450 |
| 204533 | Lyap | selected `y_s` | 0.655 | -22.94 | 1.786 | 0.291 |
| 204533 | mpc_only | selected `y_s` diagnostic | 0.357 | -3.88 | 0.494 | 0.327 |

The most important separation is:

- `y_s` tracking improved the controller's ability to follow the target it was given: mean active-target error dropped from 1.408 to 0.291.
- Raw setpoint performance is still poor over the full run: Lyap RMSE is 0.655 with `y_s` tracking versus 0.357 for `mpc_only`.
- Reward became worse with `y_s` tracking because the reward is still evaluated against raw `y_sp`, and the selected target is often far from `y_sp`.

![Change 2 tail decomposition](figures/2026-05-18_change2_lex_ysp_vs_ys/change2_tail_error_decomposition.png)

The tail-window view is more encouraging but still not enough:

| Run | Tail raw error | Tail active-target error | Tail target gap | Final physical error |
|---|---:|---:|---:|---:|
| Lyap tracks `y_sp` | 1.547 | 1.547 | 2.165 | `[-0.236, 1.423]` |
| Lyap tracks `y_s` | 0.517 | 0.214 | 0.449 | `[0.011, 0.018]` |
| `mpc_only` | 0.054 | 0.052 | 0.006 | `[0.004, -0.020]` |

So `y_s` tracking fixes one part of the logic: the controller can track the selected target. It does not fix the main bottleneck: the selected target is still frequently not a good representation of the requested setpoint under the disturbed plant.

![Lexicographic run tracking raw setpoint](figures/2026-05-18_change2_lex_ysp_vs_ys/lex_ysp_outputs_vs_targets.png)

When the MPC objective tracks raw `y_sp`, the Lyapunov certificate still pulls the trajectory around a poor target anchor. The output deviates strongly, especially in the temperature-like output.

![Lexicographic run tracking selected target](figures/2026-05-18_change2_lex_ysp_vs_ys/lex_ys_outputs_vs_targets.png)

When the MPC objective tracks `y_s`, the trajectory follows the selected target much more closely. But the selected target itself drifts away from the requested setpoint for long intervals, so the closed loop can look good relative to `y_s` while remaining poor relative to `y_sp`.

### Latest Weight Sweep: `0.1` Versus `0.0` Smoothing Weights

Two additional no-RL runs tested the same Change 2 lexicographic target solve with `use_target_output_for_tracking=True`, but reduced the target smoothing penalties:

- `results/direct_lyap_ch2_lex/20260518_205113/`: `u_ref_weight=0.1`, `x_ref_weight=0.1`.
- `results/direct_lyap_ch2_lex/20260518_205354/`: `u_ref_weight=0.0`, `x_ref_weight=0.0`.

This sweep is important because the previous `0.5`, `0.5` run could hide whether the target selector was bad because of the disturbance representation or simply because the secondary objective was holding the target too close to the previous/reference anchor.

![Change 2 weight sweep summary](figures/2026-05-18_change2_weight_sweep/change2_weight_sweep_summary.png)

The zero-weight run is clearly the best Lyapunov case so far. The `0.1`, `0.1` run is worse, not better.

| Run | Weights | RMSE mean | Reward mean | Full raw error | Tail raw error | Tail target gap |
|---|---:|---:|---:|---:|---:|---:|
| 204533 | 0.5 | 0.655 | -22.94 | 0.658 | 0.517 | 0.449 |
| 205113 | 0.1 | 0.802 | -24.50 | 0.837 | 2.021 | 1.914 |
| 205354 | 0.0 | 0.567 | -22.69 | 0.523 | 0.081 | 0.038 |
| 205354 `mpc_only` | diagnostic | 0.357 | -3.88 | 0.287 | 0.054 | 0.006 |

The mechanism is visible in the error decomposition:

![Change 2 weight sweep late decomposition](figures/2026-05-18_change2_weight_sweep/change2_weight_sweep_late_decomposition.png)

With `0.1`, `0.1`, the closed-loop error to the active target is moderate, but the selected target moves far from the requested setpoint late in the run. In the last 200 steps, mean `|y_s-y_sp|` is 1.914, while mean `|y-y_s|` is 0.499. That means the dominant late-run error is target selection, not dynamic MPC tracking.

With `0.0`, `0.0`, the selected target stays close to the requested setpoint in the tail window. Mean `|y_s-y_sp|` drops to 0.038 and mean `|y-y_sp|` drops to 0.081, close to the `mpc_only` tail value of 0.054. The final physical error is also small: `[-0.028, 0.036]`.

![Change 2 weight sweep outputs and targets](figures/2026-05-18_change2_weight_sweep/change2_weight_sweep_outputs_targets.png)

This changes the diagnosis. The target selector is not fundamentally unable to produce useful targets, because the zero-weight run does produce a good late target. But the target selector is still not good enough over the full horizon:

- Full-run Lyap RMSE is 0.567 for zero weights, still worse than 0.357 for `mpc_only`.
- Full-run mean `|y_s-y_sp|` is 0.457 for zero weights, still much larger than the tail value of 0.038.
- The target residual remains large in some intervals: max residual norm is 6.036.
- Reward is still much worse than `mpc_only` because reward is evaluated against raw `y_sp`.

![Change 2 weight sweep residual and reward](figures/2026-05-18_change2_weight_sweep/change2_weight_sweep_residual_reward.png)

Mathematically, the diagnostic split is:

$$
e_{\mathrm{raw}}(k) = y(k) - y_{\mathrm{sp}}(k)
$$

$$
e_{\mathrm{track}}(k) = y(k) - y_s(k)
$$

$$
e_{\mathrm{target}}(k) = y_s(k) - y_{\mathrm{sp}}(k)
$$

and therefore:

$$
e_{\mathrm{raw}}(k) = e_{\mathrm{track}}(k) + e_{\mathrm{target}}(k)
$$

The latest runs show that the bad cases are dominated by `e_target`, not only by `e_track`.

On the disturbance question: if the experimental assumption is that the true disturbance is unmeasured, that is acceptable. We should not claim access to `Qi`, `Qs`, or `hA` as measured controller inputs. But the controller still needs an internal offset or disturbance estimate in the target equations, otherwise the steady target is solved for the wrong plant. The current target selector is effectively solving:

$$
x_s = A x_s + B u_s
$$

$$
y_s = C x_s + \hat d_y
$$

with input bounds and secondary smoothing penalties. So even if the real disturbance is unmeasured, the question is not whether to ignore disturbance completely. The question is whether the estimated output-bias term `d_hat_y` is a good enough internal representation for the target selector. The new evidence says: first remove smoothing and improve the selector logic; only revisit a richer disturbance model if `y_s-y_sp` remains large after that.

### Interpretation Of Change 2

Change 2 helped structurally, but it did not solve the disturbed direct Lyapunov problem by itself.

What it helped:

- It makes the target-selector hierarchy more principled: output/steady-state residual is protected before smoothing is applied.
- In the `y_s`-tracking run, solver success improved to 0.999 compared with 0.990 for the raw-`y_sp` tracking run.
- The maximum target residual dropped from 15.234 in the raw-`y_sp` tracking run to 5.937 in the `y_s` tracking run.
- The final physical error in the `y_s` tracking run was small.

What it did not fix:

- The full-run Lyap RMSE is still worse than `mpc_only`.
- The selected target remains far from the requested setpoint for substantial parts of the run.
- The reward remains much worse because reward is evaluated against `y_sp`, not `y_s`.
- The result is sensitive to target smoothing weights. The zero-weight run is the best Lyapunov case, while the `0.1`, `0.1` run is worse than both `0.5`, `0.5` and `0.0`, `0.0`.

There is also a diagnostic/export issue: the summary still labels bounded target stages as `frozen_output_disturbance_bounded_ls`, even when `solve_strategy="lexicographic"` was passed. The step table does not make the active solve strategy obvious enough. The next export should include `target_solve_strategy` and lexicographic stage costs in the compact comparison table.

### Updated Recommendation

Do not move to RL yet. The latest no-RL evidence says the next work should focus on the target selector:

1. Run `use_target_output_for_tracking=True`, because this is the internally consistent tracking-MPC form.
2. Keep `u_ref_weight=0.0` and `x_ref_weight=0.0` as the current best diagnostic setting.
3. Keep `solve_strategy="lexicographic"` fixed.
4. Judge success by three metrics together: small `y-y_s`, small `y_s-y_sp`, and small `y-y_sp`.
5. Add a target-quality gate so poor targets do not receive hard Lyapunov authority.
6. Only after the zero-weight target selector is healthy should we revisit richer disturbance modeling.

The strongest next hypothesis is no longer simply "wrong disturbance model." The more precise hypothesis is: the target selector is over-regularized or under-gated during difficult intervals. The zero-weight run proves that useful targets are possible, but the full-horizon target quality is still not consistently good enough.

Start with the no-RL direct comparison, not RL training. The goal is to confirm that the target/certificate fix is working before touching reward design.

Suggested first config:

```python
direct_target_config = {
    "solve_strategy": "lexicographic",
    "disturbance_model_mode": "output",
    "target_quality": {
        "enabled": True,
        "policy": "bypass_hard_lyap",
        "max_mismatch_inf": 0.03,
        "max_residual_norm": 0.10,
        "max_rate_inf": 0.25,
    },
}
```

Things to inspect after the run:

- Does `mpc_only` still end near zero tail offset?
- Does Lyap stop enforcing hard contraction when `target_quality_bypass=True`?
- Does Lyap tail error improve relative to `[0.125, -0.598]`?
- Does `target_quality_bypass_rate` cluster around setpoint changes or disturbed intervals?
- Does `target_quality_mismatch_inf` drop under lexicographic solve?

If too many steps bypass, loosen thresholds or fix the disturbance model. If almost no steps bypass but Lyap still has offset, the quality thresholds are too loose or the target residual metric is not catching the bad anchor.

## Recommended Config For Short RL Smoke Test

After the no-RL run looks better, use a short RL smoke test. Do not start with another full 160000-step run.

Suggested first RL guard config:

```python
performance_guard_config = {
    "enabled": True,
    "reference_policy": "direct_mpc",
    "abs_tol": 0.0,
    "rel_tol": 0.05,
}
```

Suggested residual RL config:

```python
residual_rl_config = {
    "enabled": True,
    "baseline_policy": "offset_free_mpc",
    "authority_scale": 0.20,
    "shrink_error_inf": 0.05,
    "min_authority_scale": 0.10,
}
```

Suggested reward maintenance config:

```python
reward_params = {
    "maintenance_band_scale": 1.0,
    "maintenance_move_weight": 0.05,
    "jitter_weight": 0.01,
    "dwell_bonus": 0.0,
}
```

Only tune the reward after target-quality diagnostics show that the gate is no longer certifying around poor targets.

## What I Would Change First If The Next Run Is Still Bad

1. Prevent poor targets from updating the previous-target smoothing reference.
2. Add plots for `target_quality_bypass`, `target_quality_mismatch_inf`, and output error on the same time axis.
3. Compare `output` versus `mixed` disturbance mode only after verifying the augmented observer model.
4. Reduce or disable `x_ref_weight` and `u_ref_weight` for disturbed tests if lexicographic solve still shows target jumps.
5. Increase `performance_guard.rel_tol` if the RL safe gate becomes too conservative.
6. Reduce residual RL authority if action jitter remains near the setpoint.

## Risks And Caveats

The target-quality gate is intentionally conservative. If thresholds are too tight, the controller may bypass hard Lyapunov too often and behave closer to tracking MPC.

The lexicographic target solve is more principled for unreachable setpoints, but it adds a second optimization stage. It should be monitored for solve time in long notebooks.

The `state_via_B` and `mixed` disturbance modes require a compatible augmented model and observer. The current code exposes the path, but the notebooks still need to supply the correct model.

The performance guard adds a performance criterion to a safety gate. This is desirable for the current failure mode, but it can reduce RL exploration if the tolerance is too strict.

The reward maintenance terms are stateful because the dwell and jitter terms depend on previous reward calls. This is suitable for sequential rollout but should be kept in mind if the same reward object is reused across independent episodes without reset.

## Literature Basis

Muske and Badgwell discuss offset-free MPC disturbance models and why disturbance structure matters when rejecting sustained offsets: https://www.sciencedirect.com/science/article/pii/S0959152401000518

Pannocchia and Bemporad emphasize that disturbance model, observer, target calculation, and dynamic controller should be designed together for offset-free MPC: https://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeetac-distmodel.pdf

Shead, Muske, and Rossiter motivate caution around constrained target calculation, because active constraints can drive the controller toward an undesired feasible target: https://www.sciencedirect.com/science/article/abs/pii/S0959152410001812

Limon et al. motivate artificial/admissible references for tracking MPC when requested references change or are not reachable: https://www.sciencedirect.com/science/article/pii/S0959152409002169

Predictive safety-filter work supports the RL-side change: safety certification should be paired with performance logic when safe actions can still be poor tracking choices: https://www.sciencedirect.com/science/article/pii/S0005109821001175

## Validation Completed

Code syntax validation:

`python -m py_compile` passed on the touched modules using a temporary bytecode directory because the normal OneDrive pycache path denied writes.

Synthetic target validation:

| Strategy | u target | output residual |
|---|---:|---:|
| legacy_ls | 0.009901 | 0.990099 |
| lexicographic | 0.200000 | 0.800000 |

Target-solver smoke validation:

The target-solver smoke test passed. The full direct-controller smoke test could not run in this environment because `cvxpy` is not installed for the default Python interpreter.

## Bottom Line

The next experiment should not start with reward tuning or RL. The latest Change 2 weight sweep shows that `y_s` tracking can work, and the zero-smoothing target selector is the best Lyapunov case so far. But full-horizon performance is still worse than `mpc_only`, so the target selector is not yet reliable enough. Keep `solve_strategy="lexicographic"`, `use_target_output_for_tracking=True`, and `u_ref_weight=x_ref_weight=0.0`; then add target-quality gating and inspect the intervals where `y_s-y_sp` is still large. Treat the true disturbance as unmeasured, but keep an internal offset/disturbance estimate in the target equations.
