# Agent-Authority BC Latest Run Analysis

## Objective

This report analyzes the latest rerun after increasing the fixed fallback event penalty from `2.0` to `10.0` in the two safety-gate RL scripts. The direct Lyapunov MPC comparison is kept from the latest available direct run, and the matched MPC-only diagnostics remain visible because they explain how often the Lyapunov gate would have activated without intervention.

Analyzed folders:

- Cold start: `results/ColdStart/20260520_134418`
- Pretrained: `results/Pretrain/20260520_134418`
- Direct LMPC: `results/directLyap/20260520_005354`

Active scripts for future reruns:

- Direct Lyapunov MPC: `DirectLyapunovMPC.py`, saving to `results/directLyap/...`
- Cold-start safety-gate RL: `DirectLyapunovSafetyGateRL_ColdStart.py`, saving to `results/ColdStart/...`
- Pretrained safety-gate RL: `DirectLyapunovSafetyGateRL_Pretrained.py`, saving to `results/Pretrain/...`

## Methodology

This section gives the calculation-level workflow used by the three active scripts. The goal is to make clear what is physical, what is scaled, what is learned by TD3, and what is checked by the Lyapunov safety gate before the numerical results are interpreted.

### 1. Physical Plant, Controlled Variables, And Scaling

The plant is the polymer CSTR. The manipulated input vector is:

$$
u_k^{\rm phys} = [Q_{c,k}, Q_{m,k}]^\top.
$$

The controlled output vector is:

$$
y_k^{\rm phys} = [\eta_k, T_k]^\top.
$$

The scripts keep the nonlinear plant in physical units, but the controller model and the RL state use min-max scaled variables. For any physical vector $v$, the scaling map is:

$$
S(v)=\frac{v-v_{\min}}{v_{\max}-v_{\min}}.
$$

The inverse map used before applying an input to the plant is:

$$
S^{-1}(\bar v)=\bar v(v_{\max}-v_{\min})+v_{\min}.
$$

The MPC and safety calculations use scaled deviations from the steady-state anchors:

$$
\Delta u_k = S(u_k^{\rm phys})-S(u_{\rm ss}^{\rm phys}).
$$

$$
\Delta y_k = S(y_k^{\rm phys})-S(y_{\rm ss}^{\rm phys}).
$$

The setpoint is converted in the same way:

$$
\Delta y_{{\rm sp},k}=S(y_{{\rm sp},k}^{\rm phys})-S(y_{\rm ss}^{\rm phys}).
$$

Each episode contains the two physical setpoint blocks. If the block length is $N_{\rm sp}$ and there are two setpoints, then one episode contains:

$$
N_{\rm ep}=2N_{\rm sp}
$$

control steps. In the current direct script, $N_{\rm sp}=400$. The RL scripts use `DIRECT_DISTURBANCE_SETPOINT_LEN` from the shared direct-study configuration.

### 2. Output-Disturbance Augmented Model

The linear controller model is augmented with an output-disturbance estimate. Let:

$$
\hat z_k=[\hat x_k,\hat d_k]^\top.
$$

The controller prediction model is:

$$
\hat z_{k+1}=A_{\rm aug}\hat z_k+B_{\rm aug}\Delta u_k.
$$

The predicted scaled-deviation output is:

$$
\hat y_k=C_{\rm aug}\hat z_k.
$$

The implementation updates the observer with the previous measured scaled-deviation output. In the rollout loop this is:

$$
e_{{\rm obs},k}= \Delta y_{k-1}-C_{\rm aug}\hat z_k.
$$

$$
\hat z_{k+1}=A_{\rm aug}\hat z_k+B_{\rm aug}\Delta u_{{\rm exec},k}+L e_{{\rm obs},k}.
$$

This is why the stored variable `xhatdhat` is central to both direct LMPC and RL. It carries the model state estimate and the output-disturbance estimate used by the target selector, MPC solver, safety gate, and RL observation.

### 3. Direct Output-Disturbance Target Calculation

The direct Lyapunov path first computes a steady target. The steady target contains state, disturbance, input, and output components:

$$
(x_s,d_s,u_s,y_s).
$$

For the output-disturbance model, the nominal steady equations are:

$$
x_s=A x_s+B u_s.
$$

$$
y_s=Cx_s+d_s.
$$

The disturbance target is tied to the current disturbance estimate:

$$
d_s=\hat d_k.
$$

If the exact target can match the raw setpoint and remain inside the input box, the target is exact:

$$
y_s=\Delta y_{{\rm sp},k}.
$$

$$
\Delta u_{\min} \le u_s \le \Delta u_{\max}.
$$

When the exact target violates the input bounds, the bounded target solver is used. In calculation terms, it searches for the closest admissible steady target by minimizing the steady residual and optional anchoring terms:

$$
J_{\rm target}
=\|x_s-Ax_s-Bu_s\|^2
+\|Cx_s+d_s-\Delta y_{{\rm sp},k}\|^2.
$$

When input and state regularization are enabled, the objective also includes:

$$
J_u=(u_s-u_{\rm ref})^\top R_{u{\rm ref}}(u_s-u_{\rm ref}).
$$

$$
J_x=(x_s-x_{\rm ref})^\top Q_{x{\rm ref}}(x_s-x_{\rm ref}).
$$

In the current three-script workflow, the visible direct target regularization weights are:

$$
u_{\rm prev\ penalty}=0.1.
$$

$$
x_{s,{\rm prev}\ penalty}=0.1.
$$

The important interpretation is that $y_s$ may differ from the raw setpoint when the setpoint is not exactly achievable under the input box and disturbance estimate. This is why the report separates raw setpoint tracking from target diagnostics.

### 4. Direct Lyapunov MPC Optimization

After the target is available, direct LMPC solves over a prediction horizon $N_P=9$ and control horizon $N_C=3$. The decision variables are:

$$
U=\{\Delta u_{0|k},\Delta u_{1|k},\ldots,\Delta u_{N_C-1|k}\}.
$$

The predicted augmented state starts from the current observer state:

$$
z_{0|k}=\hat z_k.
$$

The prediction model is:

$$
z_{i+1|k}=A_{\rm aug}z_{i|k}+B_{\rm aug}\Delta u_{j|k}.
$$

For prediction step $i$, the control index is:

$$
j=\min(i,N_C-1).
$$

The predicted output is:

$$
y_{i+1|k}=C_{\rm aug}z_{i+1|k}.
$$

The direct tracking objective is:

$$
J_{\rm LMPC}=\sum_{i=0}^{N_P-1}(y_{i+1|k}-y_{\rm target})^\top Q_y(y_{i+1|k}-y_{\rm target})
+J_{\Delta u}.
$$

The move penalty is:

$$
J_{\Delta u}=(\Delta u_{0|k}-\Delta u_{k-1})^\top R_{\Delta u}(\Delta u_{0|k}-\Delta u_{k-1}).
$$

For later control moves:

$$
J_{\Delta u}\leftarrow J_{\Delta u}
+\sum_{j=1}^{N_C-1}(\Delta u_{j|k}-\Delta u_{j-1|k})^\top R_{\Delta u}(\Delta u_{j|k}-\Delta u_{j-1|k}).
$$

The input box is enforced at each control move:

$$
\Delta u_{\min}\le \Delta u_{j|k}\le \Delta u_{\max}.
$$

The Lyapunov function is computed on the plant-state portion of the augmented state:

$$
V_k=(\hat x_k-x_s)^\top P_x(\hat x_k-x_s).
$$

The first-step predicted value is:

$$
V_{k+1}^{\rm first}=(x_{1|k}-x_s)^\top P_x(x_{1|k}-x_s).
$$

The hard contraction condition is:

$$
V_{k+1}^{\rm first} \le \rho V_k + \epsilon_{\rm lyap}.
$$

The logged contraction margin is:

$$
m_k=V_{k+1}^{\rm first}-(\rho V_k+\epsilon_{\rm lyap}).
$$

A step satisfies the contraction check when:

$$
m_k\le 0.
$$

In the current scripts:

$$
\rho=0.99.
$$

$$
\epsilon_{\rm lyap}=10^{-3}.
$$

The direct script executes the first input of this LMPC solution. The RL scripts use this same calculation as the fallback action and as the BC teacher action.

### 5. MPC-Only Diagnostic Calculation

The MPC-only rows execute an offset-free MPC baseline, not the safety-gated RL controller. The actual fallback count is therefore:

$$
N_{\rm fallback}^{\rm actual}=0.
$$

For diagnostics, the script still evaluates whether the Lyapunov gate would have rejected the MPC-only candidate. The would-be activation count is:

$$
N_{\rm would}= \sum_{k=1}^{N} I(m_k>0).
$$

The corresponding rate is:

$$
r_{\rm would}=N_{\rm would}/N.
$$

This is the number used in MPC-only fallback-count plots unless a plot is explicitly labeled as actual fallback.

### 6. TD3 State, Actor Action, And Bounds Mapping

The TD3 observation is built from three pieces:

$$
s_k=[S_{\pm 1}(\hat z_k),S_{\pm 1}(\Delta y_{{\rm sp},k}),S_{\pm 1}(\Delta u_{k-1})].
$$

Here $S_{\pm 1}$ maps a variable from its stored min-max range to $[-1,1]$:

$$
S_{\pm 1}(v)=2S(v)-1.
$$

The actor outputs a normalized action:

$$
a_k=\pi_\theta(s_k).
$$

The normalized action is clipped to the actor box:

$$
-1\le a_{k,j}\le 1.
$$

It is then mapped to the scaled input-deviation bounds:

$$
\Delta u_{{\rm rl},k}=\Delta u_{\min}+0.5(a_k+1)(\Delta u_{\max}-\Delta u_{\min}).
$$

When an executed safe input must be stored back as an actor-space action, the inverse map is:

$$
a_{{\rm exec},k}=2\frac{\Delta u_{{\rm exec},k}-\Delta u_{\min}}{\Delta u_{\max}-\Delta u_{\min}}-1.
$$

This action-space mapping is important because the critic replay buffer stores actions in actor coordinates, while the plant and MPC use scaled input-deviation coordinates.

### 7. Agent-Authority Behavioral Cloning Calculation

During the behavioral-cloning phase, the actor remains the candidate policy. At each step, two actions are computed:

$$
\Delta u_{{\rm rl},k}
\quad\hbox{from the TD3 actor}.
$$

$$
\Delta u_{{\rm LMPC},k}
\quad\hbox{from the direct Lyapunov MPC teacher}.
$$

The safety gate receives the actor action:

$$
\Delta u_{{\rm cand},k}=\Delta u_{{\rm rl},k}.
$$

The BC target stored for the actor is the teacher action:

$$
a_{{\rm demo},k}=2\frac{\Delta u_{{\rm LMPC},k}-\Delta u_{\min}}{\Delta u_{\max}-\Delta u_{\min}}-1.
$$

The actor BC loss is the supervised action mismatch:

$$
J_{\rm BC}=\| \pi_\theta(s_k)-a_{{\rm demo},k}\|^2.
$$

The critic replay transition uses the executed safe action instead:

$$
(s_k,a_{{\rm exec},k},r_k,s_{k+1},d_k).
$$

This separation is the core methodological change. The actor is never bypassed during BC. The teacher guides the actor through $J_{\rm BC}$, but the action tested by the gate is still the actor candidate.

### 8. Safety Gate, Executed Action, And Correction Gap

For each actor candidate, the gate predicts the first-step Lyapunov behavior. If the candidate satisfies contraction, then:

$$
\Delta u_{{\rm exec},k}=\Delta u_{{\rm rl},k}.
$$

If the candidate violates the contraction check, the direct LMPC fallback is executed:

$$
\Delta u_{{\rm exec},k}=\Delta u_{{\rm LMPC},k}.
$$

The safety-correction gap is:

$$
g_k=\Delta u_{{\rm cand},k}-\Delta u_{{\rm exec},k}.
$$

The infinity-norm gap logged in the report is:

$$
g_{\infty,k}=\max_j |g_{k,j}|.
$$

The fallback indicator is:

$$
I_{{\rm fb},k}=
1\quad\hbox{if}\quad \Delta u_{{\rm exec},k}\ne \Delta u_{{\rm cand},k}.
$$

This is why the report distinguishes good closed-loop output behavior from good actor authority. A controller may track well because the safety gate frequently corrects it. That is not the same as the actor itself satisfying the Lyapunov gate.

### 9. Soft Handoff Calculation

After the BC phase, the scripts use a five-episode linear handoff. Let $h$ be the number of elapsed handoff steps and $H$ be the total handoff length. The blending coefficient is:

$$
\alpha_h=\max(0,1-h/H).
$$

The pre-gate candidate during handoff is:

$$
\Delta u_{{\rm handoff},k}
=\alpha_h\Delta u_{{\rm LMPC},k}
+(1-\alpha_h)\Delta u_{{\rm rl},k}.
$$

At the start of handoff, $\alpha_h$ is near one, so the candidate is close to the teacher. At the end, $\alpha_h$ reaches zero, so the candidate is the pure actor action. The gate is still active during handoff.

### 10. Reward Calculation Used For Learning

The reward is computed after the plant step from the tracking error, input move, and safety correction. The output tracking error is:

$$
e_k=\Delta y_{k+1}-\Delta y_{{\rm sp},k}.
$$

The input move is:

$$
\Delta u_{{\rm move},k}=\Delta u_{{\rm exec},k}-\Delta u_{k-1}.
$$

The physical tolerance band for output $i$ is:

$$
b_i=\max(k_{{\rm rel},i}|y_{{\rm sp},i}^{\rm phys}|,b_{{\rm floor},i}).
$$

The scaled band is:

$$
\bar b_i=b_i/(y_{\max,i}-y_{\min,i}).
$$

The smooth inside-band score for each output is:

$$
s_i=\sigma\left(\frac{\bar b_i-|e_{k,i}|}{\tau_{\rm frac}\bar b_i}\right).
$$

With the current active `gate = "geom"`, the joint inside-band weight is the geometric mean of the per-output scores:

$$
w_{\rm in}=\left(\prod_{i=1}^{n_y}s_i\right)^{1/n_y}.
$$

The weighted quadratic error is:

$$
J_{\rm quad}=(1-w_{\rm in})\sum_i Q_i e_{k,i}^2
+w_{\rm in}\lambda_{\rm in}\sum_i Q_i e_{k,i}^2.
$$

The move cost is:

$$
J_{\rm move}=\sum_j R_j\Delta u_{{\rm move},k,j}^2.
$$

The outside-band overflow term is:

$$
J_{\rm out}=(1-w_{\rm in})\sum_i \gamma_{\rm out}(2Q_i\bar b_i)\max(|e_{k,i}|-\bar b_i,0).
$$

The inside-band residual term is:

$$
J_{\rm in}=w_{\rm in}\sum_i \gamma_{\rm in}(2Q_i\bar b_i)\min(|e_{k,i}|,\bar b_i).
$$

For the quadratic near-zero bonus:

$$
\phi_i=\left(1-\min(|e_{k,i}|/\bar b_i,1)\right)^2.
$$

$$
B_{\rm zero}=w_{\rm in}\beta\sum_i Q_i\bar b_i^2\phi_i.
$$

The base reward is:

$$
r_{\rm base}=-(J_{\rm quad}+J_{\rm move}+J_{\rm out}+J_{\rm in})+B_{\rm zero}.
$$

When fallback is active, the correction penalty is:

$$
J_{\rm fb}=\gamma_{\rm fb}\sum_j R_{{\rm fb},j}g_{k,j}^2+c_{\rm fb}I_{{\rm fb},k}.
$$

The complete reward is:

$$
r_k=r_{\rm base}-J_{\rm fb}-J_{\rm maint}-J_{\rm jitter}+B_{\rm dwell}.
$$

For the current analyzed scripts:

$$
J_{\rm maint}=0.
$$

$$
J_{\rm jitter}=0.
$$

The reason is intentional. We are currently testing high exploration and strict fallback penalties. Maintenance and jitter costs would also punish exploration-induced motion, so they are disabled for this diagnostic run.

### 11. TD3 Return And Update Timing

TD3 trains the critic toward a discounted return target:

$$
G_k=r_k+\gamma G_{k+1}.
$$

The active scripts now use:

$$
\gamma=0.995.
$$

This discount factor is only the RL return discount. It is separate from the Lyapunov contraction factor $\rho=0.99$.

During BC, the critic is updated from the executed safe action and the actor is additionally updated with BC loss. During full RL, TD3 actor and critic updates use the replay buffer in the standard way. The actor update is delayed by `POLICY_DELAY = 2`, and the target-policy smoothing standard deviation is `0.1` for cold start and `0.01` for pretrain.

### 12. Metrics And Calculations In The Report

The report separates four kinds of evidence:

Full-horizon RMSE for output $i$ is:

$$
{\rm RMSE}_i=\sqrt{\frac{1}{N}\sum_{k=1}^{N}(y_{k,i}^{\rm phys}-y_{{\rm sp},k,i}^{\rm phys})^2}.
$$

The mean RMSE reported in the summary plot is:

$$
{\rm RMSE}_{\rm mean}=\frac{{\rm RMSE}_{\eta}+{\rm RMSE}_{T}}{2}.
$$

The final-tail offset uses the last 100 samples of the final episode:

$$
{\rm tail\ offset}_i=\frac{1}{100}\sum_{k=N-99}^{N}|y_{k,i}^{\rm phys}-y_{{\rm sp},k,i}^{\rm phys}|.
$$

The fallback rate is:

$$
r_{\rm fb}=\frac{1}{N}\sum_{k=1}^{N}I_{{\rm fb},k}.
$$

The average correction gap is:

$$
\bar g_\infty=\frac{1}{N}\sum_{k=1}^{N}g_{\infty,k}.
$$

Wall-clock seconds per control step is:

$$
t_{\rm step}=t_{\rm total}/N.
$$

Steps per second is:

$$
{\rm SPS}=N/t_{\rm total}.
$$

This separation matters because a controller can have good raw tracking but poor safety-gate authority, or good final offset but poor transient behavior.

## Full-Horizon Results

![Performance and runtime](figures/2026-05-20_fixed_event_10_latest_analysis/performance_runtime_summary.png)

| Case | Reward mean | eta RMSE | T RMSE | Mean RMSE | ms per step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold RL | -14.493 | 0.215 | 0.470 | 0.343 | 14.13 |
| Cold MPC-only | -12.962 | 0.212 | 0.442 | 0.327 | 12.28 |
| Pretrained RL | -5.944 | 0.131 | 0.264 | 0.198 | 14.85 |
| Pretrained MPC-only | -3.457 | 0.123 | 0.226 | 0.175 | 12.28 |
| Direct LMPC | -4.331 | 0.191 | 0.565 | 0.378 | 32.06 |
| Direct MPC-only | -4.331 | 0.191 | 0.565 | 0.378 | 5.61 |

Main reading: increasing the fixed event penalty to `10.0` did not improve raw tracking. Pretrained RL remains the better learned controller on full-horizon reward and raw-setpoint RMSE, but it still does not beat its corresponding MPC-only diagnostic. Cold-start RL is still worse than cold MPC-only over the full horizon. The direct LMPC rows are unchanged from the latest available direct run.

The best raw-tracking result is still the pretrained-script MPC-only diagnostic. The larger fixed event penalty made fallback frequency slightly more expensive, but it did not turn the safety-gated RL policy into a controller that dominates the offset-free MPC baseline.

## Fixed Event Penalty Effect

The new run isolates one change: `fallback_event_penalty` increased from `2.0` to `10.0`, while `gamma_fallback = 3.0` and the remaining reward weights stayed fixed.

![Fixed event penalty effect](figures/2026-05-20_fixed_event_10_latest_analysis/fixed_event_penalty_effect.png)

| RL case | Event penalty | Fallback rate | Mean RMSE | Reward mean | Fixed event mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold RL | 2.0 | 1.25% | 0.332 | -13.891 | 0.027 |
| Cold RL | 10.0 | 1.17% | 0.343 | -14.493 | 0.126 |
| Pretrained RL | 2.0 | 3.34% | 0.196 | -5.809 | 0.068 |
| Pretrained RL | 10.0 | 3.18% | 0.198 | -5.944 | 0.327 |

This is a useful negative result. The higher fixed event penalty reduced fallback frequency slightly for both RL agents, but the reduction was small and came with slightly worse full-horizon RMSE and lower augmented reward. The policy did not learn a qualitatively more gate-compatible strategy from this scalar change alone.

## RL Authority And Learning Phases

Cold-start RL still uses the safety gate less often than pretrained RL. That part remains consistent. What the new run adds is that the higher event penalty trims fallback counts slightly without closing the performance gap.

![RL authority diagnostics](figures/2026-05-20_fixed_event_10_latest_analysis/rl_authority_diagnostics.png)

| RL case | Actual intervention rate | Fallback rate | Accepted rate | Penalty mean | Action gap mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold RL | 1.26% | 1.17% | 98.74% | 0.774 | 0.036 |
| Pretrained RL | 3.27% | 3.18% | 96.73% | 1.537 | 0.077 |

Pretrained RL still asks the gate to intervene more often and receives the larger fallback penalty, but it still tracks much better than cold start. The pretrained policy prior remains helpful for tracking, even though it is less naturally compatible with the Lyapunov gate.

![Episode reward and fallback trends](figures/2026-05-20_fixed_event_10_latest_analysis/rl_episode_reward_fallback_trends.png)

The phase breakdown still shows the same mechanism: cold-start BC is very costly under high exploration, then online learning recovers. The higher event penalty did not fix that early transient.

| Case | Phase | Mean reward | Fallback count | Intervention count | Mean episode RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Cold RL | BC 1-20 | -78.206 | 110 | 160 | 0.906 |
| Cold RL | Handoff 21-25 | -9.183 | 21 | 22 | 0.279 |
| Cold RL | Online 26-200 | -7.363 | 1740 | 1832 | 0.191 |
| Pretrained RL | BC 1-20 | -9.449 | 140 | 144 | 0.316 |
| Pretrained RL | Handoff 21-25 | -4.844 | 46 | 51 | 0.217 |
| Pretrained RL | Online 26-200 | -5.575 | 4901 | 5034 | 0.177 |

Cold start is more gate-compatible, but not better overall. Pretrained RL still has the better online RMSE, even after the higher event penalty.

![Activation and contraction counts](figures/2026-05-20_fixed_event_10_latest_analysis/activation_contraction_episode_counts.png)

## Tail Offset

The final 100 steps of the final episode are used as a compact steady-offset check. Lower values mean closer approach to the final setpoint.

![Tail offset comparison](figures/2026-05-20_fixed_event_10_latest_analysis/tail_offset_comparison.png)

| Case | Tail eta abs mean | Tail T abs mean | Final eta abs | Final T abs |
| --- | ---: | ---: | ---: | ---: |
| Cold RL | 0.0032 | 0.0092 | 0.0032 | 0.0092 |
| Cold MPC-only | 0.0113 | 0.0297 | 0.0113 | 0.0297 |
| Pretrained RL | 0.0084 | 0.0446 | 0.0012 | 0.0006 |
| Pretrained MPC-only | 0.0093 | 0.0079 | 0.0093 | 0.0079 |
| Direct LMPC | 0.0030 | 0.0164 | 0.0015 | 0.0082 |
| Direct MPC-only | 0.0030 | 0.0164 | 0.0015 | 0.0082 |

Pretrained RL has a nearly zero final sample error, but its 100-step tail temperature error remains high. This means it reaches the final setpoint late or oscillates inside the tail window. Cold-start RL has better tail eta offset, but not better full-horizon tracking. The reward still does not align full-horizon tracking, final sample offset, and tail-window offset perfectly.

![Last episode tracking](figures/2026-05-20_fixed_event_10_latest_analysis/last_episode_tracking_primary_methods.png)

## MPC-only Would-Be Gate Activation

For MPC-only cases, actual fallback is zero by construction. The useful diagnostic is therefore how often the Lyapunov contraction condition would have failed if the gate had been active.

![MPC-only would-be activation](figures/2026-05-20_fixed_event_10_latest_analysis/mpc_only_would_be_activation.png)

| MPC-only case | Would-be activation rate | Actual fallback rate |
| --- | ---: | ---: |
| Cold MPC-only | 10.44% | 0.00% |
| Pretrained MPC-only | 27.84% | 0.00% |
| Direct MPC-only | 2.75% | not used |

The pretrained-script MPC-only diagnostic still has the best raw tracking and the highest would-be gate activation rate. This remains the cleanest evidence of the target-selector tension: good raw tracking can violate the current Lyapunov contraction test more often.

## Runtime Claim

Wall-clock timing still supports the claim that safety-gated RL is faster than direct LMPC. Cold RL and pretrained RL run at about 14.1 ms and 14.8 ms per control step. Direct LMPC, from the latest available direct run, runs at about 32.1 ms per step. Thus the RL safety-gate cases are about 2.2 times faster than direct LMPC in this comparison.

The direct MPC-only diagnostic is still much faster at about 5.6 ms per step. The RL-script MPC-only diagnostics are around 12.3 ms per step because they carry diagnostic and study-loop overhead. The fair claim is still not "RL is fastest"; it is "safety-gated RL is faster than direct LMPC, while plain MPC-only remains faster than both."

## Reward Function Diagnosis

With `fallback_event_penalty = 10.0`, the fixed event term is now clearly visible. The correction-gap term still dominates, especially for pretrained RL, but event frequency is no longer negligible.

![Reward penalty scale](figures/2026-05-20_fixed_event_10_latest_analysis/reward_penalty_scale.png)

| RL case | Reward mean | Base mean | Fallback penalty mean | Fixed event mean | Correction mean | Penalty share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold RL | -14.493 | -13.719 | 0.774 | 0.126 | 0.648 | 5.6% |
| Pretrained RL | -5.944 | -4.407 | 1.537 | 0.327 | 1.210 | 34.9% |

For pretrained RL, the fallback penalty is about one third of the base reward magnitude. That is large enough to be learned from. The fact that fallback dependence only dropped modestly suggests the remaining issue is not only penalty visibility. The policy may need a different learning signal, curriculum, target definition, or action parameterization to become gate-compatible without losing tracking.

![Reward base and fallback trends](figures/2026-05-20_fixed_event_10_latest_analysis/reward_base_fallback_episode_trends.png)

The current analyzed RL reward setup is:

```text
Qy_diag = [8.0, 6.0]
Rdu_diag = [1.0, 1.0]
k_rel = [0.0015, 0.00015]
band_floor_phys = [0.003, 0.035]
tau_frac = 0.5
gamma_out = 1.0
gamma_in = 3.0
gate = "geom"
lam_in = 3.0
bonus_kind = "quadratic"
beta = 1.0
gamma_fallback = 3.0
fallback_event_penalty = 10.0
maintenance_band_scale = 0.5
maintenance_move_weight = 0.0
jitter_weight = 0.0
dwell_bonus = 0.0
TD3 GAMMA = 0.995
```

## Target Diagnostics

The target diagnostics still point to the same unresolved issue: the direct target can be acceptable to the Lyapunov machinery while still being a poor raw-setpoint tracking center.

![Target diagnostics](figures/2026-05-20_fixed_event_10_latest_analysis/target_diagnostics_summary.png)

| Case | Target residual max | Mean input-target gap | Mean state-target gap | Target rate max |
| --- | ---: | ---: | ---: | ---: |
| Cold RL | 16.56 | 0.501 | 0.478 | 5.39 |
| Cold MPC-only | 15.79 | 10.486 | 1.444 | 6.17 |
| Pretrained RL | 12.15 | 0.856 | 0.307 | 5.21 |
| Pretrained MPC-only | 10.62 | 7.672 | 0.800 | 6.17 |
| Direct LMPC | 6.26 | 0.901 | 0.018 | 2.40 |
| Direct MPC-only | 6.10 | 8.065 | 0.089 | 6.17 |

The direct no-RL cases have lower target residuals than the RL-script cases, but their raw-setpoint RMSE is still high. The RL-script MPC-only rows can track better while showing much higher would-be Lyapunov activation. This is exactly the target-selector tension: a target can be easier for Lyapunov contraction but worse for raw tracking, or better for raw tracking but less compatible with the contraction test.

## Updated Findings

- Pretrained RL is now the best learned controller for full-horizon reward and RMSE.
- Cold-start RL is not better overall. It is better mainly in safety-gate authority and tail eta offset.
- The cold-start BC phase is the main damage source: high exploration with an untrained actor creates very poor early episodes before online training recovers.
- MPC-only still beats the corresponding safety-gated RL case in both cold and pretrained studies on full-horizon reward and RMSE.
- Raising the fixed fallback event penalty from `2.0` to `10.0` slightly reduced fallback rates, but it did not improve tracking and it worsened the augmented reward.
- The higher fixed event penalty is meaningful, especially for pretrained RL, but it still did not eliminate gate dependence.
- Direct LMPC remains slower than safety-gated RL and still has poor full-horizon raw-setpoint RMSE.
- The target-selector problem remains central: better Lyapunov compatibility and better raw-setpoint tracking are still pulling in different directions.

## Project History And Already-Tried Changes

This final section is the project-level handoff note. It is intentionally broader than the latest run so a future reader, including ChatGPT, does not suggest ideas that have already been implemented, tested, or deliberately rejected.

### Active Entrypoints And Current Scope

- The active root scripts are `DirectLyapunovMPC.py`, `DirectLyapunovSafetyGateRL_ColdStart.py`, and `DirectLyapunovSafetyGateRL_Pretrained.py`.
- Root notebooks were converted or archived. New work should edit Python scripts unless a notebook is explicitly requested.
- Top-level result folders are now `results/directLyap/...`, `results/ColdStart/...`, and `results/Pretrain/...`.
- The current direct/RL disturbance study uses two setpoints, `n_episodes = 200`, `set_points_len = 400`, `rho_lyap = 0.99`, and `lyap_eps = 1e-3`.
- All three active scripts keep `u_prev_penalty_weight = 0.1`, `xs_prev_penalty_weight = 0.1`, and `case_variants = ("mixed",)`.
- The active direct/RL direct-tracking calls use `use_target_output_for_tracking = False`, so the online tracking objective follows raw $y_{\rm sp}$ while the direct target still supplies the Lyapunov center.

The active direct LMPC script keeps this direct-study reward/target shaping configuration:

```text
Qy_diag = [5.0, 1.0]
Rdu_diag = [1.0, 1.0]
k_rel = [0.003, 0.0003]
band_floor_phys = [0.006, 0.07]
gamma_out = 0.5
gamma_in = 0.5
gate = "geom"
lam_in = 1.0
bonus_kind = "exp"
beta = 7.0
```

### Target Selector Work Already Done

- A four-mode target-selector API was implemented first with `current_exact_fallback_frozen_d`, `free_disturbance_prior`, `compromised_reference`, and `single_stage_robust_sstp`.
- That four-mode surface was later narrowed because it added complexity without solving the centering problem. The active standard selector is now the single refined Step A selector.
- The old selector-mode strings are kept only as compatibility inputs. They resolve to the refined Step A implementation rather than representing active competing methods.
- The refined Step A selector fixes $d_s=\hat d_k$ and solves for a steady package using output tracking, input anchoring, previous-input smoothing, previous-state smoothing, and a weak current-state anchor.
- The refined selector objective is conceptually:

$$
\begin{aligned}
J_{\rm sel}={}&
\|r_s-y_{\rm sp}\|_{Q_r}^2
+\alpha_u\|u_s-u_{k-1}\|_{R_u}^2 \\
&+\alpha_{\Delta u}\|u_s-u_{s,k-1}\|_{R_{\Delta u}}^2
+\alpha_{\Delta x}\|x_s-x_{s,k-1}\|_{Q_{\Delta x}}^2
+\alpha_x\|x_s-\hat x_k\|_{Q_x}^2 .
\end{aligned}
$$

- The refined Step A defaults already tried include `alpha_u_ref = 0.5`, `alpha_du_sel = 0.5`, `alpha_dx_sel = 0.05`, `alpha_x_ref = 0.01`, `x_weight_base = "CtQC"`, and `use_output_bounds_in_selector = True`.
- Selector warm-start was added and the runners pass previous target values into CVXPY when enabled.
- Effective-target backup was added: if the current selector fails, the safety path can use the last valid target instead of treating the target as unavailable.
- Debug exports already include selector objective terms, target residuals, current/effective target distinction, whether the effective target was reused, and target margins.

### Safety Filter And First-Step Contraction Work Already Done

- Lyapunov tolerance semantics were standardized everywhere as:

$$
V(x_{k+1}) \le \rho V(x_k)+\epsilon .
$$

- The hard safety-filter acceptance path was clarified. Slack-enabled solves can still be rejected by the hard post-check, and debug data separates attempted, solved, and hard-accepted stages.
- Trust-region enablement and trust-region slack were separated as different knobs.
- A first-step-contraction upstream-MPC experiment was built around the baseline offset-free MPC objective with one hard first-step Lyapunov inequality.
- That upstream first-step experiment intentionally has no QCQP projection stage. If the constrained MPC solve fails, it falls back to ordinary offset-free MPC and logs whether the fallback would have satisfied contraction.
- Earlier first-step/bounded-frozen target work added bounded steady-state analysis, `u_prev` anchoring, and `x_s` smoothing. Those ideas informed the current direct bounded target variants.

### Direct Output-Disturbance Lyapunov MPC Work Already Done

- A direct frozen-output-disturbance target path was implemented in `Lyapunov/frozen_output_disturbance_target.py`.
- The direct target freezes $d_s=\hat d_k$ and solves for $x_s,u_s$ using the output-disturbance model.
- Both unbounded and bounded direct targets were implemented. Bounded targets can use regularization toward $u_{k-1}$ and the previous $x_s$.
- The direct Lyapunov MPC path was implemented in `Lyapunov/direct_lyapunov_mpc.py` with one target solve and one direct Lyapunov MPC solve per online step.
- Hard and soft Lyapunov modes were implemented. Soft mode uses nonnegative Lyapunov slack with a large slack penalty, but the current active disturbance scripts use bounded hard mixed cases.
- Four-method and expanded scenario studies were already tried: unbounded hard, bounded hard, unbounded soft, bounded soft, bounded hard with `u_prev`, bounded hard with `x_s` smoothing, and mixed anchoring/smoothing.
- The direct objective was deliberately simplified. Extra steady-input objective terms and terminal objective terms were removed from the direct path. The active direct objective is output tracking plus input-move penalty, with Lyapunov enforced through the contraction constraint and related checks.
- The direct setpoint schedule length has been changed during the project. The current active setting is back to `set_points_len = 400`.
- Target-output tracking was tried in the RL direct-gate path. It produced poor training behavior, so the active RL direct tracking was reverted to raw $y_{\rm sp}$ with `direct_tracking_use_target_output = False`.

### Target-Quality, Lexicographic, And Guard Ideas Already Implemented

- Target-quality diagnostics and bypass fields were added to the direct path.
- Lexicographic bounded target support was implemented as an option for bounded steady-state target solves.
- Disturbance-model routing was added for output-disturbance and generic augmented selector paths.
- Direct-gate performance-guard hooks were added in the RL runner.
- Residual-RL hooks were added so an actor can represent a bounded residual around a baseline policy.
- These guard/residual/lexicographic tools are available, but the active scripts keep the current mixed bounded direct setup unless those options are explicitly turned on.

### RL Safety-Gate Work Already Done

- The direct RL safety gate uses `projection_backend = "direct_accept_or_fallback"`.
- The RL actor proposes a candidate action. The gate accepts it only if it satisfies the direct Lyapunov contraction check; otherwise the direct LMPC fallback is executed.
- Replay stores the executed safe action, not an unsafe raw proposal.
- The direct LMPC fallback/teacher action is computed with the same direct output-disturbance target family as the direct no-RL script.
- MPC-only cases execute offset-free MPC and keep actual fallback zero. Their fallback-count plot uses diagnostic would-be gate activation, not actual fallback zero, unless explicitly labeled actual fallback.
- Phase-aware warm-start, teacher BC, parameter-noise exploration, Gaussian exploration, and executed-action BC were all tried earlier.
- The current design is agent-authority BC: the actor proposes the candidate during BC, the safety gate decides the executed action, and direct LMPC is stored only as the imitation target for actor BC loss.
- The current BC setup uses `WARMUP_EPISODES = 0`, `BC_TEACHER_EPISODES = 20`, `bc_actor_updates_per_step = 4`, and a 5-episode linear handoff after BC.
- Recommending direct teacher execution during BC would undo the current design. The current hypothesis is that the actor needs authority from the beginning, protected by the gate.

### Reward Changes Already Tried

The reward function now supports a fixed fallback event cost:

$$
J_{\rm fb}=\gamma_{\rm fb}\sum_j R_{{\rm fb},j}g_j^2+c_{\rm fb}I_{\rm fb}.
$$

The earlier strict-offset candidate used:

```text
Qy_diag = [8.0, 4.0]
Rdu_diag = [1.0, 1.0]
k_rel = [0.0015, 0.00015]
band_floor_phys = [0.003, 0.035]
gamma_in = 2.0
lam_in = 2.0
beta = 2.0
gamma_fallback = 2.0
fallback_event_penalty = 0.5
maintenance_move_weight = 0.1
jitter_weight = 0.02
```

That run showed the fixed event penalty was too small after averaging over time: only about `0.0067` reward units per step for cold RL and `0.0143` for pretrained RL.

The next strict follow-up used:

```text
Qy_diag = [8.0, 6.0]
Rdu_diag = [1.0, 1.0]
k_rel = [0.0015, 0.00015]
band_floor_phys = [0.003, 0.035]
tau_frac = 0.5
gamma_out = 1.0
gamma_in = 3.0
gate = "geom"
lam_in = 3.0
bonus_kind = "quadratic"
beta = 1.0
gamma_fallback = 3.0
fallback_event_penalty = 2.0
maintenance_band_scale = 0.5
maintenance_move_weight = 0.0
jitter_weight = 0.0
dwell_bonus = 0.0
TD3 GAMMA = 0.995
```

That run made the fallback penalty visible, especially for pretrained RL, but gate dependence remained. The fixed event cost was then increased again, producing the latest analyzed run in this report:

```text
fallback_event_penalty = 10.0
```

All other reward weights above remained unchanged. This isolated the effect of fixed event frequency cost from the correction-gap multiplier.

Do not suggest only "increase fallback penalty from 0.5" or "increase fixed event penalty from 2.0 to 10.0" as a new idea. Both have already been implemented. The latest data show `10.0` slightly reduces fallback frequency, but it does not improve raw tracking and it worsens the augmented reward.

### Exploration, Policy Noise, And Discount Already Changed

- Cold start now uses BC exploration `0.2`, full-RL exploration decaying linearly from `0.2` to `0.01`, and TD3 target policy smoothing noise `0.1`.
- Pretrained now uses BC exploration `0.02`, full-RL exploration decaying linearly from `0.02` to `0.01`, and TD3 target policy smoothing noise `0.01`.
- BC exploration is active through `bc_behavior_noise = "gaussian"`. It does not use a smaller special BC noise floor.
- The TD3 discount factor is now `GAMMA = 0.995`. This is different from the Lyapunov contraction factor, which remains `rho_lyap = 0.99`.
- Maintenance and jitter weights are currently zero on purpose because exploration is active. Reintroducing them should be a later low-exploration polishing test, not the next diagnostic run.

### Lyapunov Epsilon Interpretation

The value `lyap_eps = 1e-3` has already been tried and improved the runs. It does not make the controller MPC-only by itself. The safety gate still evaluates:

$$
V(x_{k+1}) \le \rho V(x_k) + \epsilon,
$$

with $\rho = 0.99$ and $\epsilon = 10^{-3}$. A larger $\epsilon$ relaxes strict contraction, especially near small $V(x_k)$, but it does not remove the gate, fallback path, direct LMPC teacher, or activation diagnostics. If future results look too MPC-like, the right check is activation, fallback, contraction-margin, and correction-gap logs, not epsilon alone.

### Diagnostics And Report Infrastructure Already Added

- Wall-clock timing is saved for total seconds, seconds per episode, seconds per control step, and steps per second.
- Activation/contraction diagnostics include raw per-episode counts and a moving 10-episode average.
- Result bundles include training-phase metadata, reward parameters, timing fields, and trained-agent paths when saving is enabled.
- Safety/debug exports include target diagnostics, correction modes, fallback counts, reward components, episode tables, step tables, NPZ arrays, and comparison plots.
- The report has both Markdown and a self-contained HTML export with embedded figures for sharing.

### Results Already Observed

- In the latest analyzed run, pretrained RL had better full-horizon reward and RMSE than cold-start RL.
- Cold-start RL was not better overall. It was better mainly in safety-gate authority and final-tail eta offset.
- Cold-start BC remained fragile because high exploration with an initially untrained actor created very poor early episodes before online learning recovered.
- The corresponding MPC-only diagnostic still beat each safety-gated RL case on full-horizon reward and RMSE.
- Increasing `fallback_event_penalty` from `2.0` to `10.0` slightly reduced fallback rates but did not improve tracking.
- The stronger fallback penalty is now meaningful, but it did not eliminate pretrained gate dependence.
- Direct LMPC had reasonable final-tail offset, but worse full-horizon raw-setpoint RMSE and slower runtime than safety-gated RL.
- The RL safety-gate cases were about `2.2x` faster than direct LMPC in seconds per control step in the latest comparison.
- Direct MPC-only was much faster, but it is a diagnostic baseline, not the same controller as direct LMPC or safety-gated RL.
- Earlier and current direct-target diagnosis showed a key failure mode: the Lyapunov controller can contract around a poor or modified admissible target while raw-setpoint tracking looks worse than MPC-only.

### Do Not Re-Suggest Without New Evidence

- Do not re-suggest the old four-mode selector as the next fix; it was already implemented and then collapsed to refined Step A.
- Do not re-suggest selector warm-start, last-valid target backup, or logging current versus effective targets; those are already implemented.
- Do not re-suggest a generic first-step-contraction upstream MPC as a new architecture; it was already built without QCQP projection.
- Do not re-suggest adding `u_prev` or `x_s` target regularization as a new concept; both have already been implemented and the current active mixed case uses both at weight `0.1`.
- Do not re-suggest adding target-quality diagnostics, lexicographic bounded targets, residual-RL hooks, or a performance guard as if they are missing; these hooks exist and only need to be deliberately enabled if selected.
- Do not re-suggest converting active notebooks to scripts, cleaning root entrypoints, saving trained agents, or adding wall-clock timing; those are already done.
- Do not re-suggest plotting MPC-only fallback as zero; the report now uses would-be gate activation for that diagnostic.
- Do not re-suggest making BC execute the LMPC teacher directly; the current design intentionally keeps the RL actor in authority.
- Do not re-suggest pretrained exploration `0.02 -> 0.01` or cold-start exploration `0.2 -> 0.01`; these are already active, including during BC.
- Do not re-suggest increasing fixed fallback event penalty from `2.0` to `10.0`; that exact experiment is now analyzed in this report.
- Do not re-suggest maintenance or jitter penalties during the current high-exploration diagnostic run; they are intentionally disabled for now.
- Do not treat `lyap_eps = 1e-3` as proof that the method is MPC-only. It is a relaxed Lyapunov gate and should be judged together with activation, fallback, and correction-gap logs.
