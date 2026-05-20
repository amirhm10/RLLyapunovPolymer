# Agent-Authority BC Latest Run Analysis

## Objective

This report analyzes the latest three reruns after the agent-authority behavioral cloning update: cold-start safety-gate RL, pretrained safety-gate RL, and direct Lyapunov MPC. The matched MPC-only diagnostics are kept visible because they explain how often the Lyapunov gate would have activated without intervention.

Analyzed folders:

- Cold start: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260519_212733`
- Pretrained: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260519_212739`
- Direct LMPC: `results/direct_lyap_ch2_lex/20260519_212725`

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

For the current next-run scripts:

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

![Performance and runtime](figures/2026-05-19_agent_authority_bc_latest_analysis/performance_runtime_summary.png)

| Case | Reward mean | eta RMSE | T RMSE | Mean RMSE | ms per step |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cold RL | -6.791 | 0.130 | 0.297 | 0.214 | 14.31 |
| Cold MPC-only | -5.951 | 0.126 | 0.291 | 0.208 | 12.10 |
| Pretrained RL | -4.498 | 0.127 | 0.265 | 0.196 | 14.75 |
| Pretrained MPC-only | -3.445 | 0.124 | 0.273 | 0.198 | 11.99 |
| Direct LMPC | -4.331 | 0.191 | 0.565 | 0.378 | 26.78 |
| Direct MPC-only | -4.331 | 0.191 | 0.565 | 0.378 | 4.90 |

Main reading: pretrained RL is still the best RL controller on reward and tracking error, but cold-start RL is more compatible with the safety gate. Direct LMPC is slower than the safety-gate RL runs and has larger full-horizon raw-output RMSE because the direct target path can track a modified admissible target rather than matching the raw setpoint as tightly. The direct MPC-only diagnostic is faster, but it is not the same controller as safety-gated direct LMPC.

## Why Cold Start Looks Better

Cold start is doing better on safety-gate authority, not on full-horizon raw tracking. It has fewer actual interventions, smaller fallback penalty, and smaller actor-versus-executed action gap. Pretrained RL has better tracking and reward, but it asks the gate to correct the policy more often.

![RL authority diagnostics](figures/2026-05-19_agent_authority_bc_latest_analysis/rl_authority_diagnostics.png)

| RL case | Actual intervention rate | Fallback rate | Penalty mean | Action gap mean |
| --- | ---: | ---: | ---: | ---: |
| Cold RL | 1.35% | 1.26% | 0.413 | 0.036 |
| Pretrained RL | 2.86% | 2.76% | 0.674 | 0.062 |

The likely reason is the pretrained checkpoint was learned under an older objective and without this exact safety-gate authority structure. It starts with a stronger tracking prior, but that prior is also more assertive and mismatched to the new strict fallback penalty. Cold start begins less biased toward the old policy, so the BC-plus-gate process shapes it into actions that the gate accepts more easily. This is why cold start can look better in activation and fallback plots even though pretrained RL still wins on reward and RMSE.

![Episode reward and fallback trends](figures/2026-05-19_agent_authority_bc_latest_analysis/rl_episode_reward_fallback_trends.png)

Phase averages confirm the same pattern:

| Case | Phase | Mean reward | Fallback count | Mean episode RMSE |
| --- | --- | ---: | ---: | ---: |
| Cold RL | BC 1-20 | -7.236 | 96 | 0.297 |
| Cold RL | Handoff 21-25 | -8.112 | 29 | 0.296 |
| Cold RL | Online 26-200 | -6.703 | 2034 | 0.195 |
| Pretrained RL | BC 1-20 | -6.220 | 51 | 0.289 |
| Pretrained RL | Handoff 21-25 | -5.199 | 47 | 0.249 |
| Pretrained RL | Online 26-200 | -4.282 | 4472 | 0.179 |

## Tail Offset

The final 100 steps of the final episode are used as a compact steady-offset check. Lower values mean closer approach to the final setpoint.

![Tail offset comparison](figures/2026-05-19_agent_authority_bc_latest_analysis/tail_offset_comparison.png)

| Case | Tail eta abs mean | Tail T abs mean | Final eta abs | Final T abs |
| --- | ---: | ---: | ---: | ---: |
| Cold RL | 0.0116 | 0.0121 | 0.0116 | 0.0121 |
| Cold MPC-only | 0.0124 | 0.0766 | 0.0124 | 0.0766 |
| Pretrained RL | 0.0175 | 0.0579 | 0.0081 | 0.0165 |
| Pretrained MPC-only | 0.0046 | 0.0986 | 0.0046 | 0.0986 |
| Direct LMPC | 0.0030 | 0.0164 | 0.0015 | 0.0082 |
| Direct MPC-only | 0.0030 | 0.0164 | 0.0015 | 0.0082 |

Cold-start RL has the smallest final 100-step temperature offset among the two RL agents in this run. Pretrained RL has better full-horizon reward and RMSE, but it leaves more tail temperature offset. Direct LMPC has good final-tail offset even though its full-horizon RMSE is large, which means its main error is earlier transient or modified-target behavior rather than final steady offset.

![Last episode tracking](figures/2026-05-19_agent_authority_bc_latest_analysis/last_episode_tracking_primary_methods.png)

## MPC-only Would-Be Gate Activation

For MPC-only cases, actual fallback is zero by construction. The useful diagnostic is therefore how often the Lyapunov contraction condition would have failed if the gate had been active.

![MPC-only would-be activation](figures/2026-05-19_agent_authority_bc_latest_analysis/mpc_only_would_be_activation.png)

| MPC-only case | Would-be activation rate | Actual fallback rate |
| --- | ---: | ---: |
| Cold MPC-only | 11.03% | 0.00% |
| Pretrained MPC-only | 26.31% | 0.00% |
| Direct MPC-only | 2.75% | not used |

The two RL-script MPC-only diagnostics differ because they are coupled to different learned-agent candidates in the diagnostic comparison path. The direct script MPC-only case is a cleaner no-RL offset-free MPC diagnostic and has a much lower would-be activation rate.

## Runtime Claim

Wall-clock timing now supports the speed claim. The RL script cases run at roughly 68 to 83 steps per second, while the direct LMPC case runs at about 37 steps per second and the direct MPC-only diagnostic runs at about 204 steps per second. The meaningful comparison for safety-gated RL versus direct LMPC is seconds per step: pretrained RL is about 0.0148 s per step, cold RL is about 0.0143 s per step, and direct LMPC is about 0.0268 s per step. Thus the RL safety-gate runs are about 1.8 times faster than direct LMPC in this run while retaining better raw-output RMSE.

## Reward Function Diagnosis

The latest results suggest the reward is still not strict enough about safety-gate dependence and final offset. The important observation is not only the absolute fallback penalty, but its scale compared with the base tracking reward. In the current runs, the fixed fallback event penalty is almost negligible on an average-per-step basis because it is multiplied only by the fallback rate.

![Reward penalty scale](figures/2026-05-19_agent_authority_bc_latest_analysis/reward_penalty_scale.png)

| RL case | Reward mean | Base mean | Fallback penalty mean | Fixed event mean | Correction mean | Penalty share of base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold RL | -6.791 | -6.364 | 0.413 | 0.0067 | 0.406 | 6.5% |
| Pretrained RL | -4.498 | -3.811 | 0.674 | 0.0143 | 0.660 | 17.7% |


The fixed event component is only about 0.006 per step for cold RL and 0.014 per step for pretrained RL. So even though `fallback_event_penalty = 0.5` sounds noticeable, it is too small after averaging over all time steps. The penalty signal mostly comes from the correction gap term, which punishes large corrections but does not strongly punish frequent small fallback events.

![Reward base and fallback trends](figures/2026-05-19_agent_authority_bc_latest_analysis/reward_base_fallback_episode_trends.png)

### Current Formula

At each step, the reward function receives the scaled output error $e_k$, scaled input move $\Delta u_k$, physical setpoint $y_{\rm sp,k}$, and optionally the safety correction gap $g_k=u_{\rm cand,k}-u_{\rm exec,k}$.

The physical tolerance band is:

$$
b_i = \max(k_{{\rm rel},i}|y_{{\rm sp},i}|, b_{{\rm floor},i}).
$$

The band is converted to scaled units using the output scaling range $d_i$:

$$
\bar b_i = b_i / d_i.
$$

The smooth inside-band gate is:

$$
s_i = \sigma\left(\frac{\bar b_i-|e_i|}{\tau_{\rm frac}\bar b_i}\right).
$$

With the current active `gate = "geom"`, the joint near-setpoint weight is the geometric mean of the per-output scores:

$$
w_{\rm in}=\left(\prod_{i=1}^{n_y}s_i\right)^{1/n_y}.
$$

The quadratic tracking term changes weight inside the band:

$$
J_{\rm quad}=(1-w_{\rm in})\sum_i Q_i e_i^2+w_{\rm in}\lambda_{\rm in}\sum_i Q_i e_i^2.
$$

The move penalty is:

$$
J_{\Delta u}=\sum_j R_j(\Delta u_j)^2.
$$

Outside the band, the reward adds a linear overflow penalty:

$$
J_{\rm out}=(1-w_{\rm in})\sum_i \gamma_{\rm out}(2Q_i\bar b_i)\max(|e_i|-\bar b_i,0).
$$

Inside the band, it still penalizes residual error:

$$
J_{\rm in}=w_{\rm in}\sum_i \gamma_{\rm in}(2Q_i\bar b_i)\min(|e_i|,\bar b_i).
$$

The near-setpoint bonus is:

$$
B=w_{\rm in}\beta\sum_i Q_i\bar b_i^2\phi(z_i),\qquad z_i=|e_i|/\bar b_i.
$$

For `bonus_kind = "quadratic"`:

$$
\phi(z_i)=(1-z_i)^2.
$$

The base reward is:

$$
r_{\rm base}=-(J_{\rm quad}+J_{\Delta u}+J_{\rm out}+J_{\rm in})+B.
$$

When the safety gate changes the action, the fallback penalty is:

$$
J_{\rm fb}=\gamma_{\rm fb}\sum_j R_{{\rm fb},j}g_j^2+c_{\rm fb}I_{\rm fb}.
$$

The maintenance and jitter terms are:

$$
J_{\rm maint}=I_{\rm maint}w_{\rm maint}\sum_j R_j(\Delta u_j)^2,
$$

$$
J_{\rm jitter}=w_{\rm jitter}\sum_i Q_i(e_i-e_{i,k-1})^2.
$$

The final reward is:

$$
r_k=r_{\rm base}-J_{\rm fb}-J_{\rm maint}-J_{\rm jitter}+B_{\rm dwell}.
$$

### Parameters Used In This Analyzed Run

| Parameter | Current value | Role |
| --- | ---: | --- |
| `Qy_diag` | `[8.0, 4.0]` | Output error weights |
| `Rdu_diag` | `[1.0, 1.0]` | Input move weights |
| `k_rel` | `[0.0015, 0.00015]` | Relative output bands |
| `band_floor_phys` | `[0.003, 0.035]` | Minimum physical bands |
| `tau_frac` | `0.5` | Smoothness of inside-band gate |
| `gate` | `prod` | Joint inside-band gate for the analyzed run |
| `gamma_out` | `1.0` | Overflow penalty outside band |
| `gamma_in` | `2.0` | Residual penalty inside band |
| `beta` | `2.0` | Near-zero bonus scale |
| `lam_in` | `2.0` | Quadratic error multiplier inside band |
| `gamma_fallback` | `2.0` | Correction-gap penalty multiplier |
| `fallback_event_penalty` | `0.5` | Fixed cost per fallback event |
| `maintenance_move_weight` | `0.1` | Move suppression inside maintenance band |
| `jitter_weight` | `0.02` | Output jitter penalty |

### Implemented Next-Run Reward Setup

The next reward makes fallback events more visible and makes temperature offset more expensive near steady state. The implemented diagnostic setup also removes maintenance and jitter penalties while exploration remains intentionally active:

| Parameter | Current | Proposed | Reason |
| --- | ---: | ---: | --- |
| `Qy_diag` | `[8.0, 4.0]` | `[8.0, 6.0]` | Increase temperature importance without exploding eta weight |
| `gamma_in` | `2.0` | `3.0` | Penalize residual in-band offset more strongly |
| `gate` | `prod` | `geom` | Return to the earlier geometric gate instead of the stricter product gate |
| `lam_in` | `2.0` | `3.0` | Make near-band quadratic error less forgiving |
| `beta` | `2.0` | `1.0` | Reduce the chance that bonus hides small steady offset |
| `gamma_fallback` | `2.0` | `3.0` | Increase correction-gap cost |
| `fallback_event_penalty` | `0.5` | `2.0` | Make frequent small fallbacks visible to the average reward |
| `maintenance_move_weight` | `0.1` | `0.0` | Disable near-setpoint move penalty during high-exploration diagnosis |
| `jitter_weight` | `0.02` | `0.0` | Avoid penalizing exploration-induced output movement during this run |
| TD3 `GAMMA` | `0.99` | `0.995` | Slightly increase long-horizon credit assignment |

This setup is stricter on fallback and in-band offset, but deliberately disables maintenance and jitter while exploration is high. If fallback count decreases but tracking worsens, the event penalty is too high. If fallback count stays high, the actor still cannot find gate-compatible actions and the issue is policy adaptation rather than reward scale. The Lyapunov contraction factor remains $\rho=0.99$; only the TD3 discount factor is changed to `0.995`.

### Why Not Only Increase Fallback Penalty

Increasing only `fallback_event_penalty` would train the actor to avoid gate activation, but it would not necessarily close the steady-state offset. The current data show cold start already has lower fallback dependence while pretrained has better full-horizon tracking. Therefore the reward needs both a stronger fallback event cost and stronger in-band offset terms, especially for temperature.


## Conclusions

- Pretrained RL is the best RL case for full-horizon tracking and reward in this latest run.
- Cold-start RL is better for safety-gate authority: fewer fallbacks, smaller fallback penalty, and smaller correction gap.
- Cold-start RL also gives the smallest final-tail temperature offset among the two RL agents, even though pretrained RL is better over the full horizon.
- The new BC formulation is working as intended because the actor remains the candidate policy and the safety gate records meaningful authority/fallback differences between cold and pretrained agents.
- Direct LMPC is slower than the RL safety-gate cases and has higher full-horizon raw-setpoint RMSE here, mostly because its direct target construction allows significant modified-target behavior.
- The next useful experiment is to reduce pretrained policy mismatch rather than discard pretraining: use a short adapter BC phase, lower initial pretrained actor authority, or preload only lower layers while letting the output head adapt to the strict fallback reward.

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

The latest analyzed run in this report used the earlier strict-offset candidate:

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

The current next-run RL reward defaults already implement the stricter follow-up:

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
```

Do not suggest only "increase fallback penalty from 0.5" as a new idea. That has already been implemented through a larger correction-gap multiplier and a larger fixed event cost. The useful next check is whether the new `gamma_fallback = 3.0` and `fallback_event_penalty = 2.0` reduce activation without damaging tracking.

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
- Cold-start RL had better safety-gate authority: fewer fallback activations, smaller correction gaps, and smaller fallback penalty.
- Cold-start RL also had the smaller final-tail temperature offset among the two RL agents.
- Direct LMPC had good final-tail offset, but worse full-horizon raw-setpoint RMSE and slower runtime than safety-gated RL.
- The RL safety-gate cases were about `1.8x` faster than direct LMPC in seconds per control step in the latest timed run.
- Direct MPC-only was much faster, but it is a diagnostic baseline, not the same controller as direct LMPC or safety-gated RL.
- Earlier direct-target diagnosis showed a key failure mode: the Lyapunov controller can contract around a poor or modified admissible target while raw-setpoint tracking looks worse than MPC-only.

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
- Do not re-suggest maintenance or jitter penalties during the current high-exploration diagnostic run; they are intentionally disabled for now.
- Do not treat `lyap_eps = 1e-3` as proof that the method is MPC-only. It is a relaxed Lyapunov gate and should be judged together with activation, fallback, and correction-gap logs.
