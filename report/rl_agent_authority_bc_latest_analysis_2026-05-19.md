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

This section describes the implemented workflow before interpreting the numerical results.

### Step 1: Plant, Scaling, And Setpoints

The case study is the polymer CSTR. The manipulated inputs are coolant flow $Q_c$ and monomer flow $Q_m$. The controlled outputs are viscosity-like $\eta$ and reactor temperature $T$. The scripts run in physical plant coordinates, but the MPC and RL calculations use scaled deviation variables around the steady-state anchors:

$$
\Delta u_k = u_k^{\rm scaled}-u_{\rm ss}^{\rm scaled}.
$$

$$
\Delta y_k = y_k^{\rm scaled}-y_{\rm ss}^{\rm scaled}.
$$

The two-step setpoint schedule is generated from the same physical setpoints for all three scripts. Each setpoint block has length `set_points_len`, and each full episode spans both setpoint blocks. For the analyzed runs, the direct LMPC and RL cases used the same disturbance pattern and the same direct-output-disturbance augmentation.

### Step 2: Offset-Free Augmented Model

The controller model is augmented with an output-disturbance state. The observer state is written as $xhatdhat_k$, which combines the estimated plant state and the estimated disturbance. The observer gain $L$ updates this augmented estimate from the measured output error. This is important because all three controllers are evaluated under disturbance, so good behavior means tracking the requested output while compensating the estimated output disturbance.

### Step 3: Direct Lyapunov MPC Candidate

The direct LMPC path first builds an admissible steady target from the current augmented state, disturbance estimate, setpoint, and input bounds. The target selector returns a steady target $(x_s,d_s,u_s,y_s)$. The online MPC then computes a first input move subject to input bounds, terminal ingredients, and a first-step Lyapunov contraction test:

$$
V_{k+1}^{\rm first} \le \rho V_k + \epsilon_{\rm lyap}.
$$

In these runs, $\rho=0.99$ and $\epsilon_{\rm lyap}=10^{-3}$. The direct LMPC script executes this direct controller. The RL scripts use the same direct LMPC machinery as the safety-gate fallback and as the behavioral-cloning teacher.

### Step 4: MPC-Only Diagnostic Meaning

The MPC-only rows are not safety-gated RL controllers. They are diagnostic baselines. Actual fallback is zero by construction for MPC-only. Therefore, for MPC-only activation plots and tables, the meaningful count is the would-be gate activation count: how many steps would have failed the Lyapunov contraction check if the safety gate had been active.

### Step 5: TD3 State And Action

The TD3 actor receives an augmented RL state containing the estimated augmented state, current setpoint information, and previous input information. The actor output is a bounded input-deviation action:

$$
u_{\rm rl,dev,k} = \pi_\theta(s_k).
$$

This action is interpreted in scaled input-deviation coordinates. It is converted consistently before plant execution, reward calculation, and safety diagnostics.

### Step 6: Agent-Authority Behavioral Cloning

The behavioral-cloning phase keeps the RL actor in authority. At each BC step:

1. The actor proposes $u_{\rm rl,dev,k}$.
2. The direct LMPC teacher independently computes $u_{\rm lmpc,dev,k}$.
3. The safety gate evaluates the actor candidate, not the teacher action.
4. The plant executes the accepted actor action or the fallback action.
5. The critic replay buffer stores the executed safe action.
6. The actor demo buffer stores $u_{\rm lmpc,dev,k}$ as the imitation target.

This design avoids the older issue where BC execution could become identical to LMPC execution. The teacher guides the actor through BC loss, but the actor remains the proposed controller from the start.

### Step 7: Safety Gate And Fallback

The safety gate checks whether the actor candidate satisfies the first-step Lyapunov contraction condition. If it passes, the actor action is executed. If it fails, the gate solves or reuses the direct LMPC fallback. The logged correction gap is:

$$
g_k=u_{\rm cand,k}-u_{\rm exec,k}.
$$

This separates candidate-policy quality from final safe closed-loop behavior. A controller can track well after fallback while still being a poor autonomous actor if the correction gap and fallback rate remain high.

### Step 8: Soft Handoff

After the BC phase, the scripts apply a short linear handoff over five episodes. The candidate action is blended from the teacher-shaped behavior toward the pure actor candidate before the safety gate checks it. This avoids an abrupt switch from imitation-dominated behavior to full online RL behavior.

### Step 9: Reward And Learning Objective

The reward combines output tracking, input movement, near-setpoint residual error, fallback correction cost, and a fixed fallback event cost. The analyzed runs used the earlier strict reward with `fallback_event_penalty = 0.5`. The next-run scripts now use the stricter fallback and offset settings described later in this report, with maintenance and jitter weights set to zero during the high-exploration diagnostic run.

The TD3 discount factor for future runs is now `GAMMA = 0.995`. This changes the RL return horizon only. It does not change the Lyapunov contraction factor, which remains $\rho=0.99$.

### Step 10: Evaluation Metrics

The report separates four kinds of evidence:

- Full-horizon tracking: reward mean, output RMSE, and mean RMSE across the whole run.
- Tail offset: final-window absolute output error, used as a steady-offset diagnostic.
- Authority diagnostics: fallback rate, actual intervention rate, correction gap, and BC teacher gap.
- Runtime diagnostics: total seconds, seconds per episode, seconds per control step, and steps per second.

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

With `gate = "prod"`, the joint near-setpoint weight is:

$$
w_{\rm in}=\prod_i s_i.
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
