# Cold-Start And Pretrained RL Analysis With $\epsilon_{\mathrm{lyap}}=10^{-3}$

Date: 2026-05-19

This report analyzes the latest direct Lyapunov safety-gated TD3 runs for the polymer CSTR two-setpoint disturbance case. The purpose is to decide what should be changed next in the RL reward and safety-gate settings before further coding.

Analyzed runs:

- Cold-start RL: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260519_010620/bounded_hard_u_prev_0p1_xs_prev_0p1`
- Cold-start MPC-only diagnostic: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260519_010620/mpc_only`
- Pretrained RL: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260519_010717/bounded_hard_u_prev_0p1_xs_prev_0p1`
- Pretrained MPC-only diagnostic: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260519_010717/mpc_only`

The plotted figures are saved under `report/figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/`.

## Method Being Audited

The RL policy proposes a candidate input move in scaled deviation coordinates. The safety gate evaluates the candidate against the direct Lyapunov first-step condition

$$
V_{k+1} \le \rho V_k + \epsilon_{\mathrm{lyap}},
$$

with $\rho = 0.98$ and $\epsilon_{\mathrm{lyap}} = 10^{-3}$ in these runs. If the candidate is not accepted, the controller falls back to the direct Lyapunov MPC candidate. The reward used for learning is the base tracking/move reward minus the safety correction penalty:

$$
r_{\mathrm{aug}} =
r_{\mathrm{base}}
- \gamma_{\mathrm{fallback}}
\left\|u_{\mathrm{safe}} - u_{\mathrm{cand}}\right\|_{R_f}^2,
$$

where the saved runs analyzed here used $\gamma_{\mathrm{fallback}} = 0.25$. The current notebooks for the next run now use $\gamma_{\mathrm{fallback}} = 2.0$. In implementation terms, the logged penalty is

$$
\texttt{fallback\_penalty}
= \gamma_{\mathrm{fallback}}
\left\|u_{\mathrm{safe}} - u_{\mathrm{cand}}\right\|_{R_f}^2.
$$

The key question is whether the improved behavior is coming from a genuinely better learned policy, a more forgiving reward, or the relaxed Lyapunov bound.

## Current Reward Mathematics

This section documents the reward implemented in `TD3Agent/reward_functions.py::make_reward_fn_relative_QR`. The saved runs analyzed in this report used `gamma_fallback = 0.25`. The notebooks are now configured for the strict offset-aligned reward described in this report, with `gamma_fallback = 2.0` and a fixed fallback event penalty `c_fallback = 0.5`.

The reward is computed in scaled deviation coordinates. Let $n_u = 2$ and let the output scaling range be

$$
\Delta y_i = y_{\max,i} - y_{\min,i}.
$$

At each time step, the rollout passes:

$$
e_k = y_{k+1}^{\mathrm{dev}} - y_{\mathrm{sp},k}^{\mathrm{dev}},
$$

and

$$
\Delta u_k = u_k^{\mathrm{scaled}} - u_{k-1}^{\mathrm{scaled}}.
$$

Here $e_k$ is the output tracking error in scaled deviation coordinates, while $\Delta u_k$ is the input move in scaled input coordinates.

For each output $i$, the reward builds a physical tracking band from the setpoint:

$$
b_i^{\mathrm{phys}}(k)
=
\max\left(k_i^{\mathrm{rel}}\left|y_{\mathrm{sp},i}^{\mathrm{phys}}(k)\right|,
b_{i,\min}^{\mathrm{phys}}\right).
$$

The band is converted to scaled coordinates as

$$
b_i(k) = \frac{b_i^{\mathrm{phys}}(k)}{\Delta y_i}.
$$

The transition width for the inside-band gate is

$$
\tau_i(k) = \tau_{\mathrm{frac}} b_i(k).
$$

The per-output soft inside-band score is

$$
s_i(k)
=
\sigma\left(
\frac{b_i(k)-|e_{k,i}|}{\max(\tau_i(k), 10^{-12})}
\right),
$$

where $\sigma(\cdot)$ is the logistic sigmoid. With the current `gate = "geom"`, the combined inside-band gate is

$$
w_{\mathrm{in}}(k)
=
\left(\prod_i s_i(k)\right)^{1/n_y}.
$$

This value is near 1 only when both outputs are inside their bands. It is smaller when either eta or temperature is outside the tolerance.

The core quadratic tracking cost is

$$
J_Q(k) = \sum_i Q_i e_{k,i}^2.
$$

The effective tracking cost blends the normal quadratic cost with the inside-band quadratic cost:

$$
J_{\mathrm{err}}(k)
=
(1-w_{\mathrm{in}}(k))J_Q(k)
+ w_{\mathrm{in}}(k)\lambda_{\mathrm{in}}J_Q(k).
$$

The move cost is

$$
J_{\Delta u}(k)
=
\sum_j R_j \Delta u_{k,j}^2.
$$

The reward also adds linear tracking pressure outside and inside the band. The edge slope for output $i$ is

$$
m_i(k) = 2 Q_i b_i(k).
$$

The outside-band linear cost is

$$
J_{\mathrm{out}}(k)
=
(1-w_{\mathrm{in}}(k))
\sum_i
\gamma_{\mathrm{out}} m_i(k)
\max(|e_{k,i}|-b_i(k), 0).
$$

The inside-band linear cost is

$$
J_{\mathrm{in}}(k)
=
w_{\mathrm{in}}(k)
\sum_i
\gamma_{\mathrm{in}} m_i(k)
\min(|e_{k,i}|, b_i(k)).
$$

The current reward includes a closeness bonus. Define the normalized error

$$
z_i(k) = \frac{|e_{k,i}|}{\max(b_i(k), 10^{-12})}.
$$

For the current `bonus_kind = "exp"`,

$$
\phi_{\exp}(z)
=
\frac{\exp(-\kappa z)-\exp(-\kappa)}
{1-\exp(-\kappa)},
$$

with $\kappa = \texttt{bonus\_k} = 12$. The bonus is

$$
B(k)
=
w_{\mathrm{in}}(k)\,\beta\,
\sum_i Q_i b_i(k)^2 \phi(z_i(k)).
$$

The base reward is therefore

$$
r_{\mathrm{base}}(k)
=
-\left[
J_{\mathrm{err}}(k)
+ J_{\Delta u}(k)
+ J_{\mathrm{out}}(k)
+ J_{\mathrm{in}}(k)
\right]
+ B(k).
$$

When the safety gate changes the candidate action, the fallback/correction gap is

$$
g_k = u_{\mathrm{cand},k}^{\mathrm{dev}} - u_{\mathrm{safe},k}^{\mathrm{dev}}.
$$

The correction penalty is active only when the executed safe action differs from the RL candidate:

$$
J_{\mathrm{fallback}}(k)
=
\mathbf{1}_{\mathrm{fallback}}(k)
\,\gamma_{\mathrm{fallback}}\,
\sum_j R_{f,j}g_{k,j}^2.
$$

The implementation also supports optional maintenance and jitter terms:

$$
J_{\mathrm{maint\_move}}(k)
=
\mathbf{1}_{|e_k|\le b^{\mathrm{maint}}}
\,\gamma_{\mathrm{maint}}\,
\sum_j R_j\Delta u_{k,j}^2,
$$

and

$$
J_{\mathrm{jitter}}(k)
=
\gamma_{\mathrm{jitter}}\,
\sum_i Q_i(e_{k,i}-e_{k-1,i})^2.
$$

The dwell reward is

$$
B_{\mathrm{dwell}}(k)
=
\gamma_{\mathrm{dwell}}N_{\mathrm{consecutive\ inside}}(k).
$$

In the current notebooks these optional terms are effectively off:

| Term | Current value |
|---|---|
| `maintenance_move_weight` | `0.0` |
| `jitter_weight` | `0.0` |
| `dwell_bonus` | `0.0` |

The full augmented reward is

$$
r(k)
=
r_{\mathrm{base}}(k)
- J_{\mathrm{fallback}}(k)
- J_{\mathrm{maint\_move}}(k)
- J_{\mathrm{jitter}}(k)
+ B_{\mathrm{dwell}}(k).
$$

Current implemented next-run reward parameters:

| Parameter | Value |
|---|---|
| `Qy_diag` | `[8.0, 4.0]` |
| `Rdu_diag` | `[1.0, 1.0]` |
| `k_rel` | `[0.0015, 0.00015]` |
| `band_floor_phys` | `[0.003, 0.035]` |
| `tau_frac` | `0.5` |
| `gamma_out` | `1.0` |
| `gamma_in` | `2.0` |
| `beta` | `2.0` |
| `gate` | `"prod"` |
| `lam_in` | `2.0` |
| `bonus_kind` | `"quadratic"` |
| `gamma_fallback` | `2.0` |
| `fallback_event_penalty` | `0.5` |
| `R_fallback_diag` | `[1.0, 1.0]` |
| `maintenance_band_scale` | `0.5` |
| `maintenance_move_weight` | `0.1` |
| `jitter_weight` | `0.02` |
| `dwell_bonus` | `0.0` |

The two most important limitations of the current reward are:

- This stricter setup has not yet been retrained, so the expected improvement is reward alignment rather than proven closed-loop performance.
- The fixed fallback event penalty is intentionally conservative at `0.5`; a later ablation can test `1.0` if fallback dependence remains high.

## Main Performance

The scalar comparison is mixed. MPC-only still has better mean reward and slightly better full-horizon temperature RMSE, but the RL cases are competitive and have better tail tracking in the last cycle.

![Performance summary](figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/performance_summary.png)

Performance metrics:

| Case | Reward mean | T RMSE | eta RMSE | Fallback |
|---|---|---|---|---|
| Cold RL | -5.182 | 0.407 | 0.134 | 1.45% |
| Cold MPC-only | -4.703 | 0.393 | 0.133 | 0.00% |
| Pretrained RL | -2.837 | 0.328 | 0.133 | 2.77% |
| Pretrained MPC-only | -2.272 | 0.310 | 0.133 | 0.00% |

The full-horizon numbers still favor MPC-only. However, full-horizon RMSE is dominated by transient behavior and does not isolate the offset-free question.

## Tail Tracking And Offset

The last-cycle tracking plot shows the more useful control story. Both RL cases reduce the late offset relative to their MPC-only diagnostic, especially in temperature.

![Last-cycle tracking](figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/last_cycle_tracking.png)

Tail metrics were computed from the last 100 samples of each setpoint segment in the final cycle.

| Case | Last T RMSE | Tail abs T | Final abs T |
|---|---|---|---|
| Cold RL | 0.244 | 0.016 | 0.0048 |
| Cold MPC-only | 0.219 | 0.028 | 0.0508 |
| Pretrained RL | 0.221 | 0.073 | 0.0295 |
| Pretrained MPC-only | 0.285 | 0.162 | 0.2891 |

Tail eta metrics:

| Case | Last eta RMSE | Tail abs eta | Final abs eta |
|---|---|---|---|
| Cold RL | 0.121 | 0.0029 | 0.0027 |
| Cold MPC-only | 0.115 | 0.0049 | 0.0096 |
| Pretrained RL | 0.115 | 0.0042 | 0.0060 |
| Pretrained MPC-only | 0.119 | 0.0092 | 0.0143 |

This is the strongest positive result: the learned policy is not simply worse than MPC-only. It is worse on the global reward but better near steady tracking in the final cycle. That means the reward is not aligned tightly enough with the research objective. It is still letting the training objective value transient behavior and action economy more strongly than final offset removal.

## Fallback Penalty

The fallback penalty is present but not yet strong enough to shape the policy decisively.

![Fallback diagnostics](figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/fallback_diagnostics.png)

Fallback and penalty summary:

| Case | Fallback rate | Mean penalty | Penalty sum |
|---|---|---|---|
| Cold RL | 1.45% | 0.086 | 13,816 |
| Pretrained RL | 2.77% | 0.104 | 16,685 |

When fallback happens, the penalty is meaningful locally, but averaged over all training steps it is small relative to the reward gap against MPC-only. For cold RL, the mean reward deficit versus MPC-only is about 0.48 reward units per step, while the mean fallback penalty is only 0.086. For pretrained RL, the reward deficit is about 0.56 per step, while the mean fallback penalty is 0.104.

This supports increasing the fallback penalty. A good first sweep would be:

- $\gamma_{\mathrm{fallback}} = 1.0$
- $\gamma_{\mathrm{fallback}} = 2.0$
- optional fixed event penalty for any fallback or actual intervention

The fixed event penalty matters because the current penalty is mostly gap-weighted. A fallback with a small correction gap can still be cheap, even though scientifically it means the policy failed the safety gate.

## Reward Alignment

The reward trace confirms that the scalar reward still favors MPC-only, even when RL gives better final offset behavior.

![Reward traces](figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/reward_traces.png)

This suggests the next reward change should not merely increase the fallback penalty. The base tracking reward also needs to be stricter around the setpoint. The current relative-band reward gives a smooth bonus inside a tolerance band; that is useful for early learning, but it may be too forgiving once the policy is close.

Recommended reward changes for the next coding step:

- Increase the temperature output weight because most visible offset problems are in $T$.
- Shrink the relative/floor bands or add a second tighter near-zero error term.
- Reduce or cap the inside-band bonus so being "good enough" is not rewarded as strongly as being offset-free.
- Add a near-setpoint maintenance term that penalizes residual steady error over the last part of each setpoint segment.
- Add a fixed penalty for fallback/actual intervention in addition to the correction-gap penalty.

The goal is to make the reward prefer small late error even if the transient reward is slightly worse.

## Reward Setup Sweep From Saved Trajectories

The saved trajectories were rescored offline with four reward setups. This is not a substitute for retraining, because a new reward changes the learned policy. It is still useful as a sanity check: a reward intended to highlight offset-free behavior should give better relative scores to trajectories that actually have lower late offset.

The four tested setups were:

| Setup | Purpose | Main changes |
|---|---|---|
| Logged gamma 0.25 | Reproduce latest saved runs | Existing reward with fallback weight 0.25 |
| Fallback gamma 2.0 | Immediate next run | Same base reward, fallback weight 2.0 |
| Strict offset candidate | Recommended reward candidate | Tighter bands, stronger T weight, less inside-band forgiveness |
| Very strict no bonus | Stress test | Very tight bands, no bonus, stronger penalties |

The reward-delta plot shows RL reward minus its matched MPC-only diagnostic. Positive values mean the reward prefers the RL trajectory over the MPC-only trajectory for that time window.

![Reward variant deltas](figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/reward_variant_deltas.png)

Offline reward deltas:

| Setup | Cold full | Cold tail | Pre full | Pre tail |
|---|---|---|---|---|
| Logged gamma 0.25 | -0.479 | -0.032 | -0.565 | 0.081 |
| Fallback gamma 2.0 | -1.084 | -0.032 | -1.294 | 0.081 |
| Strict offset candidate | -1.171 | -0.000 | -1.352 | 0.310 |
| Very strict no bonus | -1.494 | 0.012 | -1.700 | 0.465 |

This supports two points. First, increasing only the fallback penalty is a clean next run because it directly discourages safety-filter dependence, but it will make the saved RL trajectories score worse globally unless the actor learns to avoid fallback. Second, the stricter offset rewards make the pretrained RL trajectory look much better in the tail, which matches the actual tracking plot. That means the strict offset reward is directionally aligned with the offset-free objective.

The current and strict tracking bands are shown below against the last-cycle physical errors. The strict bands are roughly half of the current tolerances:

- Current bands: `k_rel = [0.003, 0.0003]`, `band_floor_phys = [0.006, 0.07]`
- Strict bands: `k_rel = [0.0015, 0.00015]`, `band_floor_phys = [0.003, 0.035]`

![Tail error versus reward bands](figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/tail_error_vs_reward_bands.png)

The strict bands are not arbitrary. They sit in the range where the final-cycle RL trajectories are close to acceptable, while the pretrained MPC-only diagnostic clearly remains outside the temperature band after the second setpoint. This is why the strict reward starts to favor pretrained RL in the tail.

For the strict offset candidate, the mean component sizes are:

![Strict reward component means](figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/strict_reward_component_means.png)

| Case | Tracking cost | Fallback penalty | Inside-band gate |
|---|---|---|---|
| Cold RL | 4.229 | 0.691 | 0.096 |
| Pretrained RL | 3.705 | 0.834 | 0.164 |

The inside-band gate becomes much smaller than in the logged reward, which is desirable here. The policy should not receive a large "good enough" bonus unless both outputs are genuinely close to their setpoints.

## Recommended Reward Hyperparameters

The previous low-risk immediate run was to use only the fallback-weight change:

| Parameter | Immediate value | Reason |
|---|---|---|
| `gamma_fallback` | `2.0` | Tests whether stronger punishment reduces fallback dependence |
| `Qy_diag` | `[5.0, 1.0]` | Keep base reward unchanged for a clean comparison |
| `k_rel` | `[0.003, 0.0003]` | Keep current band for attribution |
| `band_floor_phys` | `[0.006, 0.07]` | Keep current band for attribution |

The implemented default now skips that intermediate attribution-only run and uses the strict offset candidate:

| Parameter | Implemented value | Reason |
|---|---|---|
| `Qy_diag` | `[8.0, 4.0]` | Increase tracking pressure, especially temperature |
| `Rdu_diag` | `[1.0, 1.0]` | Keep move cost stable while changing tracking terms |
| `k_rel` | `[0.0015, 0.00015]` | Halve relative tolerance around each setpoint |
| `band_floor_phys` | `[0.003, 0.035]` | Halve minimum physical tolerance |
| `tau_frac` | `0.5` | Sharpen the inside/outside transition |
| `gamma_out` | `1.0` | Penalize outside-band error more strongly |
| `gamma_in` | `2.0` | Keep pressure on residual error inside the band |
| `beta` | `2.0` | Reduce the inside-band bonus from the current value 7.0 |
| `gate` | `"prod"` | Require both outputs to be close before large inside-band credit |
| `lam_in` | `2.0` | Make near-setpoint quadratic error matter more |
| `bonus_kind` | `"quadratic"` | Reward closeness smoothly without the strong exponential bonus |
| `gamma_fallback` | `2.0` | Keep stronger fallback punishment |
| `fallback_event_penalty` | `0.5` | Penalize every safety-gate fallback event, even with small correction gaps |
| `maintenance_band_scale` | `0.5` | Activate maintenance terms only in a tight core band |
| `maintenance_move_weight` | `0.1` | Discourage unnecessary moves once close |
| `jitter_weight` | `0.02` | Mildly discourage output chatter |
| `dwell_bonus` | `0.0` | Keep dwell memory disabled for the first strict reward run |

The mathematical intent of these changes is:

1. Make the zero-error neighborhood smaller.

Current:

$$
b_i^{\mathrm{phys}}
=
\max(k_i^{\mathrm{rel}}|y_{\mathrm{sp},i}|, b_{i,\min}^{\mathrm{phys}}).
$$

Change:

$$
k_i^{\mathrm{rel}} \leftarrow 0.5 k_i^{\mathrm{rel}},
\qquad
b_{i,\min}^{\mathrm{phys}} \leftarrow 0.5 b_{i,\min}^{\mathrm{phys}}.
$$

This makes the reward define "near zero offset" more strictly. In the current data, the pretrained RL trajectory is close enough to benefit from this stricter band in the tail, while the pretrained MPC-only diagnostic remains outside the temperature band after the second setpoint.

2. Increase temperature tracking pressure.

Current:

$$
Q = \operatorname{diag}(5, 1).
$$

Change:

$$
Q = \operatorname{diag}(8, 4).
$$

This keeps eta important but raises the temperature penalty, which is where the most visible steady offset appears.

3. Reduce the "good enough" bonus.

Current:

$$
B(k)
=
w_{\mathrm{in}}(k)\,\beta\,
\sum_i Q_i b_i(k)^2\phi_{\exp}(z_i(k)),
\qquad
\beta = 7.
$$

Change:

$$
B(k)
=
w_{\mathrm{in}}(k)\,\beta\,
\sum_i Q_i b_i(k)^2(1-z_i(k))^2,
\qquad
\beta = 2.
$$

The current exponential bonus can reward being inside a broad band too strongly. The quadratic bonus with a smaller $\beta$ still rewards closeness, but it gives less credit for merely being acceptable.

4. Require both outputs to be close.

Current:

$$
w_{\mathrm{in}}
=
\left(\prod_i s_i\right)^{1/n_y}.
$$

Change:

$$
w_{\mathrm{in}}
=
\prod_i s_i.
$$

The product gate is stricter than the geometric mean. It prevents one well-tracked output from masking the other output's residual offset.

5. Keep pressure inside the band.

Current:

$$
\lambda_{\mathrm{in}} = 1,
\qquad
\gamma_{\mathrm{in}} = 0.5.
$$

Change:

$$
\lambda_{\mathrm{in}} = 2,
\qquad
\gamma_{\mathrm{in}} = 2.
$$

This makes the reward continue pushing toward zero after the trajectory enters the band.

6. Penalize fallback as both a correction size and an event.

Current implemented term:

$$
J_{\mathrm{fallback}}
=
\mathbf{1}_{\mathrm{fallback}}
\gamma_{\mathrm{fallback}}\,
\left\|u_{\mathrm{safe}}-u_{\mathrm{cand}}\right\|_{R_f}^2.
$$

Proposed extension:

$$
J_{\mathrm{fallback,new}}
=
\mathbf{1}_{\mathrm{fallback}}
\left[
\gamma_{\mathrm{fallback}}
\left\|u_{\mathrm{safe}}-u_{\mathrm{cand}}\right\|_{R_f}^2
+ c_{\mathrm{fallback}}
\right].
$$

The implemented default uses `gamma_fallback = 2.0` and `c_fallback = 0.5`. The fixed term matters because a small correction can still indicate that the actor failed the Lyapunov safety gate. If fallback dependence remains high, `c_fallback = 1.0` is the next ablation.

7. Add near-setpoint maintenance behavior.

The reward function already has the hooks:

$$
J_{\mathrm{maint\_move}}
=
\mathbf{1}_{|e|\le b^{\mathrm{maint}}}
\gamma_{\mathrm{maint}}\,
\left\|\Delta u\right\|_R^2,
$$

and

$$
J_{\mathrm{jitter}}
=
\gamma_{\mathrm{jitter}}\,
\left\|e_k-e_{k-1}\right\|_Q^2.
$$

Recommended first values:

| Factor | Value | Purpose |
|---|---|---|
| `maintenance_band_scale` | `0.5` | Only activate maintenance inside a tight core band |
| `maintenance_move_weight` | `0.1` | Discourage input movement once near zero offset |
| `jitter_weight` | `0.02` | Mildly discourage output chatter |
| `dwell_bonus` | `0.0` initially | Avoid runaway episode-length reward until other terms are stable |

The dwell bonus should stay off at first. It can become useful later, but it introduces memory into the reward and can dominate long episodes if not capped.

The very strict no-bonus setup is not recommended as the first coding target. It does emphasize the pretrained RL tail, but it penalizes the full run too heavily and may make cold-start learning brittle. It is better as a later ablation after the strict offset candidate is stable.

The reward implementation now includes a fixed intervention penalty:

$$
r_{\mathrm{aug}}
= r_{\mathrm{base}}
- \gamma_{\mathrm{fallback}}\left\|u_{\mathrm{safe}}-u_{\mathrm{cand}}\right\|_{R_f}^2
- c_{\mathrm{fallback}}\mathbf{1}_{\mathrm{fallback}}.
$$

The first implemented value is `c_fallback = 0.5`. This punishes any fallback event even when the correction gap is small.

## Effect Of $\epsilon_{\mathrm{lyap}}=10^{-3}$

With $\rho=0.98$ and $\epsilon_{\mathrm{lyap}}=10^{-3}$,

$$
\frac{\epsilon_{\mathrm{lyap}}}{1-\rho}
=
\frac{10^{-3}}{0.02}
= 0.05.
$$

When $V_k < 0.05$, the bound $\rho V_k + \epsilon_{\mathrm{lyap}}$ can be larger than $V_k$. In that regime, the condition no longer enforces strict decrease; it enforces practical boundedness around the target.

![Epsilon relaxation audit](figures/2026-05-19_rl_eps1e-3_cold_pretrain_analysis/eps_relaxation_audit.png)

Lyapunov relaxation diagnostics:

| Case | V below 0.05 | Eps-only accepted | Last-cycle eps-only |
|---|---|---|---|
| Cold RL | 9.82% | 5.45% | 50.13% |
| Cold MPC-only | 11.92% | 5.58% | 26.13% |
| Pretrained RL | 19.81% | 7.81% | 22.50% |
| Pretrained MPC-only | 35.78% | 11.76% | 26.75% |

The answer is therefore nuanced:

- $\epsilon_{\mathrm{lyap}}=10^{-3}$ is not making the full run identical to MPC-only. The RL candidate is still accepted, rejected, penalized, and sometimes sent to fallback.
- It is weakening the Lyapunov gate substantially near convergence. In the final cycle, many steps are accepted only because of the $\epsilon$ term.
- It is best interpreted as a practical-stability tolerance, not as a strict Lyapunov contraction certificate.

For training, $\epsilon_{\mathrm{lyap}}=10^{-3}$ may be useful because it prevents near-target numerical fragility from dominating learning. For evaluation, it is too loose if the claim is strict Lyapunov decrease. The next comparison should evaluate the same learned policy under $\epsilon_{\mathrm{lyap}} \in \{10^{-3}, 10^{-4}, 10^{-5}, 0\}$ without retraining, if the current rollout tooling can support evaluation-only replay.

## Conclusions

The latest runs are genuinely better than earlier attempts in the sense that the RL policy can reduce final offset, especially for the pretrained case. However, the overall reward and full-horizon temperature RMSE still favor MPC-only, so the current reward is not yet selecting the controller behavior we care about most.

The three next coding changes should be:

1. Run the current notebooks with $\gamma_{\mathrm{fallback}} = 2.0$, then add a fixed fallback or intervention event penalty if fallback dependence remains.
2. Tighten the offset-free part of the reward: add a stricter near-setpoint residual term, especially for $T$, and reduce the reward forgiveness inside the current tolerance band.
3. Treat $\epsilon_{\mathrm{lyap}}=10^{-3}$ as a training aid, not the final certificate: keep it for learning if needed, but run evaluation sweeps at $10^{-4}$, $10^{-5}$, and $0$.

The most important scientific point is that the RL improvement is showing up in late offset metrics before it shows up in the aggregate reward. That is exactly the signal that the next coding step should target reward alignment rather than only increasing exploration or changing network noise.
