# Latest Strict-Reward Cold-Start And Pretrained RL Analysis

## Objective

This report analyzes the latest complete direct Lyapunov safety-gated TD3 runs for the polymer CSTR two-setpoint disturbance study.

The analyzed result folders are:

- Cold start: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260519_143052`
- Pretrained: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260519_143039`

The immediately nearby `20260519_142904` and `20260519_142910` folders appear incomplete, so they were not used.

## Setup Verified From Results

Each study compares the safety-gated RL case against the matched MPC-only diagnostic:

- RL case: `bounded_hard_u_prev_0p1_xs_prev_0p1`
- MPC-only case: `mpc_only`
- Horizon length per episode: 800 steps
- Number of episodes: 200
- Total steps per case: 160000
- Two setpoints per episode, each held for 400 steps

The current scripts use the strict reward defaults:

| Parameter | Value |
|---|---|
| `Qy_diag` | `[8.0, 4.0]` |
| `k_rel` | `[0.0015, 0.00015]` |
| `band_floor_phys` | `[0.003, 0.035]` |
| `gate` | `"prod"` |
| `bonus_kind` | `"quadratic"` |
| `gamma_fallback` | `2.0` |
| `fallback_event_penalty` | `0.5` |
| `maintenance_move_weight` | `0.1` |
| `jitter_weight` | `0.02` |

One important consistency issue: these latest saved result files are not the earlier $\epsilon_{\mathrm{lyap}}=10^{-3}$ setup.

| Study | $\rho$ | $\epsilon_{\mathrm{lyap}}$ |
|---|---:|---:|
| Cold start | 0.990 | 1e-6 |
| Pretrained | 0.995 | 0.0 |

This matters because the safety gate is now stricter than the relaxed $\epsilon_{\mathrm{lyap}}=10^{-3}$ runs discussed earlier. Therefore the latest results should be interpreted as strict-reward plus stricter Lyapunov acceptance, not only strict-reward.

At the time this report was written, the local script files had been edited toward a matched relaxed-gate setup with $\rho=0.99$ and $\epsilon_{\mathrm{lyap}}=10^{-3}$. That active source setting is different from the saved result folders analyzed here, so a new rerun is needed before drawing conclusions about the matched $\epsilon_{\mathrm{lyap}}=10^{-3}$ case.

## Method Summary

At each step the actor proposes a candidate input $u_{\mathrm{cand}}$. The direct Lyapunov safety gate accepts it only if the predicted Lyapunov decrease condition is satisfied:

$$
V(x_{k+1}^{\mathrm{cand}} - x_s)
\le
\rho V(x_k - x_s) + \epsilon_{\mathrm{lyap}}.
$$

If the candidate is rejected, the controller falls back to the direct Lyapunov MPC action $u_{\mathrm{safe}}$.

The implemented reward is:

$$
r_k
=
r_{\mathrm{base},k}
-
\gamma_{\mathrm{fallback}}
\left\|u_{\mathrm{safe},k}-u_{\mathrm{cand},k}\right\|_{R_f}^2
-
c_{\mathrm{fallback}}
\mathbf{1}_{\mathrm{fallback},k}
-
J_{\mathrm{maint},k}
-
J_{\mathrm{jitter},k}.
$$

For these runs:

$$
\gamma_{\mathrm{fallback}}=2.0,
\qquad
c_{\mathrm{fallback}}=0.5.
$$

The error metrics below use the same alignment as the saved exporter: output $y_{k+1}$ is compared against the setpoint active at step $k$.

## Main Result

The strict reward did not yet make RL outperform MPC-only. MPC-only remains better on full-horizon reward and output RMSE in both cold-start and pretrained studies.

![Episode diagnostics](figures/2026-05-19_latest_strict_reward_rl_analysis/episode_learning_diagnostics.png)

Full-horizon performance:

| Case | Reward mean | eta RMSE | T RMSE | Mean RMSE |
|---|---:|---:|---:|---:|
| Cold RL | -7.055 | 0.129 | 0.298 | 0.214 |
| Cold MPC-only | -6.134 | 0.126 | 0.284 | 0.205 |
| Pretrained RL | -4.751 | 0.127 | 0.271 | 0.199 |
| Pretrained MPC-only | -3.267 | 0.122 | 0.244 | 0.183 |

Interpretation:

- Cold RL is close to cold MPC-only on full-horizon tracking, but still worse.
- Pretrained RL improves compared with cold RL, but pretrained MPC-only improves more.
- The scalar reward still favors MPC-only by a large margin, especially in the pretrained study.

## Steady Offset And Final Tail

The final 100 steps of the second setpoint are the most relevant region for offset-free behavior.

![Tail offset summary](figures/2026-05-19_latest_strict_reward_rl_analysis/tail_offset_summary.png)

Final-setpoint tail metrics:

| Case | Tail eta abs mean | Tail T abs mean | Final eta abs | Final T abs |
|---|---:|---:|---:|---:|
| Cold RL | 0.0146 | 0.0597 | 0.0080 | 0.0223 |
| Cold MPC-only | 0.0001 | 0.0086 | 0.0001 | 0.0086 |
| Pretrained RL | 0.0103 | 0.0679 | 0.0000 | 0.0251 |
| Pretrained MPC-only | 0.0067 | 0.0227 | 0.0067 | 0.0227 |

Strict-band success in the last 100 steps of the final setpoint:

| Case | Inside both strict bands |
|---|---:|
| Cold RL | 8.0% |
| Cold MPC-only | 100.0% |
| Pretrained RL | 14.0% |
| Pretrained MPC-only | 0.0% |

The pretrained RL result has one encouraging feature: its final eta offset is almost zero. However, this is not enough to claim offset-free superiority because its temperature tail error is larger than MPC-only. Cold MPC-only is the cleanest final-tail case because both outputs stay inside the strict bands throughout the final 100 steps.

The pretrained MPC-only strict-band score is 0% because eta sits slightly outside the very tight eta band, even though its temperature error is smaller than pretrained RL. This is a useful reminder that the strict-band metric is intentionally harsh.

## Fallback And Safety-Gate Dependence

The fixed fallback event penalty is active, but the correction-size penalty is still the dominant part of the fallback cost.

![Reward and fallback decomposition](figures/2026-05-19_latest_strict_reward_rl_analysis/reward_fallback_decomposition.png)

Safety-gate metrics:

| Case | Fallback rate | Actual intervention | Accepted rate | Solver holds |
|---|---:|---:|---:|---:|
| Cold RL | 1.40% | 1.40% | 98.60% | 150 |
| Cold MPC-only | 0.00% | 0.00% | 0.00% | 0 |
| Pretrained RL | 3.04% | 3.04% | 96.96% | 5 |
| Pretrained MPC-only | 0.00% | 0.00% | 0.00% | 0 |

Reward penalty decomposition:

| Case | Base reward mean | Aug reward mean | Correction penalty | Fixed event penalty |
|---|---:|---:|---:|---:|
| Cold RL | -6.560 | -7.055 | 0.473 | 0.007 |
| Cold MPC-only | -6.119 | -6.134 | 0.000 | 0.000 |
| Pretrained RL | -3.928 | -4.751 | 0.794 | 0.015 |
| Pretrained MPC-only | -3.255 | -3.267 | 0.000 | 0.000 |

The fixed event penalty behaves correctly, but its average contribution is small because it is only $0.5$ times the fallback rate. For pretrained RL, the fixed contribution is about 0.015 reward units per step, while the correction-size term contributes about 0.794. This means the current fallback punishment is mostly driven by large correction gaps, not the event count itself.

The pretrained RL fallback rate is higher than cold RL. That is a warning sign: pretraining gives better base reward and better full-horizon tracking than cold RL, but the safety gate still needs to intervene more often.

## Input Activity

![Input activity summary](figures/2026-05-19_latest_strict_reward_rl_analysis/input_activity_summary.png)

Input movement and bound activity:

| Case | Mean abs dQc | Mean abs dQm | Bound-hit rate |
|---|---:|---:|---:|
| Cold RL | 1.496 | 1.162 | 1.85% |
| Cold MPC-only | 1.509 | 0.948 | 1.88% |
| Pretrained RL | 1.009 | 1.133 | 0.59% |
| Pretrained MPC-only | 0.966 | 0.837 | 0.66% |

The pretrained cases move less and hit constraints less often than the cold cases. RL does not appear to be failing because of excessive input saturation. The larger issue is that the learned candidate still needs safety fallback and does not yet reduce temperature tail error below MPC-only.

## Last-Episode Tracking

![Last episode tracking](figures/2026-05-19_latest_strict_reward_rl_analysis/last_episode_tracking.png)

Last-episode metrics:

| Case | Last reward | Last eta RMSE | Last T RMSE | Last fallback count |
|---|---:|---:|---:|---:|
| Cold RL | -3.219 | 0.113 | 0.209 | 16 |
| Cold MPC-only | -2.670 | 0.113 | 0.198 | 0 |
| Pretrained RL | -4.076 | 0.122 | 0.243 | 31 |
| Pretrained MPC-only | -2.781 | 0.115 | 0.194 | 0 |

This last-episode view confirms the full-horizon story. RL is not collapsing, but it is also not beating MPC-only. The pretrained RL policy is still leaning on fallback more than desired, and its last-episode temperature tracking is worse than the matched MPC-only diagnostic.

## Scientific Interpretation

The strict reward is directionally reasonable, but this run suggests it is not sufficient by itself. The policy can learn a reasonable candidate, but the safety gate still corrects it often enough that the actor is not reliably internalizing the Lyapunov-feasible action structure.

The current result separates three effects:

- Reward alignment improved: fallback penalties now visibly reduce the augmented reward when safety intervention occurs.
- Safety feasibility remains the bottleneck: RL has nonzero fallback, especially in the pretrained case.
- Offset-free tracking is still dominated by MPC-only: the final-tail temperature error remains better under MPC-only.

The key practical conclusion is that we should not make the reward harsher blindly. The strict reward already makes RL worse in scalar reward when it uses fallback. The next change should help the actor learn the feasible MPC-like correction, not only punish it afterward.

## Risks And Inconsistencies Found

- The latest cold-start and pretrained runs do not use the same Lyapunov acceptance settings. Cold uses $\rho=0.99$, $\epsilon_{\mathrm{lyap}}=10^{-6}$, while pretrained uses $\rho=0.995$, $\epsilon_{\mathrm{lyap}}=0.0$.
- The previous discussion focused on $\epsilon_{\mathrm{lyap}}=10^{-3}$, but these latest results are stricter. This makes direct comparison with the earlier report unsafe unless we rerun both cases with matched $\rho$ and $\epsilon_{\mathrm{lyap}}$.
- The result exporter does not yet include explicit `fallback_event_penalty` component columns in these saved `step_table.csv` files. The fixed event contribution can still be inferred from `fallback_penalty - 2.0 * weighted_correction_gap`, but future runs should log the component directly.
- The MPC-only diagnostic has no actor candidate, so its `accepted_rate=0` is expected and should not be interpreted as unsafe behavior.

## Recommended Next Experiment

Run a controlled matched-gate experiment before changing the reward again. If the goal is to continue the $\epsilon_{\mathrm{lyap}}=10^{-3}$ line of study, use the active relaxed-gate setup in both scripts and treat it as a new experiment. If the goal is to isolate the strict saved runs, rerun both studies with matched strict settings.

Recommended matched relaxed-gate setup:

| Item | Value |
|---|---|
| Cold $\rho$ | 0.99 |
| Pretrained $\rho$ | 0.99 |
| Cold $\epsilon_{\mathrm{lyap}}$ | 1e-3 |
| Pretrained $\epsilon_{\mathrm{lyap}}$ | 1e-3 |
| Reward | Current strict reward |
| `fallback_event_penalty` | 0.5 |
| Exploration/noise | Keep current cold and pretrained settings |

Purpose:

- Determine whether the pretrained case is worse because the actor is worse, or because its gate is stricter.
- Make cold and pretrained comparisons scientifically clean.
- Keep the reward fixed so the next result isolates the Lyapunov gate setting.

Metrics that should improve:

- Pretrained RL fallback rate should drop below 2%.
- Final-tail temperature abs mean should move below 0.035.
- Last-episode fallback count should move toward the cold RL level or lower.

If the matched-gate run still shows the same pattern, the next coding step should add a supervised correction-loss or replay weighting around fallback states, because pure reward punishment is not enough to teach the actor the safe correction.

## Remaining Uncertainty

The current evidence is from one latest cold-start run and one latest pretrained run. The conclusions are reliable for these saved folders, but not yet statistically robust. A seed sweep or at least one matched-gate rerun is needed before deciding whether the strict reward should become the final thesis/paper configuration.
