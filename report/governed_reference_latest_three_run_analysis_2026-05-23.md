# Governed-Reference Latest Three-Run Analysis

Date: 2026-05-23

## Data Sources

This report analyzes the latest completed 300-episode runs after promoting the governed-reference target selector to the active default.

| Method | Result folder | Primary case |
|---|---|---|
| Direct Lyapunov MPC | `results/directLyap/20260523_014446/` | `lyap_governed_reference` |
| Cold-start safety-gate RL | `results/ColdStart/20260523_014457/` | `rl_gate_governed_reference` |
| Pretrained safety-gate RL | `results/Pretrain/20260523_014514/` | `rl_gate_governed_reference` |

Each run has 300 episodes and 240,000 control steps. Each episode contains two setpoint blocks of length 400. The active target selector is governed-reference, and the controller still tracks the raw setpoint:

$$
y_{\mathrm{track},k}=y_{sp,k}.
$$

The governed target remains a Lyapunov certificate target:

$$
y_{sp,k}\rightarrow r_k\rightarrow (x_{s,k},u_{s,k},y_{s,k}).
$$

## Important Comparison Caveat

The direct MPC and RL runners now use the same governed-reference target idea, but their tuning is not fully identical.

The direct MPC runner uses:

$$
Q_y=\operatorname{diag}(5,1).
$$

The RL runners use the stricter reward/controller setup:

$$
Q_y=\operatorname{diag}(8,6).
$$

Therefore, direct-versus-RL RMSE comparisons are useful as current workflow evidence, but they are not a perfectly isolated controller-family ablation. The most controlled comparisons are:

- direct Lyap versus direct `mpc_only` inside `results/directLyap/20260523_014446/`,
- cold RL versus cold `mpc_only` inside `results/ColdStart/20260523_014457/`,
- pretrained RL versus pretrained `mpc_only` inside `results/Pretrain/20260523_014514/`.

## Executive Findings

The governed-reference target selector is numerically reliable in the latest runs. Direct Lyap has 100% target success, 100% solver success, and 100% hard contraction satisfaction. The RL runs also have 100% target success and no target reuse or target failure.

The strongest full-horizon tracking result is pretrained RL. Its mean output RMSE is 0.193, compared with 0.296 for cold-start RL and 0.378 for direct Lyap. However, direct Lyap still has the best near-steady tail offset.

Cold-start RL has slightly stronger safety-gate authority than pretrained RL. Cold-start RL needed actual intervention on 0.603% of steps, while pretrained RL needed intervention on 0.978% of steps. This repeats the earlier pattern: cold-start can gain more authority, but pretrained is still better for full-horizon tracking.

The governed-reference target selector helped direct Lyap become very safe. In the direct run, `mpc_only` and direct Lyap are almost identical, and `mpc_only` has zero would-be Lyapunov unsafe rate. This means the governed target made the diagnostic offset-free MPC trajectory already satisfy the direct Lyapunov contraction test in this run.

## Summary Figures

![Performance and runtime summary](figures/2026-05-23_governed_reference_latest_three/01_performance_runtime_summary.png)

![Safety authority summary](figures/2026-05-23_governed_reference_latest_three/02_safety_authority_summary.png)

![Target and tail summary](figures/2026-05-23_governed_reference_latest_three/03_target_tail_summary.png)

![RL episode trends](figures/2026-05-23_governed_reference_latest_three/04_rl_episode_trends.png)

## Final Evaluation Episode

The final episode is especially important because the RL scripts force the last episode into evaluation/test behavior. For direct Lyap there is no learning policy, so episode 300 is used as the matching final-episode window. This is not the same as a separate saved-agent test, but it is the cleanest within-run evaluation slice.

![Final episode raw setpoint tracking](figures/2026-05-23_governed_reference_latest_three/05_final_episode_tracking.png)

The final episode is much more favorable to the RL agents than the full-horizon training average. Cold-start RL and pretrained RL are almost tied on final-episode RMSE, and both are better than direct Lyap in this final slice.

| Method | Final $\eta$ RMSE | Final $T$ RMSE | Final mean RMSE |
|---|---:|---:|---:|
| Direct Lyap | 0.189 | 0.548 | 0.368 |
| Cold RL | 0.127 | 0.249 | 0.188 |
| Pretrained RL | 0.127 | 0.251 | 0.189 |

The final 20 samples of episode 300 show very small near-steady errors for all three methods:

| Method | Tail $\eta$ error | Tail $T$ error |
|---|---:|---:|
| Direct Lyap | 0.0017 | 0.0092 |
| Cold RL | 0.0044 | 0.0051 |
| Pretrained RL | 0.0013 | 0.0055 |

This changes the interpretation of cold-start RL. Over the full training horizon cold-start RL is worse than pretrained RL because its early episodes are costly. But by the final evaluation episode, cold-start RL has essentially caught up to pretrained RL in tracking. The remaining question is whether this final behavior persists in a separate saved-agent evaluation.

The final-episode safety counts are low:

| RL case | Actual interventions | Verified fallbacks | Diagnostic unsafe |
|---|---:|---:|---:|
| Cold RL | 3 | 3 | 0 |
| Pretrained RL | 4 | 4 | 0 |

This supports the view that both final RL policies are mostly operating inside the Lyapunov safety gate. Cold-start RL has slightly stronger authority in this final episode, but the difference is small.

## Final Episode Target Command And Target Output

For the direct Lyap run, the saved arrays include the full governed-reference decomposition:

$$
y_{sp,k}\rightarrow r_k\rightarrow y_{s,k}.
$$

![Direct final episode governed command and target](figures/2026-05-23_governed_reference_latest_three/06_direct_final_episode_rs_ys_ysp.png)

In direct Lyap episode 300, the mean mismatch values are:

| Signal gap | Mean $\ell_\infty$ gap |
|---|---:|
| $\|r_s-y_{sp}\|_\infty$ | 0.288 |
| $\|y_s-r_s\|_\infty$ | 0.046 |
| $\|y_s-y_{sp}\|_\infty$ | 0.328 |

This is the intended governed-reference behavior. Most of the target modification comes from the command governor moving $r_s$ away from the raw setpoint. The second-stage steady target then tracks the governed command closely because $\|y_s-r_s\|_\infty$ is much smaller.

For RL, the arrays currently export $y_s$ and $y_{sp}$, but `r_target_phys_store` is `NaN`. The step table confirms `selector_stage = governed_reference_target`, so governed-reference is active, but the intermediate command $r_s$ is not currently available in the RL bundle.

![RL final episode target output and setpoint](figures/2026-05-23_governed_reference_latest_three/07_rl_final_episode_ys_ysp.png)

The final-episode target mismatch is:

| Method | Mean $\|y_s-y_{sp}\|_\infty$ | Max $\|y_s-y_{sp}\|_\infty$ |
|---|---:|---:|
| Direct Lyap | 0.328 | 3.700 |
| Cold RL | 0.167 | 6.342 |
| Pretrained RL | 0.170 | 6.491 |

The RL mean target mismatch is lower than direct Lyap in episode 300, which is consistent with the strong final-episode tracking. The maximum mismatch is still large around transitions, which is expected when the setpoint changes and the governed target temporarily protects feasibility.

![Final episode target mismatch diagnostics](figures/2026-05-23_governed_reference_latest_three/08_final_episode_target_mismatch.png)

The key diagnostic point is that governed-reference is doing two different jobs:

- $r_s-y_{sp}$ shows how much the requested setpoint is governed.
- $y_s-r_s$ shows whether the steady target can realize the governed command.
- $y_s-y_{sp}$ is the total Lyapunov-target mismatch seen by the controller.

For paper-quality analysis, the RL exporter should store $r_s$ directly so the same decomposition can be shown for cold-start and pretrained RL.

## Full-Horizon Tracking

The primary performance comparison is raw setpoint tracking in physical output units.

| Method | $\eta$ RMSE | $T$ RMSE | Mean RMSE | Reward mean |
|---|---:|---:|---:|---:|
| Direct Lyap | 0.191 | 0.565 | 0.378 | -4.333 |
| Cold RL | 0.187 | 0.406 | 0.296 | -11.448 |
| Pretrained RL | 0.133 | 0.253 | 0.193 | -4.594 |

Reward should be interpreted carefully across direct and RL because the direct and RL scripts use different reward/controller weights. Within the RL pair, the reward comparison is meaningful: pretrained RL is much better than cold-start RL over the full run.

The same-run `mpc_only` baselines are:

| Run | `mpc_only` $\eta$ RMSE | `mpc_only` $T$ RMSE | `mpc_only` mean RMSE |
|---|---:|---:|---:|
| Direct runner | 0.191 | 0.565 | 0.378 |
| Cold runner | 0.187 | 0.388 | 0.287 |
| Pretrain runner | 0.120 | 0.217 | 0.169 |

Within each RL runner, the RL policy is still slightly worse than its same-run `mpc_only` baseline. This means the current RL agent is not yet beating offset-free MPC in raw tracking. The value of RL at this point is speed and policy autonomy under a safety gate, not superior tracking versus the local `mpc_only` baseline.

## Tail Offset And Settling

The final-tail behavior tells a different story from full-horizon RMSE. For the last 50 episodes, the average last-20-step error per setpoint block is:

| Method | Tail $\eta$ error | Tail $T$ error |
|---|---:|---:|
| Direct Lyap | 0.0017 | 0.0093 |
| Cold RL | 0.0080 | 0.0208 |
| Pretrained RL | 0.0050 | 0.0174 |

Direct Lyap still gives the cleanest near-steady offset. The RL methods, especially pretrained RL, are better over the full horizon because they reduce transient tracking error more strongly, but direct Lyap is still the most offset-free near the end of each setpoint block.

This is important scientifically: governed-reference improved target feasibility and direct Lyapunov reliability, but it did not erase the classic tradeoff between full-horizon transient RMSE and final-tail offset.

## Safety-Gate Authority

For the RL cases, actual intervention means the safety layer did not simply execute the actor candidate unchanged.

| RL case | Actual intervention | Verified fallback | No-intervention rate |
|---|---:|---:|---:|
| Cold RL | 0.603% | 0.525% | 99.397% |
| Pretrained RL | 0.978% | 0.904% | 99.022% |

Cold-start RL has stronger gate authority because it requires fewer interventions. However, this lower fallback rate does not automatically mean better tracking. Pretrained RL accepts slightly more safety correction but achieves much better full-horizon RMSE.

For `mpc_only`, fallback is not executed. The useful value is the diagnostic would-be activation rate:

| Diagnostic case | Would-be activation |
|---|---:|
| Cold-run `mpc_only` | 0.673% |
| Pretrain-run `mpc_only` | 0.940% |
| Direct-run `mpc_only` | 0.000% |

The direct-run `mpc_only` diagnostic being zero is a strong sign that governed-reference can make the direct Lyapunov certificate compatible with the offset-free MPC action under the direct runner tuning.

## Target Selector Behavior

The governed-reference target succeeds in all primary cases, but target quality differs strongly across trajectories.

| Method | Target success | Mean $\|y_s-y_{sp}\|_\infty$ | Mean target residual |
|---|---:|---:|---:|
| Direct Lyap | 1.000 | 0.557 | 0.064 |
| Cold RL | 1.000 | 0.741 | 0.196 |
| Pretrained RL | 1.000 | 0.256 | 0.099 |

Pretrained RL keeps the target much closer to the raw setpoint than cold-start RL. This is not because the selector is a fixed mapping. The target is recomputed inside the closed loop from the current observer state, disturbance estimate, previous target, previous input, and current setpoint. A better trajectory can lead to a better next target.

This is the most important mechanistic explanation for why pretrained RL performs better here:

$$
y_{s,k}
=
g(\hat x_k,\hat d_k,u_{k-1},x_{s,k-1},r_{k-1},y_{sp,k}).
$$

Pretrained RL stays in a region where the governed target remains closer to the requested setpoint. Cold-start RL explores more and induces larger target movement and target mismatch.

The target regularization terms also show this:

| Method | Mean $\|u_s-u_{k-1}\|_\infty$ | Mean $\|x_s-x_{s,k-1}\|_\infty$ |
|---|---:|---:|
| Direct Lyap | 0.013 | 0.011 |
| Cold RL | 0.241 | 0.372 |
| Pretrained RL | 0.108 | 0.079 |

Cold-start RL causes much larger target/input movement. That does not break feasibility, but it makes the closed-loop target problem less steady.

## Runtime

Measured wall-clock cost per control step:

| Method | Seconds/step | Steps/second |
|---|---:|---:|
| Direct Lyap | 0.0406 | 24.6 |
| Cold RL | 0.0223 | 44.8 |
| Pretrained RL | 0.0219 | 45.7 |

The RL safety-gate runs are about 1.8 times faster than direct Lyap under the current instrumentation. This supports the practical motivation for the safety-gate RL workflow: keep a Lyapunov-backed safety layer while using the learned actor for faster candidate actions.

This wall-clock comparison should still be reported cautiously because debug export, logging, and runner structure can affect timing. It is useful evidence, but a clean deployment-time benchmark should be done later with minimal plotting/export overhead.

## Learning Dynamics

Cold-start RL begins much worse but catches up late in training. In the first episode, cold-start RL has reward mean -77.9 and 12 verified fallbacks. In the final 50 episodes, its mean episode RMSE is 0.174.

Pretrained RL starts much closer to useful behavior. Its first episode reward mean is -17.3 with 7 verified fallbacks. In the final 50 episodes, its mean episode RMSE is 0.176.

So the late-stage RMSE values are similar, but pretrained RL wins the full-horizon metric because it avoids the severe early-training penalty. This means pretraining still matters if the training run itself is part of the evidence. If only the final saved agent matters, the saved-agent evaluation file should be used next.

## Method Interpretation

The new governed-reference selector is a clear improvement for direct Lyapunov feasibility. The direct-run `mpc_only` diagnostic has zero would-be unsafe rate, and direct Lyap satisfies hard contraction at every step without slack.

The RL result is more nuanced:

- Pretrained RL is the best full-horizon tracker.
- Cold-start RL has slightly stronger gate authority.
- Direct Lyap remains the cleanest final-tail offset controller.
- Same-run `mpc_only` is still better than RL in raw RMSE.
- Target mismatch is now a measurable mechanism, not a hidden issue.

The target-selector problem is not gone, but it is better instrumented. The governed-reference target is now stable enough to support the next RL experiments.

## Recommended Next Step

Run the saved-agent evaluation next, using the saved agents from:

```text
results/ColdStart/20260523_014457/rl_gate_governed_reference/
results/Pretrain/20260523_014514/rl_gate_governed_reference/
```

The saved-agent test should compare:

- final cold RL agent plus safety gate,
- final pretrained RL agent plus safety gate,
- same-tuning `mpc_only`,
- direct Lyap with governed-reference.

The key reason is that online-training metrics mix learning transient with final policy quality. The latest data suggest cold-start RL may catch up late, so the saved-agent test is the right place to decide whether cold or pretrained is actually better after training.

## Reproducibility Notes

Generated summary figures are stored under:

```text
report/figures/2026-05-23_governed_reference_latest_three/
```

The compact extracted metrics file is:

```text
report/governed_reference_latest_three_run_metrics_2026-05-23.json
```

Final-episode extracted metrics are stored at:

```text
report/figures/2026-05-23_governed_reference_latest_three/final_episode_metrics.json
```

One logging gap remains: the direct run stores governed command diagnostics such as $r_k-y_{sp}$ and $y_s-r_k$, but the RL arrays currently store `r_target_phys_store` as `NaN`. The RL step table confirms `selector_stage = governed_reference_target`, so the governed selector was active, but the command itself is not exported in the RL bundle. Before paper-grade reporting, the RL debug exporter should store $r_k$ explicitly.
