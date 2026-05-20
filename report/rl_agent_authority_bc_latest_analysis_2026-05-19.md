# Agent-Authority BC Latest Run Analysis

## Objective

This report analyzes the latest three reruns after the agent-authority behavioral cloning update: cold-start safety-gate RL, pretrained safety-gate RL, and direct Lyapunov MPC. The matched MPC-only diagnostics are kept visible because they explain how often the Lyapunov gate would have activated without intervention.

Analyzed folders:

- Cold start: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260519_212733`
- Pretrained: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260519_212739`
- Direct LMPC: `results/direct_lyap_ch2_lex/20260519_212725`

## Setup Checked

The RL scripts use the new BC authority setup. The actor proposes the candidate action, the direct LMPC controller supplies the demo target for actor BC, and the safety gate decides whether the actor action or fallback action is executed. Both RL scripts use $\rho=0.99$, $\epsilon_{\mathrm{lyap}}=10^{-3}$, strict offset reward defaults, trained-agent saving, and wall-clock timing.

![Performance and runtime](figures/2026-05-19_agent_authority_bc_latest_analysis/performance_runtime_summary.png)

## Full-Horizon Results

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

## Conclusions

- Pretrained RL is the best RL case for full-horizon tracking and reward in this latest run.
- Cold-start RL is better for safety-gate authority: fewer fallbacks, smaller fallback penalty, and smaller correction gap.
- Cold-start RL also gives the smallest final-tail temperature offset among the two RL agents, even though pretrained RL is better over the full horizon.
- The new BC formulation is working as intended because the actor remains the candidate policy and the safety gate records meaningful authority/fallback differences between cold and pretrained agents.
- Direct LMPC is slower than the RL safety-gate cases and has higher full-horizon raw-setpoint RMSE here, mostly because its direct target construction allows significant modified-target behavior.
- The next useful experiment is to reduce pretrained policy mismatch rather than discard pretraining: use a short adapter BC phase, lower initial pretrained actor authority, or preload only lower layers while letting the output head adapt to the strict fallback reward.
