# Latest Runner Analysis With Slide-Ready Figures

## Objective

This report revises the latest runner analysis using the figure logic from `report/rl_agent_authority_bc_latest_analysis_2026-05-19.html`, but with a smaller set of fresh data-derived figures. The earlier draft embedded too many raw runner figures. This version keeps only the figures needed for the result story and regenerates them from the latest saved arrays and tables.

Important analysis rules:

- Episode 1 is excluded from all aggregate and trend analysis.
- The report starts with only the main three methods: Direct LMPC, cold-start RL with LMPC safety gate, and pretrained RL with LMPC safety gate.
- The no-safety diagnostic is not included in the main comparison.
- The no-safety diagnostic is renamed as `Pretrained RL without safety gate` and is compared only against pretrained RL with the safety gate.
- Wall-clock comparison is reported only as a post-BC proxy because the result files store total wall-clock time, not per-episode wall-clock time.

Analyzed latest result folders:

- Direct LMPC: `results/directLyap/20260524_211850/lyap_mix_u0p1_x0p1_lex`
- Cold-start RL with LMPC safety gate: `results/ColdStart/20260524_211856/bounded_hard_u_prev_0p1_xs_prev_0p1`
- Pretrained RL with LMPC safety gate: `results/Pretrain/20260524_211859/bounded_hard_u_prev_0p1_xs_prev_0p1`
- Pretrained RL without safety gate: `results/Pretrain/20260524_211859/mpc_only`

The generated figures and computed metrics are stored in:

- `report/figures/2026-05-25_latest_runner_data_figures/`
- `analysis/latest_runner_report_figures.py`

## Main Finding

The latest data show a clean tradeoff:

- Pretrained RL with the LMPC safety gate gives the best main-method raw-setpoint tracking after episode 1.
- Cold-start RL learns strongly, but its full-window RMSE is still higher because the early post-episode-1 learning transient is large.
- Direct LMPC has the best final 100-step tail offset for both outputs, but its full-window raw-setpoint RMSE is higher because it spends more of the run around modified admissible targets.
- The safety gate remains important. Pretrained RL tracks better than cold start, but it needs more gate intervention.
- The pretrained no-safety diagnostic tracks even better, but it would have activated the gate on about 29% of steps after episode 1, so it should not be presented as a safe controller.

## Main Three-Method Comparison

![Main performance and runtime](figures/2026-05-25_latest_runner_data_figures/performance_runtime_summary.svg)

The main comparison excludes episode 1 and excludes the no-safety diagnostic. The wall-clock panel uses the recorded mean milliseconds per step as a post-BC proxy because per-episode timing was not logged.

| Method | Reward mean | eta RMSE | T RMSE | Mean RMSE | Tail eta abs | Tail T abs | ms / step |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct LMPC | -4.339 | 0.191 | 0.566 | 0.378 | 0.0030 | 0.0164 | 36.20 |
| Cold-start RL + LMPC gate | -10.982 | 0.175 | 0.393 | 0.284 | 0.0127 | 0.0034 | 19.00 |
| Pretrained RL + LMPC gate | -5.477 | 0.124 | 0.246 | 0.185 | 0.0128 | 0.0129 | 20.13 |

Interpretation:

- Pretrained RL with the LMPC gate is the strongest main method for full-window raw-setpoint tracking.
- Cold-start RL is much better than its first episode suggested, but the early learning transient after episode 1 still affects the full-window average.
- Direct LMPC is not weak in final offset. Its issue is the full-window raw-setpoint error and visible input cycling, not final steady offset.
- Direct LMPC reward is still not strictly comparable with the RL reward because the direct runner logs a different reward setup.

## Episode Trends

![Episode reward and RMSE trends](figures/2026-05-25_latest_runner_data_figures/episode_reward_rmse_trends.svg)

Episode 1 is excluded from the plotted lines. The BC teacher window is episodes 1-20, the handoff window is episodes 21-25, and online training begins at episode 26.

Important reading:

- Cold-start RL improves rapidly after the initial learning window.
- Pretrained RL starts much closer to useful control and stays ahead in RMSE for most of the run.
- Direct LMPC is steady rather than learned. Its reward line should be treated cautiously because its reward logger differs from the RL reward.

Post-BC metrics, episodes 26-300:

| Method | Reward mean | eta RMSE | T RMSE | Mean RMSE | Actual intervention |
|---|---:|---:|---:|---:|---:|
| Direct LMPC | -4.339 | 0.191 | 0.565 | 0.378 | n/a |
| Cold-start RL + LMPC gate | -7.203 | 0.121 | 0.257 | 0.189 | 1.218% |
| Pretrained RL + LMPC gate | -5.342 | 0.119 | 0.230 | 0.175 | 3.757% |

The post-BC window narrows the gap between cold-start and pretrained RL, but pretrained still has lower tracking error.

## Safety-Gate Authority

![RL authority diagnostics](figures/2026-05-25_latest_runner_data_figures/rl_authority_diagnostics.svg)

The safety-gate comparison is only between the two RL safety-gated methods. Direct LMPC is not a safety-gated actor, so it is not included in this authority plot.

| RL method | Actual intervention | Fallback-only rate | Candidate executed | Fallback penalty mean | Post-BC correction gap |
|---|---:|---:|---:|---:|---:|
| Cold-start RL + LMPC gate | 1.191% | 1.095% | 98.809% | 0.607 | 0.184 |
| Pretrained RL + LMPC gate | 3.527% | 3.428% | 96.473% | 1.403 | 0.478 |

Interpretation:

- Cold-start RL is more compatible with the gate.
- Pretrained RL tracks better but asks for more safety correction.
- This is not a contradiction. A policy can have better tracking but lower gate authority.

## Reward And Fallback Trends

![RL reward and fallback trends](figures/2026-05-25_latest_runner_data_figures/rl_reward_fallback_trends.svg)

The reward and intervention trends show the same mechanism as the authority scorecard:

- Cold-start RL moves from poor early reward toward a much better online regime.
- Pretrained RL has stronger reward throughout most of the run.
- Pretrained RL pays a larger fallback penalty because the gate corrects more of its actions.

The result is mixed rather than one-sided. Pretraining gives a stronger tracking prior, but the prior is not fully aligned with the latest strict Lyapunov gate and fallback penalty.

## Final Tail Offset

![Tail offset comparison](figures/2026-05-25_latest_runner_data_figures/tail_offset_comparison.svg)

The final 100 steps of episode 300 are used as a compact steady-offset check against the raw setpoint.

| Method | Tail eta abs mean | Tail T abs mean | Final eta abs | Final T abs |
|---|---:|---:|---:|---:|
| Direct LMPC | 0.0030 | 0.0164 | 0.0015 | 0.0081 |
| Cold-start RL + LMPC gate | 0.0127 | 0.0034 | 0.0127 | 0.0034 |
| Pretrained RL + LMPC gate | 0.0128 | 0.0129 | 0.0128 | 0.0129 |

This is the place where Direct LMPC looks strongest. It has the smallest eta tail offset and a small temperature tail offset. Cold-start RL has the smallest temperature tail offset among the RL methods.

## Final-Episode Tracking

![Final episode tracking](figures/2026-05-25_latest_runner_data_figures/last_episode_tracking_primary_methods.svg)

The final-episode plot is the best single tracking figure for slides:

- The setpoint switches from `[4.5, 324]` to `[3.4, 321]`.
- All three methods settle to the second setpoint.
- Pretrained RL has a visibly strong final episode.
- Direct LMPC is close at the tail, but earlier full-run diagnostics still matter because its target path can differ from the raw setpoint.

## Pretrained With And Without The Safety Gate

The diagnostic formerly labeled as `mpc_only` is renamed here as `Pretrained RL without safety gate`. This section compares only pretrained RL with the safety gate against pretrained RL without the safety gate.

![Pretrained with and without safety gate](figures/2026-05-25_latest_runner_data_figures/pretrained_with_without_safety_gate.svg)

| Pretrained case | Reward mean | eta RMSE | T RMSE | Mean RMSE | Actual gate | Would-be gate |
|---|---:|---:|---:|---:|---:|---:|
| With LMPC safety gate | -5.477 | 0.124 | 0.246 | 0.185 | 3.527% | 0.000% |
| Without safety gate | -3.279 | 0.120 | 0.215 | 0.168 | 0.000% | 29.168% |

Post-BC window, episodes 26-300:

| Pretrained case | Reward mean | Mean RMSE | ms / step | Actual gate | Would-be gate |
|---|---:|---:|---:|---:|---:|
| With LMPC safety gate | -5.342 | 0.175 | 20.13 | 3.757% | 0.000% |
| Without safety gate | -3.019 | 0.158 | 17.08 | 0.000% | 30.342% |

Interpretation:

- The no-safety diagnostic has better tracking and reward.
- It is not a certified safe controller.
- Its would-be gate activation rate is high, so the safety gate is not cosmetic.
- The right slide label is `RL without safety gate`, not fallback.

## Method Notes

The figures were regenerated from the latest `arrays.npz` and `summary.json` files. For each step, the output used for raw tracking is:

$$
y_k^{\rm raw}=y_{\rm system,k+1}.
$$

The physical raw setpoint is:

$$
y_{{\rm sp},k}^{\rm phys}.
$$

The raw tracking error is:

$$
e_k=y_k^{\rm raw}-y_{{\rm sp},k}^{\rm phys}.
$$

Channel RMSE is:

$$
{\rm RMSE}_i=
\sqrt{
\frac{1}{N}
\sum_{k\in\mathcal W}
e_{k,i}^2
}.
$$

The reported mean RMSE is:

$$
{\rm RMSE}_{\rm mean}
=
\frac{{\rm RMSE}_\eta+{\rm RMSE}_T}{2}.
$$

The actual gate intervention rate for safety-gated RL is:

$$
r_{\rm int}
=
\frac{1}{N}
\sum_{k\in\mathcal W}
I_{{\rm intervention},k}.
$$

For the no-safety diagnostic, actual intervention is zero by construction. The useful safety diagnostic is the would-be gate activation:

$$
r_{\rm would}
=
\frac{1}{N}
\sum_{k\in\mathcal W}
I_{{\rm diagnostic\ safety\ active},k}.
$$

## Slide-Ready Claims

Use these claims carefully:

- In the latest single run, pretrained RL with the LMPC safety gate gives the best main-method full-window raw-setpoint tracking after episode 1.
- Cold-start RL is more gate-compatible, with fewer interventions and smaller correction penalties.
- Direct LMPC gives the strongest final-tail offset, but it is slower and has larger full-window raw-setpoint RMSE.
- Pretrained RL without the safety gate tracks better, but it would trigger the safety gate on about 29% of post-episode-1 steps.
- The safety gate is necessary under this setup. It is not only a plotting or bookkeeping layer.

## Limitations

- These are single-run results, not seed-averaged results.
- Direct LMPC reward is not reward-matched to the RL runners.
- Per-episode wall-clock timing was not logged, so the post-BC runtime comparison uses the recorded mean milliseconds per step as a proxy.
- The no-safety diagnostic folder is still named `mpc_only` on disk, but the scientific label in the report is `Pretrained RL without safety gate`.
- The current figures are SVGs for report and slide drafting. If the Beamer deck cannot include SVG directly, convert the selected SVGs to PDF or PNG before adding them to the slides.

## Recommended Next Figures For Slides

Use these in the slide deck first:

- `performance_runtime_summary.svg`
- `episode_reward_rmse_trends.svg`
- `rl_authority_diagnostics.svg`
- `last_episode_tracking_primary_methods.svg`
- `pretrained_with_without_safety_gate.svg`

Keep `tail_offset_comparison.svg` as a backup or a small supporting result. It is useful when explaining why Direct LMPC should not be dismissed even though its full-window RMSE is larger.
