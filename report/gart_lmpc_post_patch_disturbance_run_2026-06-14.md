# GART-LMPC Post-Patch Disturbance Run

Run analyzed: `results/GARTLMPC/20260614_124954`

Target-only companion run: `results/GARTTargetSelectorStudy/20260614_124509`

## Executive Assessment

This disturbance closed-loop run completed successfully for both enabled cases:

- `old_governed_reference`
- `gart_target_raw_objective`

The patched GART raw controller is numerically stable and solver-clean in this run: all 4000 MPC solves were optimal, all first-step hard Lyapunov contraction checks passed, and no Lyapunov slack was used. Tracking performance is essentially identical to the old governed-reference baseline. The mean reward difference is only about `3.90e-7` per step, which is numerical rather than scientific.

The important failure mode is not closed-loop tracking. The important failure mode is target quality: the GART target selector is still very conservative. In the GART raw case, the selected target was classified unreachable for `92.225%` of steps, and `y_s` changed on only 415 of 3999 target-to-target transitions. This is why the target diagnostic plots show `y_s` as mostly constant or stair-stepped.

## Run Setup

| Setting | Value |
|---|---:|
| Plant mode | `disturb` |
| Episodes/tests | 5 |
| Setpoint length | 400 |
| Total closed-loop steps | 4000 |
| Disturbance-after-step flag | `False` |
| Nominal `qi` | 108.0 |
| Final `qi` | 102.6 |
| Nominal `qs` | 459.0 |
| Final `qs` | 481.95 |
| Nominal `ha` | 1,050,000 |
| Final `ha` | 966,000 |

The GART controller used hard Lyapunov contraction with the raw tracking objective. Mixed objective cases were intentionally not enabled.

## Performance

| Case | Reward Mean | Reward Sum | eta RMSE | T RMSE | Mean RMSE |
|---|---:|---:|---:|---:|---:|
| old governed reference | -4.143741 | -16574.965 | 0.186273 | 0.552958 | 0.369616 |
| GART raw objective | -4.143741 | -16574.964 | 0.186273 | 0.552958 | 0.369616 |

The tracking result is a pass for the raw GART path: the new target selector and certificate did not degrade the actual plant output tracking relative to the old governed-reference baseline.

![Case comparison](../results/GARTLMPC/20260614_124954/plots/comparison_tracking_target_error.png)

![Old governed-reference output tracking](../results/GARTLMPC/20260614_124954/old_governed_reference/direct_style/plots/fig_mpc_outputs_full.png)

![GART raw output tracking](../results/GARTLMPC/20260614_124954/gart_target_raw_objective/direct_style/plots/fig_mpc_outputs_full.png)

## Certificate And Solver Reliability

| Case | Solver Success | Hard Contraction | Slack Max | Solver Status |
|---|---:|---:|---:|---|
| old governed reference | 1.000 | 1.000 | 0.0 | `optimal`: 4000 |
| GART raw objective | 1.000 | 1.000 | 0.0 | `optimal`: 4000 |

For the GART raw case, the target contraction probe also succeeded on every step. The mean good-margin convention was positive:

| Metric | GART Raw |
|---|---:|
| Governor probe success rate | 1.000 |
| Mean probe margin, good sign | 0.900295 |
| Max probe margin, good sign | 17.965303 |
| Min input headroom | 0.444000 |

![GART Lyapunov diagnostics](../results/GARTLMPC/20260614_124954/gart_target_raw_objective/direct_style/plots/04_lyapunov_diagnostics.png)

## Target Selector Diagnostics

| Case | Target Mismatch Mean | Target Mismatch Max | Governor Active | Target Stage Counts |
|---|---:|---:|---:|---|
| old governed reference | 0.548526 | 4.980628 | 0.856 | `governed_reference_target`: 4000 |
| GART raw objective | 2.966007 | 4.953267 | 0.899 | `hold_previous`: 3584, `stage2`: 416 |

GART target quality classes:

| Class | Rate |
|---|---:|
| Exact target | 0.000000 |
| Good target | 0.059500 |
| Acceptable target | 0.077750 |
| Classified unreachable | 0.922250 |
| Accepted and usable for LMPC | 1.000000 |

The selector was accepted and usable because the correctness patch now separates solve success from target quality. A target can be accepted for Lyapunov certification while still being far from the raw setpoint. That is exactly what happened here.

![GART target diagnostics](../results/GARTLMPC/20260614_124954/gart_target_raw_objective/direct_style/plots/05_target_diagnostics.png)

![GART governed-reference diagnostics](../results/GARTLMPC/20260614_124954/gart_target_raw_objective/direct_style/plots/06_governed_reference_diagnostics.png)

## Why `y_s` Is Mostly Constant

The flat `y_s` is not mainly an MPC tracking-objective effect. It is mostly target-selector behavior.

The GART selector uses a two-stage target optimization. Stage 1 chooses the closest feasible steady output target:

$$
\min_{x_s,u_s}\; \|W_y(Cx_s + C_d d_{\mathrm{cert}} - r)\|_2^2
$$

subject to the steady-state equation, input bounds, output bounds if present, and terminal tightening constraints. Stage 2 then stays inside the stage-1 primary-cost shell and chooses a smoother, more centered target:

$$
\min_{x_s,u_s}\;
\|W_{\mathrm{mid}}(u_s-u_{\mathrm{mid}})\|_2^2
+ \|W_u(u_s-u_{s,\mathrm{prev}})\|_2^2
+ \|W_x(x_s-x_{s,\mathrm{prev}})\|_2^2
+ \|W_y(y_s-y_{s,\mathrm{prev}})\|_2^2 .
$$

For this run:

| Target Parameter | Value |
|---|---:|
| `input_headroom_frac` | 0.03 |
| `Wy_diag` | `[5.0, 1.0]` |
| `W_u_smooth_diag` | `[1.0, 1.0]` |
| `W_x_smooth_diag` | `[0.01, ..., 0.01]` |
| `W_y_smooth_diag` | `[1.0, 1.0]` |
| `W_u_mid_diag` | `[0.01, 0.01]` |
| `dy_s_max` | `[0.496272, 0.383859]` |
| `du_s_max` | `[0.998000, 0.740000]` |
| `rho` | 0.99 |
| `eps` | 0.001 |

The `u_mid` term is present, but it is not the main reason `y_s` stays flat. It only pulls the steady input target away from input limits inside the stage-1 primary-cost shell. The stronger explanations are:

- The dynamic governor returned `hold_previous` on 3584 of 4000 steps.
- The previous-target smoothing terms directly penalize target movement when a fresh stage-2 target is accepted.
- The target was classified unreachable for `92.225%` of steps, so matching the raw setpoint was usually not possible under the current bounds, certified disturbance estimate, rate limits, headroom, and contraction probe.

Quantitatively, `y_s` changed on only 415 of 3999 transitions. The mean infinity-norm target move was `0.002375`, the 95th percentile was `0.001234`, and the maximum was `0.383859`.

## How The MPC Penalized The Target

For `gart_target_raw_objective`, the effective target penalties were disabled:

| MPC Parameter | Value |
|---|---:|
| `Q_raw_diag` | `[5.0, 1.0]` |
| `Q_target_diag` | `[5.0, 1.0]` |
| `R_us_diag` | `[1.0, 1.0]` |
| `Rdu_diag` | `[1.0, 1.0]` |
| `eta_y` | 0.0 |
| `eta_u` | 0.0 |
| `target_term_gate_enabled` | `True` |
| `target_term_gate_delta_y` | 0.5 |
| `target_term_gate_min_alpha` | 0.5 |
| `target_term_gate_disable_on_hold` | `True` |

So the raw GART closed-loop objective was effectively:

$$
\sum_{k=1}^{N_p}\|y_{k}-y_{sp}\|_{Q_{\mathrm{raw}}}^2
+ \sum_{k=0}^{N_c-1}\|\Delta u_k\|_{R_{\Delta u}}^2 ,
$$

with Lyapunov terminal and first-step contraction constraints centered on `x_s`. It did not add:

$$
\eta_y\|y_k-y_s\|_{Q_{\mathrm{target}}}^2
\quad \text{or} \quad
\eta_u\|u_k-u_s\|_{R_{u_s}}^2 ,
$$

because `eta_y = eta_u = 0`.

The target-term gate also would have disabled those terms on most steps even if nonzero weights were requested. The logged gate reasons were:

| Gate Reason | Steps |
|---|---:|
| `target_setpoint_mismatch,governor_alpha_too_small,hold_previous` | 3584 |
| `ok` | 311 |
| `target_setpoint_mismatch` | 96 |
| `target_setpoint_mismatch,governor_alpha_too_small` | 9 |

This is the main reason the GART raw case tracks almost exactly like the old baseline even while `y_s` is conservative.

## Why Mixed Was Not Run

The mixed objective is intentionally disabled by default because it would add target-centered performance terms:

$$
\eta_y\|y_k-y_s\|_{Q_{\mathrm{target}}}^2
+ \eta_u\|u_k-u_s\|_{R_{u_s}}^2 .
$$

That is only desirable when `y_s` is close to the raw setpoint and moving with the reference. In this run, `y_s` was mostly held and often far from the desired setpoint. Running mixed under these conditions would ask the MPC to split effort between the real setpoint and a conservative certified target. That was the failure mode seen in the previous nominal and disturbance reports.

The current evidence says:

- Raw objective is safe and tracks well.
- The target certificate path is feasible and stable.
- The target selector is not yet high-quality enough to be used as a performance target.

Mixed should be re-enabled only after the observer-replay target ablation improves the accepted target mismatch and reduces the hold rate, or after the gated mixed objective is tested with strict activation conditions.

## Target-Only Companion Run

The latest target-only companion run was synthetic:

| Metric | Value |
|---|---:|
| Steps | 40 |
| Solve success | 1.000 |
| Accepted | 1.000 |
| Usable for LMPC | 1.000 |
| Mean target error | 3.015127 |
| 95th percentile target error | 3.353419 |
| Governor active rate | 0.250 |
| Hold previous rate | 0.000 |
| Unreachable rate | 1.000 |

This run confirms that the target selector is conservative even in the synthetic diagnostic. It should not be treated as the final closed-loop distribution because it does not replay the real observer trajectory.

## What Was Stored

The closed-loop run stored:

- `summary.json`: run-level metrics and artifact paths.
- `comparison.csv`: compact cross-case metric table.
- Per-case `config.json`: target, MPC, and run configuration.
- Per-case `steps.csv`: step-level target acceptance, governor, contraction, reward, and gate diagnostics.
- Per-case `payload.pickle`: full rollout payload including plant outputs, inputs, setpoints, observer states, direct solver diagnostics, and `target_info_storage`.
- Per-case `arrays.npz`: compact arrays for outputs, applied inputs, setpoints, rewards, and observer states.
- Per-case `direct_style_bundle.pickle`: direct-run compatible export bundle.
- Per-case direct-style plots for output tracking, input trajectories, state-target error, Lyapunov diagnostics, target diagnostics, governed-reference diagnostics, tail-window summary, and reward summary.

## Next Experiment

The next useful experiment is not mixed closed-loop yet. It is target-quality ablation with observer replay:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe experiments/run_gart_target_selector_study.py --full --confirm-full --target-ablation --no-closed-loop --mode disturb --n-tests 5 --set-points-len 400
```

The goal is to identify which target restriction causes most of the hold behavior:

- `dx_s_max` rate limits
- `du_s_max` rate limits
- `dy_s_max` rate limits
- `input_headroom_frac`
- contraction probe acceptance
- certified disturbance filtering

After a target configuration gives low target mismatch and low hold rate on observer replay, run closed-loop raw again, then test a gated mixed objective.
