# GART-LMPC Latest Result Analysis

Date: 2026-06-14  
Closed-loop run analyzed: `results/GARTLMPC/20260613_235051`  
Target-only run analyzed: `results/GARTTargetSelectorStudy/20260613_235051`

## Executive Assessment

The latest saved GART run shows a clean split:

- `old_governed_reference` did not fail.
- `gart_target_raw_objective` did not fail and is effectively indistinguishable from the old governed-reference baseline.
- `gart_target_mixed_objective` failed badly.
- `gart_target_mixed_soft` also failed badly, and the soft Lyapunov slack did not rescue the method.

The important diagnosis is that GART itself did not fail when it was used only as the target selector and Lyapunov certification center. The failure appeared when the MPC objective was also regularized toward the GART target $y_s,u_s$. In this run the GART target was frequently far from the raw setpoint and was held by the governor for most of the trajectory. Once the mixed objective gave that held target direct authority in the cost, the controller was pulled away from the actual setpoint-tracking task and the closed loop became unstable-looking in output space.

This specific run was nominal, not disturbance-applied, according to `summary.json`:

- `plant_mode = nominal`
- `n_tests = 5`
- `set_points_len = 400`
- `n_steps = 4000`

The saved GART case summaries still contain final `qi/qs/ha` schedule values, but in this nominal run those schedules are diagnostic bookkeeping rather than applied plant disturbances.

## Methods Compared

The four closed-loop cases were:

| Case | Target selector | MPC performance objective | Lyapunov mode |
|---|---|---|---|
| `old_governed_reference` | Previous governed reference | Tracks governed/raw reference | Hard |
| `gart_target_raw_objective` | GART target | Tracks raw setpoint $y_{sp}$ | Hard, centered at GART target |
| `gart_target_mixed_objective` | GART target | Tracks raw $y_{sp}$ plus target-centered $y_s,u_s$ terms | Hard |
| `gart_target_mixed_soft` | GART target | Same mixed objective | Soft Lyapunov slack |

The raw GART objective keeps the performance target as the real requested setpoint:

$$
J_{\mathrm{raw}}
\approx
\sum_k \lVert y_k - y_{sp,k} \rVert_Q^2
+ \lVert \Delta u_k \rVert_R^2.
$$

The mixed GART objective adds target-centered regularization:

$$
J_{\mathrm{mixed}}
\approx
J_{\mathrm{raw}}
+ \sum_k \lVert y_k - y_{s,k} \rVert_{Q_s}^2
+ \lVert u_k - u_{s,k} \rVert_{R_s}^2.
$$

That addition is only safe when $y_s$ is a good nearby proxy for $y_{sp}$. The decomposition is:

$$
\lVert y - y_{sp} \rVert
\le
\lVert y - y_s \rVert
+ \lVert y_s - y_{sp} \rVert.
$$

In the failed cases, the second term was large. So making $\lVert y-y_s\rVert$ small no longer implied good setpoint tracking.

## Closed-Loop Performance

| Case | Reward mean | $y_1$ RMSE | $y_2$ RMSE | Mean RMSE |
|---|---:|---:|---:|---:|
| old governed | -4.128 | 0.187 | 0.553 | 0.370 |
| GART raw | -4.128 | 0.187 | 0.553 | 0.370 |
| GART mixed hard | -2748.022 | 1.902 | 38.067 | 19.984 |
| GART mixed soft | -3248.075 | 1.946 | 41.750 | 21.848 |

The reward and RMSE numbers show that `gart_target_raw_objective` preserved the old controller behavior. The two mixed methods were not small degradations; they were full closed-loop failures, especially in the temperature-like output.

![Closed-loop comparison](../results/GARTLMPC/20260613_235051/plots/comparison_tracking_target_error.png)

## Cases That Did Not Fail

### Old Governed Reference

The old governed-reference baseline tracked the setpoint schedule with the expected transient overshoot at setpoint changes, but it remained bounded and solver-feasible over all 4000 steps.

![Old governed reference outputs](../results/GARTLMPC/20260613_235051/old_governed_reference/direct_style/plots/fig_mpc_outputs_full.png)

### GART Raw Objective

The raw GART case produced almost exactly the same output behavior as the old governed-reference baseline. This is the strongest positive result from the run: replacing the certification/target-selector center with GART did not break the controller as long as the MPC objective still tracked the raw setpoint.

![GART raw objective outputs](../results/GARTLMPC/20260613_235051/gart_target_raw_objective/direct_style/plots/fig_mpc_outputs_full.png)

| Case | Solver success | Hard contraction | Solver fallback | Target stage |
|---|---:|---:|---:|---|
| old governed | 1.000 | 1.000 | 0 / 4000 | governed target: 4000 |
| GART raw | 1.000 | 1.000 | 0 / 4000 | stage2: 4000 |

For `gart_target_raw_objective`, the GART target is doing the certification job while the optimizer still chases $y_{sp}$. That is why this case answers a different question from the old bounded selector: the performance objective is the same kind of raw tracking, but the steady target is now produced by the certified GART target solver instead of the previous bounded/governed-reference logic.

## Cases That Failed

### GART Mixed Hard

The hard mixed method failed because the target-centered terms became harmful. The target selector mostly stopped moving and the MPC was then regularized toward a stale or conservative target rather than the setpoint schedule.

![GART mixed hard outputs](../results/GARTLMPC/20260613_235051/gart_target_mixed_objective/direct_style/plots/fig_mpc_outputs_full.png)

### GART Mixed Soft

The soft version did not solve the problem. It relaxed the Lyapunov contraction through slack, but the bad target-centered pull remained in the performance objective. The result was even worse reward and output RMSE.

![GART mixed soft outputs](../results/GARTLMPC/20260613_235051/gart_target_mixed_soft/direct_style/plots/fig_mpc_outputs_full.png)

| Case | Solver success | Hard contraction | Slack max | Fallback steps |
|---|---:|---:|---:|---:|
| GART mixed hard | 0.881 | 0.881 | 0.000 | 475 |
| GART mixed soft | 0.799 | 0.799 | 251210.167 | 803 |

The soft method had 803 solver-failure hold-previous actions and a maximum Lyapunov slack above 251k. That means the soft mode was relaxing a badly stressed optimization, not restoring a meaningful stabilizing behavior.

## Target And Governor Diagnostics

| Case | Mean $\lVert y_s-y_{sp}\rVert_\infty$ | Mean $\lVert y-y_{sp}\rVert_\infty$ | Governor active | Dominant target stage |
|---|---:|---:|---:|---|
| old governed | 0.524 | 0.418 | 0.610 | governed reference |
| GART raw | 0.539 | 0.418 | 0.056 | stage2 |
| GART mixed hard | 3.959 | 32.082 | 0.981 | hold previous |
| GART mixed soft | 3.959 | 34.315 | 0.981 | hold previous |

The failed mixed cases had the same core signature:

- The target mismatch $\lVert y_s-y_{sp}\rVert_\infty$ jumped to about 3.96.
- The governor was active in about 98.1% of steps.
- The target stage was `hold_previous` for 3924 of 4000 steps.

This is the clearest failure mechanism in the logs. The mixed objective was not tracking a reachable moving target; it was being attracted to a target that was mostly frozen.

![GART mixed target diagnostics](../results/GARTLMPC/20260613_235051/gart_target_mixed_objective/direct_style/plots/05_target_diagnostics.png)

The Lyapunov diagnostics support the same interpretation. The failed hard mixed run repeatedly hit contraction stress and solver infeasibility instead of maintaining the clean feasible pattern seen in the raw case.

![GART mixed Lyapunov diagnostics](../results/GARTLMPC/20260613_235051/gart_target_mixed_objective/direct_style/plots/04_lyapunov_diagnostics.png)

## Target-Only Study

The paired target-only study also shows that the GART target solver was conservative before the closed-loop objective was involved:

| Metric | Value |
|---|---:|
| Steps | 4000 |
| Target success rate | 1.000 |
| Mean target error, infinity norm | 3.011 |
| 95th percentile target error | 3.308 |
| Governor active rate | 0.056 |
| Hold-previous rate | 0.000 |
| Unreachable rate | 1.000 |

This target-only result should be interpreted carefully. It does not mean the closed loop must fail, because the raw GART controller did not fail. It does mean the target selector is frequently declaring the requested references unreachable under its current steady-state, rate, headroom, and contraction-probe restrictions. The mixed objective becomes unsafe when it treats this conservative auxiliary target as a performance target.

![Target-only target vs setpoint](../results/GARTTargetSelectorStudy/20260613_235051/plots/target_vs_setpoint.png)

## Why The Technique Failed

The failed technique is specifically the mixed GART objective, not the whole GART path.

The failure chain is:

1. The GART target solver often produces a target that is not close enough to the raw requested setpoint.
2. In the mixed closed-loop cases, the governor then holds the previous target for most steps.
3. The MPC cost penalizes deviation from that held target through $y-y_s$ and $u-u_s$ terms.
4. The optimizer is pulled away from the actual setpoint schedule.
5. The plant outputs show large oscillations and temperature spikes.
6. Feasibility degrades, causing hold-previous fallback actions.
7. Soft Lyapunov slack relaxes the constraint but does not remove the wrong target pull, so performance remains failed.

This is why `gart_target_raw_objective` worked while `gart_target_mixed_objective` failed. Raw GART uses the GART target for certification but keeps the actual performance goal as $y_{sp}$. Mixed GART lets the conservative auxiliary target compete with the actual tracking task.

## What Did Not Fail

The useful conclusion is that the GART target-selector path is still viable in a limited role:

- GART target for Lyapunov centering: viable in this run.
- Raw setpoint tracking objective: viable in this run.
- GART target as an additional performance attractor: not viable with the current tuning.
- Soft Lyapunov relaxation: not sufficient when the objective is pulling toward the wrong target.

## Recommended Next Fix

The next experiment should keep the raw GART case as the reference and disable the mixed target-centered terms by default. Then test one change at a time:

1. Add a gate on mixed terms: only enable $y-y_s$ and $u-u_s$ penalties when $\lVert y_s-y_{sp}\rVert_\infty$ is below a small threshold.
2. Reduce or remove the $u-u_s$ penalty first, because it can fight the input moves needed for raw setpoint tracking.
3. Log a per-step decomposition of tracking error:

$$
e_{\mathrm{raw}} = y-y_{sp}, \qquad
e_{\mathrm{target}} = y-y_s, \qquad
e_{\mathrm{selector}} = y_s-y_{sp}.
$$

4. Run the same 5-test, 400-step nominal case first, then the disturbance case after the raw/mixed behavior is clean.

The current evidence supports using `gart_target_raw_objective` as the working GART candidate and treating the two mixed methods as failed experimental variants.
