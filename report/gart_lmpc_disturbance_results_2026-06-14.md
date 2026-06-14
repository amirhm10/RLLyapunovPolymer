# GART-LMPC Disturbance Run Analysis

Date: 2026-06-14  
Closed-loop run analyzed: `results/GARTLMPC/20260614_003718`  
Plant mode: `disturb`

## Executive Assessment

The disturbance run confirms the same scientific conclusion as the earlier nominal run, but with an important extra detail: in disturbance mode, the mixed hard and mixed soft cases did not fail because the optimizer became infeasible. They failed even though the solver, target selector, and Lyapunov contraction flags all reported success.

The good cases were:

- `old_governed_reference`
- `gart_target_raw_objective`

The poor cases were:

- `gart_target_mixed_objective`
- `gart_target_mixed_soft`

The failure mechanism is performance-target mismatch. The mixed objective adds penalties toward the GART steady target $y_s,u_s$. In this disturbance run the GART steady target became far from the requested setpoint and was held by the governor in about 93.6% of steps. The MPC was therefore pulled toward a conservative or stale auxiliary target instead of the requested setpoint schedule.

The key point is that the Lyapunov certificate and the tracking objective are different objects. A controller can satisfy the contraction test around $x_s$ while still tracking $y_{sp}$ poorly if $y_s$ is far from $y_{sp}$.

## Run Setup

All output plots are in physical units. Target-selector error metrics such as `target_reference_error_inf_mean` are saved in the controller's scaled/deviation coordinates.

| Item | Value |
|---|---:|
| Run timestamp | `20260614_003718` |
| Plant mode | `disturb` |
| Number of tests | 5 |
| Setpoint length | 400 |
| Total saved steps | 4000 |
| `disturbance_after_step` | `False` |
| Target-only run paired with this timestamp | No |

The disturbance schedule used the same direct-run values:

| Quantity | Nominal | Multiplier | Final |
|---|---:|---:|---:|
| `Qi` | 108.0 | 0.95 | 102.6 |
| `Qs` | 459.0 | 1.05 | 481.95 |
| `hA` | 1050000.0 | 0.92 | 966000.0 |

The physical setpoints were:

| Setpoint | $\eta$ | $T$ |
|---|---:|---:|
| High | 4.5 | 324.0 |
| Low | 3.4 | 321.0 |

## Methodology

The closed-loop procedure is:

1. Generate the direct-run setpoint schedule with `set_points_len = 400`.
2. Maintain an augmented observer state `xhatdhat`, with physical state estimate plus output disturbance estimate.
3. Compute a GART target from the current reference and observer state.
4. Solve the GART-LMPC step.
5. Apply the first input move to the nonlinear polymer CSTR.
6. Update the observer from the measured output innovation.
7. Save direct-style plots and metrics.

The GART target selector first solves a closest-reachable steady-target problem. It then performs a secondary smoothing/headroom solve inside the primary-cost shell. The target is accepted only if it passes the contraction probe, unless the governor shrinks the requested target motion or holds the previous target.

The MPC objective is:

$$
J =
\sum_{k=0}^{N_p-1}
\lVert y_k-y_{sp}\rVert_{Q_{\mathrm{raw}}}^2
+
\eta_y \lVert y_k-y_s\rVert_{Q_s}^2
+
\sum_{k=0}^{N_c-1}
\eta_u \lVert u_k-u_s\rVert_{R_s}^2
+
\lVert \Delta u_k\rVert_{R_{\Delta u}}^2.
$$

The first-step Lyapunov condition is:

$$
V(x_1-x_s)
\le
\rho V(x_0-x_s)+\epsilon.
$$

For the soft case, this becomes:

$$
V(x_1-x_s)
\le
\rho V(x_0-x_s)+\epsilon+s,
\qquad s\ge 0,
$$

with `slack_penalty = 1000000.0`.

## Parameter Values

The saved run used these values:

```yaml
plant_and_schedule:
  delta_t: 0.5
  output_names: [eta, T]
  input_names: [Qc, Qm]
  steady_state_inputs_phys: [471.6, 378.0]
  input_bounds_phys:
    lower: [71.6, 78.0]
    upper: [870.0, 670.0]
  setpoints_phys:
    high: [4.5, 324.0]
    low: [3.4, 321.0]
  disturbance:
    nominal_qi: 108.0
    nominal_qs: 459.0
    nominal_ha: 1050000.0
    qi_change: 0.95
    qs_change: 1.05
    ha_change: 0.92
    final_qi: 102.6
    final_qs: 481.95
    final_ha: 966000.0
    disturbance_after_step: false

model_and_mpc:
  n_x_physical: 7
  n_disturbance_outputs: 2
  n_augmented_states: 9
  n_outputs: 2
  n_inputs: 2
  prediction_horizon: 9
  control_horizon: 3
  Q_raw_diag: [5.0, 1.0]
  Q_target_diag: [5.0, 1.0]
  R_us_diag: [1.0, 1.0]
  Rdu_diag: [1.0, 1.0]
  terminal_set_on: true
  first_step_contraction_on: true
  rho: 0.99
  eps: 0.001
  alpha_terminal_min: 1.0e-8
  slack_penalty: 1000000.0

gart_target_selector:
  disturbance_filter:
    alpha_d: 0.2
    alpha_d_slow: 0.02
    d_rate_max: [0.12283629394410155, 0.9932571501288501]
    d_min: [-4.06547624, -11.475203941124079]
    d_max: [3.399137704920187, 6.2029936]
    innovation_gate: null
    innovation_norm: inf
    freeze_on_bad_innovation: false
  input_headroom_frac: 0.03
  output_headroom_frac: 0.0
  dy_s_max: [0.19850867847075593, 0.9444864478506058]
  du_s_max: [4.6517676637274565, 2.819054808941175]
  dx_s_max:
    - 0.6596061179426039
    - 1.1120632071098469
    - 0.36339378396507316
    - 1.176074428105147
    - 0.00697335752605171
    - 0.00697335752605171
    - 0.00697335752605171
  primary_tol_abs: 1.0e-8
  primary_tol_rel: 1.0e-6
  Wy_diag: [5.0, 1.0]
  W_u_smooth_diag: [1.0, 1.0]
  W_x_smooth_diag: [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
  W_y_smooth_diag: [1.0, 1.0]
  W_u_mid_diag: [0.01, 0.01]
  require_contraction_probe: true
  contraction_margin_tol: 1.0e-8
  governor_enabled: true
  governor_grid: [1.0, 0.75, 0.5, 0.25, 0.0]
  governor_bisect_iters: 8
  solver_pref: null
```

The case-specific objective settings were:

| Case | $\eta_y$ | $\eta_u$ | Lyapunov mode |
|---|---:|---:|---|
| `gart_target_raw_objective` | 0.0 | 0.0 | hard |
| `gart_target_mixed_objective` | 0.05 | 0.05 | hard |
| `gart_target_mixed_soft` | 0.05 | 0.05 | soft |

## Closed-Loop Performance

| Case | Reward mean | $\eta$ RMSE | $T$ RMSE | Mean RMSE |
|---|---:|---:|---:|---:|
| old governed | -4.144 | 0.186 | 0.553 | 0.370 |
| GART raw | -4.145 | 0.186 | 0.553 | 0.370 |
| GART mixed hard | -19.115 | 0.370 | 1.602 | 0.986 |
| GART mixed soft | -19.174 | 0.363 | 1.701 | 1.032 |

The raw GART case again matches the old governed-reference baseline. The mixed hard and soft cases are much worse in reward and output RMSE, especially in temperature.

![Disturbance comparison](../results/GARTLMPC/20260614_003718/plots/comparison_tracking_target_error.png)

## Feasibility And Lyapunov Diagnostics

| Case | Solver success | Hard contraction | Slack max | Fallback steps |
|---|---:|---:|---:|---:|
| old governed | 1.000 | 1.000 | 0.000 | 0 |
| GART raw | 1.000 | 1.000 | 0.000 | 0 |
| GART mixed hard | 1.000 | 1.000 | 0.000 | 0 |
| GART mixed soft | 1.000 | 1.000 | 7.84e-13 | 0 |

This table is the central diagnostic. The mixed methods did not fail because of infeasible optimization, solver failure, or active soft slack. They produced poor tracking while satisfying the implemented Lyapunov condition.

![Mixed hard Lyapunov diagnostics](../results/GARTLMPC/20260614_003718/gart_target_mixed_objective/direct_style/plots/04_lyapunov_diagnostics.png)

## Target And Governor Diagnostics

| Case | Mean $\lVert y_s-y_{sp}\rVert_\infty$ | Output error mean | Governor active | Target stage |
|---|---:|---:|---:|---|
| old governed | 0.549 | 0.451 | 0.856 | governed: 4000 |
| GART raw | 0.564 | 0.451 | 0.056 | stage2: 4000 |
| GART mixed hard | 4.027 | 2.268 | 0.936 | hold: 3743 |
| GART mixed soft | 4.327 | 2.244 | 0.936 | hold: 3742 |

The failed mixed cases have a very different target-selector signature:

- The target mismatch is about 7 times larger than raw GART.
- The governor is active for about 93.6% of steps.
- The target is held for about 3743 of 4000 steps.
- The selected steady target becomes almost flat while the requested setpoint continues switching.

![Mixed hard target diagnostics](../results/GARTLMPC/20260614_003718/gart_target_mixed_objective/direct_style/plots/05_target_diagnostics.png)

## Output Evidence

The raw GART objective keeps the actual performance objective pointed at the requested setpoint. It allows the GART target to act as a Lyapunov certificate center without making the controller chase every conservative target transient.

![GART raw disturbance outputs](../results/GARTLMPC/20260614_003718/gart_target_raw_objective/direct_style/plots/fig_mpc_outputs_full.png)

The mixed hard case follows a compromise trajectory that is visibly biased away from the requested setpoint. The steady target is high and nearly frozen after the early transient.

![GART mixed hard disturbance outputs](../results/GARTLMPC/20260614_003718/gart_target_mixed_objective/direct_style/plots/fig_mpc_outputs_full.png)

The mixed soft case shows the same behavior. Since the slack is essentially zero, softening the Lyapunov condition is not the relevant change in this run.

![GART mixed soft disturbance outputs](../results/GARTLMPC/20260614_003718/gart_target_mixed_soft/direct_style/plots/fig_mpc_outputs_full.png)

## Why Mixed Hard And Mixed Soft Failed

The failure is a closed-loop coupling between the target selector and the MPC objective.

1. GART computes a certified target $y_s,u_s$.
2. Early in the disturbed run, the mixed objective starts penalizing distance from $y_s,u_s$.
3. This changes the closed-loop trajectory compared with raw GART.
4. The target selector then sees a state/reference combination where the full requested target motion is not accepted.
5. The governor holds or strongly shrinks the target.
6. The mixed objective continues to pull toward the held target.
7. The Lyapunov contraction test remains satisfied around that held target.
8. The output tracking error relative to the real setpoint grows.

The raw method avoids this loop because it does not put $y_s,u_s$ directly into the performance objective:

$$
J_{\mathrm{raw}}
=
\sum_k
\lVert y_k-y_{sp}\rVert_{Q_{\mathrm{raw}}}^2
+
\lVert \Delta u_k\rVert_{R_{\Delta u}}^2.
$$

The mixed methods add:

$$
\eta_y\lVert y_k-y_s\rVert_{Q_s}^2
+
\eta_u\lVert u_k-u_s\rVert_{R_s}^2.
$$

This is dangerous when:

$$
\lVert y_s-y_{sp}\rVert_\infty
$$

is large. In the disturbance run, that mismatch was about 4.03 for mixed hard and 4.33 for mixed soft.

## Why The Soft Case Did Not Help

The soft case was not meaningfully using slack:

- `slack_lyap_mean = 1.53e-14`
- `slack_lyap_max = 7.84e-13`
- `slack_lyap_active_steps = 0`

So the poor behavior is not caused by a too-strict Lyapunov inequality. The soft case remains almost the same method as mixed hard because the constraint is already satisfied. The issue is the target-centered terms in the objective, not the hard/soft treatment of the contraction constraint.

## Disturbance Estimator Interpretation

The disturbance model did not appear to be the dominant failure. In the mixed target diagnostics, the certified disturbance states track the estimated disturbance after the early transient. The mean disturbance target errors were:

| Case | Mean disturbance target error |
|---|---:|
| GART raw | 0.151 |
| GART mixed hard | 0.214 |
| GART mixed soft | 0.217 |

The mixed cases are somewhat worse, but the large tracking failure lines up more directly with the target mismatch and governor hold behavior.

## Proof Interpretation

This run is a useful warning for the proof. The implemented contraction statement is about the certified target:

$$
V(x^+-x_s)
\le
\rho V(x-x_s)+\epsilon.
$$

It is not automatically a proof of raw setpoint tracking:

$$
y \rightarrow y_{sp}.
$$

To connect the certificate to setpoint tracking, we need:

$$
\lVert y-y_{sp}\rVert
\le
\lVert y-y_s\rVert
+
\lVert y_s-y_{sp}\rVert.
$$

The mixed cases tried to reduce the first term, $\lVert y-y_s\rVert$, but the second term, $\lVert y_s-y_{sp}\rVert$, became large. Therefore the proof can still hold while the performance objective fails.

The raw case is currently the scientifically cleaner candidate:

- It keeps the proof target and performance target separate.
- It preserves direct setpoint tracking behavior.
- It still uses GART for the Lyapunov certification center.

## Recommended Next Experiment

The next experiment should keep `gart_target_raw_objective` as the default GART candidate. Mixed target terms should only be reintroduced behind a guard:

$$
\eta_y,\eta_u > 0
\quad \text{only if} \quad
\lVert y_s-y_{sp}\rVert_\infty \le \delta.
$$

Suggested first guard values:

| Parameter | Suggested value |
|---|---:|
| $\delta$ | 0.5 |
| $\eta_y$ when accepted | 0.01 |
| $\eta_u$ when accepted | 0.0 |

This tests whether mild output target regularization helps only when the selector target is already close to the requested reference. The input target penalty should remain off first, because $u-u_s$ can fight the input moves needed for raw tracking.

The next run should include:

- `gart_target_raw_objective`
- gated mixed output-only objective
- gated mixed output-plus-input objective only after output-only works
- target-only disturbance diagnostics with the same timestamp

The metric that must improve is raw output RMSE without increasing governor hold rate. If the governor hold rate rises above the raw GART value, the mixed terms are still interacting badly with the target selector.

## Current Conclusion

For the disturbance case, `gart_target_raw_objective` is acceptable and should be the working GART-LMPC path. The two mixed variants should be treated as failed variants in their current form.

The main reason is not solver failure and not Lyapunov infeasibility. The main reason is that the mixed objective turns a conservative certified target into a competing performance target.
