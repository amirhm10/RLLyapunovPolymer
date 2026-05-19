# Direct Lyapunov Three-Notebook Latest Analysis

Date: 2026-05-13

## Objective

This note analyzes the latest saved results for the three direct Lyapunov notebooks:

- `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

The main question is not only which notebook looks best. The more important question is why the RL studies can still show offset, jitter, or poor training episodes even though the direct Lyapunov safety gate is enforcing contraction.

## 1. Files inspected

### Notebooks

- [DirectLyapunovMPC_FourMethodDisturbance.ipynb](../DirectLyapunovMPC_FourMethodDisturbance.ipynb)
- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](../DirectLyapunovSafetyGateRL_ColdStart.ipynb)

### Controller and target-selection code

- [Simulation/run_rl_lyapunov.py](../Simulation/run_rl_lyapunov.py)
- [Lyapunov/direct_lyapunov_mpc.py](../Lyapunov/direct_lyapunov_mpc.py)
- [Lyapunov/lyapunov_core.py](../Lyapunov/lyapunov_core.py)
- [Lyapunov/frozen_output_disturbance_target.py](../Lyapunov/frozen_output_disturbance_target.py)
- [utils/direct_lyapunov_study.py](../utils/direct_lyapunov_study.py)

### Latest saved exports used

- Direct four-method latest complete sweep:
  [Data/debug_exports/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260511_170444/comparison_summary.json](../Data/debug_exports/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260511_170444/comparison_summary.json)
- Direct four-method latest rerun snapshot:
  [Data/debug_exports/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260512_071324/bounded_hard/summary.json](../Data/debug_exports/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260512_071324/bounded_hard/summary.json)
- Pretrained RL latest complete sweep:
  [Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313/comparison_summary.json](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313/comparison_summary.json)
- Cold-start RL latest complete sweep:
  [Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260512_071317/comparison_summary.json](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260512_071317/comparison_summary.json)

### Earlier analysis used as supporting context

- [report/direct_four_method_disturbance_settling_analysis_2026-05-11.md](./direct_four_method_disturbance_settling_analysis_2026-05-11.md)
- [report/direct_rl_last_episode_settling_analysis_2026-05-11.md](./direct_rl_last_episode_settling_analysis_2026-05-11.md)
- [report/safe_rl_implementation_summary.md](./safe_rl_implementation_summary.md)

## 2. What the current method is doing

All three notebooks use the direct output-disturbance target path rather than the standard `Lyapunov/target_selector.py` path. The selected target is recomputed from the current observer state and disturbance estimate, then the controller or RL safety gate works relative to that target.

For the direct Lyapunov gate, the physical-state error is

$$
e_x(k) = \hat x(k) - x_s(k)
$$

and the candidate action is accepted when

$$
V_{k+1}^{\mathrm{cand}} \le \rho V_k + \varepsilon_{\mathrm{lyap}}
$$

with

$$
V_k = e_x(k)^\top P_x e_x(k).
$$

This is the important point for interpretation:

- the certificate is centered on the selected steady target `(x_s, u_s)`
- it is not a direct certificate of raw setpoint tracking
- it is evaluated every step, even when no visible fallback happens

The tracking objective in the direct MPC path is separately chosen as

$$
y_{\mathrm{target},k} =
\begin{cases}
y_s(k), & \text{if target-output tracking is enabled} \\
y_{\mathrm{sp}}(k), & \text{if raw-setpoint tracking is enabled}
\end{cases}
$$

In the current saved notebook configurations:

- `DirectLyapunovMPC_FourMethodDisturbance.ipynb` uses `use_target_output_for_tracking = False`
- both RL notebooks call the direct path with `direct_tracking_use_target_output = False`
- all three notebooks use `rho_lyap = 0.99`
- the RL notebooks use:
  - no Gaussian behavior noise in warmup or BC
  - parameter-noise behavior in full RL
  - `warmup_behavior_source = "policy"` for pretrained
  - `warmup_behavior_source = "direct_lyapunov_mpc"` for cold start

So the current direct studies are still mixing:

- contraction around `(x_s, u_s)`
- tracking cost around raw `y_sp`

That structural mismatch is the main reason contraction does not automatically imply clean settling to the raw setpoint.

## 3. Mathematical interpretation

The present direct notebooks are effectively solving two different objectives at once:

1. Safety objective

$$
V_{k+1} \le \rho V_k + \varepsilon_{\mathrm{lyap}}
$$

around the selected admissible target `(x_s, u_s)`.

2. Tracking objective

$$
\min \sum_i \|y_i - y_{\mathrm{sp},i}\|_{Q_y}^2 + \|\Delta u_i\|_{R_{\Delta u}}^2
$$

because `use_target_output_for_tracking = False`.

If `y_s = y_sp` and `y_s` is stationary, those objectives are aligned.

If `y_s != y_sp` or `y_s` keeps moving, they are not aligned. In that case the direct Lyapunov gate can correctly certify contraction while the closed loop still appears to:

- settle with offset
- switch between admissibility and raw-setpoint pressure
- or oscillate because the admissible target itself is drifting

This is consistent with predictive safety-filter logic. The safety layer protects a local admissible target set. It does not guarantee that the admissible target coincides with the user's raw setpoint at every step.

## 4. Main result interpretation

### 4.1 Latest runs actually used

| Notebook | Latest saved run used for analysis | Why this run was used |
| --- | --- | --- |
| Direct four-method MPC | `20260511_170444` | This is the latest complete four-case sweep |
| Direct four-method MPC rerun check | `20260512_071324` | Latest rerun exists, but only `bounded_hard` completed |
| Pretrained RL | `20260512_071313` | Latest complete four-case sweep |
| Cold-start RL | `20260512_071317` | Latest complete four-case sweep |

Important naming caveat:

- the latest notebook code sets `anchor_weight=0.25` and `smoothness_weight=0.25`
- the saved case labels still say `...0p1`
- step tables confirm the saved runs are using `0.25`, so the labels are stale

### 4.2 Notebook-level summary

| Notebook | Best current case in latest complete run | Reward mean | Output RMSE mean | Main current issue |
| --- | --- | ---: | ---: | --- |
| Direct four-method MPC | `bounded_hard_u_prev_0p1` | -6.488 | 0.470 | raw-setpoint vs selected-target mismatch remains |
| Pretrained RL | `bounded_hard_u_prev_0p1_xs_prev_0p1` | -4.719 | 0.383 | unanchored case still jitters more than anchored cases |
| Cold-start RL | `bounded_hard_u_prev_0p1_xs_prev_0p1` | -3.621 | 0.381 | single-anchor runs can collapse at RL onset |

### 4.3 Direct four-method disturbance notebook

Latest complete four-case sweep:

| Case | Reward mean | Output RMSE mean | Solver success | Hard contraction rate |
| --- | ---: | ---: | ---: | ---: |
| `bounded_hard` | -15.892 | 0.765 | 0.910 | 0.910 |
| `bounded_hard_u_prev_0p1` | -6.488 | 0.470 | 0.998 | 0.998 |
| `bounded_hard_xs_prev_0p1` | -12.999 | 0.695 | 0.977 | 0.977 |
| `bounded_hard_u_prev_0p1_xs_prev_0p1` | -6.774 | 0.475 | 0.998 | 0.998 |

Interpretation:

- plain `bounded_hard` is still the weak baseline
- previous-input anchoring is the single biggest improvement
- state smoothness alone helps less than input anchoring
- combined anchoring is nearly as good as input anchoring alone in this saved direct MPC sweep

The direct disturbance figures still support the older structural diagnosis:

![Direct four-method disturbance target vs raw setpoint](figures/root_migrated/direct_four_method_disturbance_last_episode_target_vs_setpoint_2026-05-11.png)

The key point is not only that `bounded_hard` oscillates. The more important point is that the selected target `y_s` is not the same object as the raw setpoint `y_sp`, and in the disturbance path it can still move late in the episode.

The most recent rerun on 2026-05-12 only produced:

- `bounded_hard`

and its summary reproduces the same poor `bounded_hard` behavior rather than overturning the May 11 conclusion. So the direct four-method notebook still needs a fresh complete rerun if we want a true all-cases update under the newest notebook cell state.

### 4.4 Pretrained direct safety-gate RL notebook

Latest complete pretrained RL sweep:

| Case | Reward mean | Output RMSE mean | Verified rate | Accepted rate | Fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bounded_hard` | -6.782 | 0.417 | 0.995 | 0.935 | 0.060 |
| `bounded_hard_u_prev_0p1` | -4.712 | 0.394 | 0.999 | 0.984 | 0.014 |
| `bounded_hard_xs_prev_0p1` | -5.697 | 0.394 | 0.997 | 0.938 | 0.059 |
| `bounded_hard_u_prev_0p1_xs_prev_0p1` | -4.719 | 0.383 | 0.999 | 0.985 | 0.013 |

Interpretation:

- the combined regularized case is the best overall pretrained result
- the input-anchor-only case is essentially tied in reward and only slightly worse in RMSE
- the unanchored case still needs much more fallback intervention
- `x_s` smoothness alone improves RMSE somewhat, but it does not reduce gate intervention as effectively as input anchoring

Latest last-episode overlay:

![Latest pretrained RL last-episode overlay](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313/comparison_plots/comparison_outputs_last_episode.png)

The latest pretrained evidence says:

- the direct gate is mostly certifying rather than correcting in the anchored cases
- the remaining issue is not "the gate never activates"
- the remaining issue is that the unanchored case still carries visible second-segment jitter, while anchored cases are calmer

### 4.5 Cold-start direct safety-gate RL notebook

Latest complete cold-start RL sweep:

| Case | Reward mean | Output RMSE mean | Verified rate | Accepted rate | Fallback rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bounded_hard` | -6.434 | 0.432 | 0.994 | 0.935 | 0.059 |
| `bounded_hard_u_prev_0p1` | -362.948 | 7.493 | 0.998 | 0.981 | 0.017 |
| `bounded_hard_xs_prev_0p1` | -8.008 | 0.794 | 0.997 | 0.941 | 0.057 |
| `bounded_hard_u_prev_0p1_xs_prev_0p1` | -3.621 | 0.381 | 0.999 | 0.983 | 0.016 |

Interpretation:

- the combined case is clearly the best cold-start result
- the `u_prev`-only case is unusable as a default despite a decent final episode
- the `x_s`-only case also has a clear whole-run robustness problem
- the plain `bounded_hard` case is not catastrophic, but it still needs much more fallback and has worse RMSE than the combined case

Latest last-episode overlay:

![Latest cold-start RL last-episode overlay](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260512_071317/comparison_plots/comparison_outputs_last_episode.png)

This figure matters because it separates two failure modes:

- the last episode itself can look acceptable
- the whole training run can still be poor because a few RL-onset episodes collapse badly

That is exactly what happens in the cold-start `u_prev` case.

### 4.6 Cold-start `u_prev` collapse happens at RL onset, not in the final episode

From the saved episode table for `bounded_hard_u_prev_0p1` in the cold-start study:

| Episode | Reward mean | Fallback count | Output RMSE mean | Output-2 RMSE |
| --- | ---: | ---: | ---: | ---: |
| 31 | -43629.095 | 14 | 82.233 | 162.257 |
| 32 | -27383.577 | 54 | 65.112 | 127.627 |
| 33 | -961.011 | 16 | 12.053 | 23.262 |

These are the first full-RL episodes after the warmup plus BC stages.

So for the latest cold-start single-anchor case, the dominant issue is:

- abrupt instability when the controller leaves teacher-driven behavior and enters full RL

This is not the same failure mode as "the final evaluation episode never settles."

### 4.7 Large target residual spikes are a real warning sign

Cold-start target residual maxima:

| Case | `target_residual_total_norm_max` |
| --- | ---: |
| `bounded_hard` | 16.286 |
| `bounded_hard_u_prev_0p1` | 1189.887 |
| `bounded_hard_xs_prev_0p1` | 1062.061 |
| `bounded_hard_u_prev_0p1_xs_prev_0p1` | 13.533 |

This is strong evidence that the bad cold-start single-anchor runs are not only reward-noisy. They are hitting severe transient target-path pathologies that the combined regularization largely avoids.

## 5. Why contraction does not guarantee raw setpoint settling here

This is the central answer to the user's concern.

### 5.1 The gate is already active even when fallback is not visible

In the direct RL supervisor, the candidate action is checked every step. If it satisfies:

- input bounds
- move bounds
- Lyapunov contraction around `(x_s, u_s)`

then the gate accepts it immediately.

So "activation" is not the same as "visible correction." The current anchored RL cases have accepted rates around `0.98`, which means the safety layer is active but mostly certifying the RL action instead of overriding it.

### 5.2 The Lyapunov certificate is around `(x_s, u_s)`, not around raw `y_sp`

If the admissible target selector chooses a shifted steady target because of disturbance, bounds, or anchor regularization, then the gate may certify a move that contracts toward that shifted target even when the raw setpoint error is not yet zero.

That is not a bug in the Lyapunov test by itself. It is exactly what the current formulation asks it to do.

### 5.3 The current direct notebooks still track raw `y_sp` in the direct MPC objective

Because `use_target_output_for_tracking = False`, the fallback MPC objective still pulls toward raw `y_sp` even while the admissible steady target may be `y_s != y_sp`.

So the architecture is currently split between:

- admissibility around `y_s`
- tracking pressure around `y_sp`

This is the main structural reason for:

- offset-like settling in anchored cases
- oscillatory compromise in unanchored cases
- and user-visible confusion about why "contraction" did not force exact setpoint settling

### 5.4 `rho_lyap = 0.99` is a mild contraction requirement

With

$$
V_{k+1} \le 0.99 V_k + \varepsilon_{\mathrm{lyap}},
$$

the gate only requires about a 1 percent one-step decrease in the Lyapunov metric, and that metric lives in state space around `x_s`.

That is weaker than demanding:

- fast output settling
- exact raw setpoint convergence
- or monotone decay of output error in physical units

### 5.5 Moving targets break the user's intuition

If `x_s`, `u_s`, and `y_s` are recomputed every step and can move because of disturbance or regularization, then "contract to the target" no longer means "settle to one fixed point." It means "stay locally contractive around the current admissible target sequence."

That distinction is exactly what the saved disturbance figures are showing.

## 6. Bugs, inconsistencies, or risks found

### 6.1 Stale case names

The latest notebook code uses `anchor_weight=0.25` and `smoothness_weight=0.25`, but the saved case labels still say `0p1`.

This is a real reporting risk because it makes the current runs look like the older 0.1-weight study when they are not.

### 6.2 Latest direct-MPC rerun is incomplete

The newest direct four-method export on 2026-05-12 contains only `bounded_hard`. So there is no true latest complete four-case direct-MPC sweep under the newest notebook save.

### 6.3 The disturbance path is not the previously tuned standard selector path

These direct notebooks use the disturbance-specific target generator in `Lyapunov/frozen_output_disturbance_target.py`, not the standard `Lyapunov/target_selector.py` path that had earlier settling-oriented refinements.

So older "target-selector fixes" do not automatically solve the current direct disturbance studies.

### 6.4 Cold-start RL onset is still too abrupt for single-anchor cases

The worst latest RL failure is not final-episode non-settling. It is the episode-31 to episode-33 collapse in the cold-start `u_prev` case and the broader fragility of the `x_s`-only case.

### 6.5 Saved latest RL bundles still need better tail diagnostics

The latest debug exports provide excellent whole-run summaries and final-episode plots, but they do not yet save a compact physical-units tail table for:

- mean error to raw `y_sp`
- mean error to selected `y_s`
- target bias `y_s - y_sp`
- target variation in the final windows

That would make future settling audits much easier.

## 7. Figure and report updates made

This task adds one new report:

- [report/direct_lyapunov_three_notebooks_latest_analysis_2026-05-13.md](./direct_lyapunov_three_notebooks_latest_analysis_2026-05-13.md)

No new figures were generated in this pass. The report reuses audited figures from the saved latest exports and the earlier direct disturbance target-vs-setpoint figure.

## 8. Literature connections

No new external citations were added in this pass.

The interpretation here is consistent with the safe-RL and predictive-safety-filter framing already summarized in:

- [report/safe_rl_implementation_summary.md](./safe_rl_implementation_summary.md)

In particular:

- a predictive safety layer certifies local admissibility
- a stabilizing contraction test need not imply raw-reference optimality
- moving-target admissibility and raw-setpoint tracking can conflict when they are not explicitly aligned

## 9. Recommended next experiment

### Experiment A: align the direct tracking objective with the selected admissible target

- Purpose: test whether the main visible non-settling is caused by `y_sp` vs `y_s` mismatch
- Files: `DirectLyapunovMPC_FourMethodDisturbance.ipynb`, `DirectLyapunovSafetyGateRL_Pretrained.ipynb`, `DirectLyapunovSafetyGateRL_ColdStart.ipynb`
- Change: rerun with `use_target_output_for_tracking=True` and `direct_tracking_use_target_output=True`
- Metric: final-episode tail MAE to raw `y_sp`, tail MAE to `y_s`, fallback rate, accepted rate
- Watch for: cleaner settling to `y_s` but larger visible offset to raw `y_sp`
- Figure: output, raw setpoint, and selected target in the same tail plot
- Confirmation criterion: if jitter drops sharply while `y - y_s` becomes small, then target/reference mismatch is the dominant mechanism

### Experiment B: keep combined regularization as the default RL setting

- Purpose: remove the two latest cold-start failure modes that appear in the single-anchor cases
- Files: `utils/direct_lyapunov_study.py`, both RL notebooks
- Change: keep the combined `u_ref_weight + x_ref_weight` case as the default comparison baseline and stop treating the single-anchor cold-start cases as equally viable
- Metric: reward mean, output RMSE mean, target residual max, catastrophic-episode count
- Watch for: mild steady-state bias in exchange for better robustness
- Figure: comparison output-RMSE bar chart and episode reward trace
- Confirmation criterion: the combined case should remain best or near-best in both whole-run reward and last-episode behavior

### Experiment C: soften the BC-to-RL handoff

- Purpose: eliminate the episode-31 collapse seen in cold-start single-anchor training
- Files: `Simulation/run_rl_lyapunov.py`, `TD3Agent/agent.py`, both RL notebooks
- Change: keep a decaying teacher/BC influence for episodes 31 to 40, or ramp parameter-noise amplitude gradually instead of switching immediately into full RL updates
- Metric: episode-31 to episode-40 reward, output RMSE, fallback count, number of episodes with reward mean below `-100`
- Watch for: too much teacher influence can suppress later improvement
- Figure: episode reward with phase boundaries and fallback count per episode
- Confirmation criterion: the catastrophic episode cluster should disappear without degrading the final-episode overlay

### Experiment D: add explicit output-target smoothing inside the disturbance target generator

- Purpose: stabilize `y_s` directly instead of only regularizing `u_s` or `x_s`
- Files: `Lyapunov/frozen_output_disturbance_target.py`
- Change: add a light penalty on `y_s(k) - y_s(k-1)` or an equivalent output-target anchor
- Metric: final-window standard deviation of `y_s`, solver success rate, output RMSE mean
- Watch for: too much smoothing can increase steady-state compromise
- Figure: final-episode output vs raw setpoint vs selected target
- Confirmation criterion: the disturbance notebook should keep the current anchored robustness while reducing target drift in `bounded_hard`-like cases

## 10. Remaining uncertainty

- The latest direct-MPC notebook needs a new complete four-case rerun under the newest saved notebook state.
- The latest RL exports do not yet save a compact physical-units tail table for `y`, `y_sp`, and `y_s`.
- The cold-start single-anchor failures may be caused by a mix of actor update shock, parameter-noise interaction, observer transient, and target-selector sensitivity. The current bundles show that the failure is real, but they do not isolate that mix cleanly yet.

## 11. Files changed

- [report/direct_lyapunov_three_notebooks_latest_analysis_2026-05-13.md](./direct_lyapunov_three_notebooks_latest_analysis_2026-05-13.md)

## 12. How to verify the analysis

1. Open the three latest comparison summaries listed above and confirm the reported reward, RMSE, and rate values.
2. Open the latest RL last-episode overlays and confirm that the combined case is strongest overall in both pretrained and cold-start studies.
3. Open the cold-start `bounded_hard_u_prev_0p1` episode table and verify that episodes 31 to 33 contain the catastrophic reward and RMSE spikes.
4. Open the direct disturbance target-vs-setpoint figure and confirm that the selected target can differ from the raw setpoint in the oscillatory cases.
