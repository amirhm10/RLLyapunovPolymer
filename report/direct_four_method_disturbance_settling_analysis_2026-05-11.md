# Direct Four-Method Disturbance Settling Analysis

## Question

Why does `DirectLyapunovMPC_FourMethodDisturbance.ipynb` still show the pattern

1. reaching near the setpoint,
2. then becoming oscillatory instead of settling,

even though the target-selector settling issue had previously been improved elsewhere?

## Short Answer

The latest results show that this is not primarily a replay of the old standard target-selector issue. The main reasons are:

1. This notebook is not using the same selector path that was tuned before.
2. The notebook still sets `use_target_output_for_tracking = False`, so the tracking MPC stage follows raw `$y_{\mathrm{sp}}$` while the admissible target selector may move `$y_s$`.
3. In the latest disturbance results, `$y_s$` itself is still moving near the end of the final episode for two of the three exported cases, so the closed loop is not trying to settle to one fixed admissible target.

In other words, the oscillatory symptom is reappearing because the disturbance-case target-generation path and the tracking reference path are still structurally misaligned.

## Files Checked

- `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
- `utils/direct_lyapunov_study.py`
- `Lyapunov/direct_lyapunov_mpc.py`
- `Lyapunov/frozen_output_disturbance_target.py`
- `Data/debug_exports/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260511_122254/...`
- `Data/debug_exports/direct_lyapunov_mpc_bounded_three_scenario_two_setpoint_nominal/20260511_012052/...`

## Method Reconstruction

### Tracking reference used by the direct Lyapunov controller

Inside `Lyapunov/direct_lyapunov_mpc.py`, the stage-cost tracking reference is chosen as

$$
y_{\text{target},k} =
\begin{cases}
y_{s,k}, & \text{if use\_target\_output\_for\_tracking = True} \\
y_{\mathrm{sp},k}, & \text{if use\_target\_output\_for\_tracking = False}
\end{cases}
$$

The notebook currently sets:

```python
use_target_output_for_tracking = False
```

so the controller is still tracking raw setpoint, not the bounded admissible target output.

### Target-generation path in this notebook

The latest disturbance export shows that the dominant target stage is:

- `frozen_output_disturbance_bounded_ls`

That path comes from `Lyapunov/frozen_output_disturbance_target.py`, not from the standard `Lyapunov/target_selector.py` path that had earlier settling-oriented refinements.

So the previous settling fix does not automatically carry over here.

### Disturbance target problem actually being solved

For the disturbed direct notebook, the target stage is effectively a bounded steady-state least-squares problem with optional regularization toward previous input or previous state:

$$
\min_{x_s,u_s}
\|r_{\text{ss}}(x_s,u_s; \hat d_k, y_{\mathrm{sp},k})\|^2
+
\|W_u^{1/2}(u_s-u_{\mathrm{ref}})\|^2
+
\|W_x^{1/2}(x_s-x_{\mathrm{ref}})\|^2
$$

subject to input bounds on `$u_s$`.

This notebook's three exported cases are:

- `bounded_hard`
- `bounded_hard_u_prev_0p1`
- `bounded_hard_xs_prev_0p1`

The latest folder does **not** contain the combined `bounded_hard_u_prev_0p1_xs_prev_0p1` case, so the latest four-method sweep appears incomplete.

## Figure

The figure below shows the final episode in physical units for the latest disturbance export. The dashed black line is the tracking reference used by the MPC stage, and the orange dash-dot line is the selected target `$y_s$`.

![Final-episode output, raw setpoint, and selected target](figures/root_migrated/direct_four_method_disturbance_last_episode_target_vs_setpoint_2026-05-11.png)

## Quantitative Findings

### Latest disturbance run: final-episode tail metrics

All metrics below are computed in physical units from the latest disturbance export:

| Case | Seg-2 MAE to tracking ref `$y_1$` | Seg-2 MAE to target `$y_1$` | Tail MAE to tracking ref `$y_2$` | Tail MAE to target `$y_2$` | Last-100 std of target `$y_{s,1}$` | Last-100 std of target `$y_{s,2}$` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bounded_hard` | 0.437 | 0.130 | 1.035 | 1.233 | 0.448 | 2.952 |
| `bounded_hard_u_prev_0p1` | 0.161 | 0.052 | 0.660 | 0.584 | 0.117 | 1.309 |
| `bounded_hard_xs_prev_0p1` | 0.436 | 0.116 | 0.814 | 1.211 | 0.341 | 4.303 |

Interpretation:

- The previous-input anchor case is the only latest disturbance case where the target itself becomes comparatively calm near the end.
- `bounded_hard` and `bounded_hard_xs_prev_0p1` still have large last-100 target variation, especially in temperature.
- For `$y_1$`, the plant output is much closer to the selected target than to the raw tracking reference in the second half of the final episode.

### Raw tracking reference versus selected target are materially different

Average absolute mismatch over the final 800 steps:

| Case | Mean `$\|y_s - y_{\mathrm{track}}\|$` in `$y_1$` | Mean `$\|y_s - y_{\mathrm{track}}\|$` in `$y_2$` |
| --- | ---: | ---: |
| `bounded_hard` | 0.236 | 1.558 |
| `bounded_hard_u_prev_0p1` | 0.176 | 0.879 |
| `bounded_hard_xs_prev_0p1` | 0.259 | 1.534 |

So the controller is being asked to track `$y_{\mathrm{sp}}$` even when the bounded target generator is selecting a materially different admissible output.

### Tail solver fallback is present, but it is not the dominant mechanism

Final 800-step method counts in the latest disturbance run:

| Case | `direct_lyapunov_mpc` | `solver_fail_hold_prev` |
| --- | ---: | ---: |
| `bounded_hard` | 762 | 38 |
| `bounded_hard_u_prev_0p1` | 789 | 11 |
| `bounded_hard_xs_prev_0p1` | 764 | 36 |

This matters, but these fallback counts are too small to explain the entire oscillatory pattern by themselves. The larger issue is that the selected target is still moving and the stage-cost reference is still the raw setpoint.

### This is not only a disturbance-case issue

The latest nominal export shows the same structural pattern:

- `use_target_output_for_tracking = False`
- bounded target and tracking reference remain different in the tail
- the previous-input anchor case is again the most stable

So the problem is broader than one disturbed run. Disturbance worsens it, but the structural mismatch already exists in the nominal case too.

## Why the Earlier Settling Fix Did Not Solve This Notebook

The earlier fix was associated with the standard target-selector objective path. This notebook is different in two important ways:

1. It uses `solve_output_disturbance_target(...)` from `Lyapunov/frozen_output_disturbance_target.py`.
2. It still tracks raw setpoint because `use_target_output_for_tracking = False`.

That means we can improve selector smoothing elsewhere and still see oscillatory behavior here, because:

- the disturbance target stage is a different optimization problem,
- the selected target can keep moving near the end,
- the controller objective is not centered on that selected target anyway.

## Most Likely Mechanism In The Latest Results

The latest disturbance behavior is most consistent with this chain:

1. The bounded disturbance target LS stage shifts the admissible target away from raw setpoint.
2. That target continues moving, especially in `bounded_hard` and `bounded_hard_xs_prev_0p1`.
3. The tracking MPC stage still chases raw setpoint instead of the selected target.
4. The controller then repeatedly trades off between feasibility, contraction, and a moving mismatch between `$y_{\mathrm{sp}}$` and `$y_s$`.
5. Small amounts of solver fallback add roughness, but they are secondary.

## Practical Implications

The latest results do **not** support the claim that the direct disturbance notebook has a settled target and only a bad actuator response. For two of the three exported latest cases, the target itself is still visibly unsettled near the end.

The strongest evidence in the latest sweep is that:

- `u_ref_weight = 0.1` helps meaningfully
- `x_ref_weight = 0.1` alone is not enough here
- the raw-setpoint tracking choice remains a major structural cause of the oscillatory symptom

## Recommended Next Fixes

Priority order:

1. Re-run this notebook with `use_target_output_for_tracking = True`.
2. Add the combined `u_ref_weight + x_ref_weight` case back into the disturbance sweep, since it is missing from the latest export folder.
3. If the target still moves too much, add an explicit output-target settling/smoothing term to the `frozen_output_disturbance_bounded_ls` path, not only to the standard selector path.
4. Re-check whether the observer disturbance estimate is drifting enough late in the episode to keep moving the steady-state target.

## Bottom Line

The latest oscillatory behavior in `DirectLyapunovMPC_FourMethodDisturbance.ipynb` is real, but it is not evidence that the old selector fix simply "stopped working." The deeper issue is that this notebook is using a different disturbance-target optimization path and is still tracking raw `$y_{\mathrm{sp}}$` instead of the admissible bounded target `$y_s$`. In the latest disturbance export, `$y_s$` still moves substantially near the end for two of the three cases, which is enough to recreate the non-settling appearance.
