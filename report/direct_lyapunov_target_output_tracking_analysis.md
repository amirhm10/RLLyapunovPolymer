# Direct Lyapunov Target-Output Tracking Analysis

This note analyzes the latest direct frozen-output-disturbance Lyapunov MPC run:

`Data/debug_exports/direct_lyapunov_mpc_ten_scenario/20260424_162348`

That run used:

- `plant_mode = "nominal"`
- `disturbance_after_step = False`
- `set_points_len = 1500`
- `use_target_output_for_tracking = True`

The experiment asked whether the MPC stage objective should track the
constraint-aware modified output target `y_s` instead of the raw requested
setpoint `y_sp`.

## Main Finding

Tracking `y_s` made the optimization internally consistent, but it made the
external setpoint tracking worse. The controller became very good at following
the modified reference selected by the target layer, while that modified
reference was often far from the requested setpoint.

The practical conclusion is:

```text
Raw requested reference: y_sp
Constraint-aware modified reference: y_s
Controller default should track: y_sp
Diagnostics should report: y_s - y_sp and y - y_sp
```

So the notebook default has been restored to:

```python
use_target_output_for_tracking = False
```

The `True` setting should remain a diagnostic variant, not the default
controller policy.

## Latest Run Summary

Performance against the raw requested setpoint:

| Case | Reward | Output RMSE | Solver |
| --- | ---: | ---: | ---: |
| `unbounded_hard` | -36.833 | 1.208 | 0.0% |
| `bounded_hard` | -231.453 | 5.570 | 99.6% |
| `unbounded_soft` | -98.459 | 0.513 | 97.7% |
| `bounded_soft` | -313.137 | 6.504 | 99.8% |
| `bounded_hard_u_prev` | -27.658 | 0.807 | 100.0% |
| `bounded_soft_u_prev` | -27.658 | 0.807 | 100.0% |
| `bounded_hard_u_prev_1p0` | -35.139 | 1.115 | 100.0% |
| `bounded_soft_u_prev_1p0` | -35.136 | 1.115 | 100.0% |
| `bounded_hard_u_prev_10p0` | -36.607 | 1.196 | 100.0% |
| `bounded_soft_u_prev_10p0` | -36.605 | 1.196 | 100.0% |

Reference-error diagnostics in scaled-deviation infinity norm:

| Case | Mean `|y_s-y_sp|` | Mean `|y-y_sp|` | Mean `|y-y_target|` |
| --- | ---: | ---: | ---: |
| `bounded_soft_u_prev` | 2.254 | 2.263 | 0.0277 |
| `bounded_soft_u_prev_1p0` | 2.878 | 2.881 | 0.0040 |
| `bounded_soft_u_prev_10p0` | 3.022 | 3.022 | 0.0004 |
| `bounded_soft` | 6.283 | 4.714 | 3.3272 |
| `bounded_hard` | 4.544 | 3.630 | 2.4394 |

This is the failure signature. For the regularized bounded cases,
`|y-y_target|` is almost zero, but `|y_s-y_sp|` and `|y-y_sp|` are large. The
controller is doing what the modified objective asks, but the modified
reference is not the requested operating point.

## Comparison To Previous Raw-Setpoint Tracking

The previous long-setpoint analysis used the same 1500-step nominal setup but
kept the MPC stage objective on `y_sp`.

| Case | RMSE, raw `y_sp` objective | RMSE, `y_s` objective |
| --- | ---: | ---: |
| `bounded_hard` | 0.831 | 5.570 |
| `bounded_soft` | 0.963 | 6.504 |
| `bounded_soft_u_prev` | 0.844 | 0.807 |
| `bounded_soft_u_prev_1p0` | 0.605 | 1.115 |
| `bounded_soft_u_prev_10p0` | 3.303 | 1.196 |

The target-output objective improves some feasibility and smoothness metrics,
but it does that by letting the selected target drift away from `y_sp`. The
previous best case, `bounded_soft_u_prev_1p0`, became worse against the true
external setpoint: output RMSE increased from `0.605` to `1.115`, and mean
reward fell from `-9.648` to `-35.136`.

## What Happened

With `use_target_output_for_tracking = True`, the MPC stage cost becomes:

```math
\sum_i \| y_{k+i|k} - y_s(k) \|_Q^2
```

instead of:

```math
\sum_i \| y_{k+i|k} - y_{sp}(k) \|_Q^2.
```

The Lyapunov center and the stage objective then agree. This makes the
optimization cleaner, and the regularized bounded cases solve every step with
essentially no Lyapunov slack. But the target selector has an escape route: it
can choose an admissible `y_s` that is easy to certify rather than one that is
close enough to the requested `y_sp`.

For `bounded_soft_u_prev_1p0`, the last-window behavior makes this clear:

| Segment | RMSE `y-y_sp` eta | RMSE `y-y_sp` T | RMSE `y-y_s` eta | RMSE `y-y_s` T |
| --- | ---: | ---: | ---: | ---: |
| S1 high | 0.602 | 0.312 | 0.00005 | 0.0011 |
| S2 low | 0.459 | 2.286 | 0.00024 | 0.0044 |
| S3 high | 0.633 | 0.639 | 0.00008 | 0.0016 |
| S4 low | 0.437 | 2.068 | 0.00022 | 0.0038 |

The output is nearly exactly on `y_s`, but the offset from `y_sp` is persistent
and large, especially in reactor temperature during low-setpoint segments.

The target offset has the same sign and magnitude as the output offset:

| Segment | Mean `y_s-y_sp` eta | Mean `y_s-y_sp` T | Mean `y-y_sp` eta | Mean `y-y_sp` T |
| --- | ---: | ---: | ---: | ---: |
| S1 high | -0.602 | -0.310 | -0.602 | -0.311 |
| S2 low | 0.459 | 2.282 | 0.459 | 2.286 |
| S3 high | -0.633 | -0.638 | -0.633 | -0.639 |
| S4 low | 0.437 | 2.064 | 0.437 | 2.068 |

That means the tracking error is not primarily oscillation around the target.
It is target displacement: the target layer moved the reference, and the MPC
faithfully followed the moved reference.

## Findings

1. `use_target_output_for_tracking = True` is not a good default for this
   architecture.
2. The target selector cannot be treated as a replacement reference generator
   unless it has a strong, explicit penalty on `y_s - y_sp`.
3. The raw requested setpoint `y_sp` must remain visible in the MPC stage cost
   if the external metric is `y - y_sp`.
4. `y_s` is still useful as a Lyapunov center and feasibility diagnostic, but
   using it as the stage target hides poor requested-setpoint tracking.
5. The new diagnostics are valuable: `y_s - y_sp` immediately explains why the
   target-output run looked feasible but performed poorly.

## Recommendation

Keep the notebook default on raw-setpoint objective tracking:

```python
use_target_output_for_tracking = False
```

For the next controller revision, keep logging both:

- target modification: `y_s - y_sp`
- external tracking: `y - y_sp`

If target-output tracking is revisited, it should be paired with a selector
objective or constraint that directly limits `y_s - y_sp`. Without that, the
controller can solve a clean Lyapunov problem for the wrong reference.

