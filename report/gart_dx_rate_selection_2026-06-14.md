# GART dx_s Rate Selection After Disturbance Run

Run analyzed: `results/GARTLMPC/20260614_204326`

This run compared the raw GART-LMPC controller under disturbance mode with finite target-state rate bounds:

$$
|x_{s,i}(k)-x_{s,i}(k-1)| \le dx_{s,\max,i}.
$$

All three cases used raw setpoint tracking in the MPC objective, `dy_rate_scale = 2`, `input_headroom_frac = 0.01`, no `u_mid` tie-breaker, no $x_s$ smoothing, and no $y_s$ smoothing. The run itself used `eps = 1e-3`; the next runner has been changed to `eps = 1e-4`.

![GART dx-rate comparison](../results/GARTLMPC/20260614_204326/plots/comparison_tracking_target_error.png)

## Performance

| case | reward mean | output RMSE | mean output error | mean target mismatch |
|---|---:|---:|---:|---:|
| dx5 | -4.157947 | 0.370104 | 0.451414 | 0.563679 |
| dx10 | -4.157946 | 0.370104 | 0.451414 | 0.563869 |
| dx20 | -4.157946 | 0.370104 | 0.451414 | 0.563807 |

The three controllers are effectively indistinguishable in tracking and reward. The tiny numerical advantage of dx10/dx20 is not meaningful enough to justify a looser proof-relevant rate bound.

## Reliability

| case | solve rate | contraction rate | governor active | target stage |
|---|---:|---:|---:|---|
| dx5 | 1.000 | 1.000 | 0.01125 | stage2 only |
| dx10 | 1.000 | 1.000 | 0.01125 | stage2 only |
| dx20 | 1.000 | 1.000 | 0.01125 | stage2 only |

No case failed the hard contraction check in this run, and the governor only acted on 45 of 4000 steps.

## dx_s Bound Usage

| case | mean `dx_s_inf` | max `dx_s_inf` | mean `dx_s_max_inf` | max component usage |
|---|---:|---:|---:|---:|
| dx5 | 0.022427 | 0.308750 | 342.088833 | 0.003037 |
| dx10 | 0.023738 | 0.308719 | 684.177666 | 0.001519 |
| dx20 | 0.024027 | 0.308719 | 1368.355332 | 0.000759 |

The `dx_s_max_active_rate = 1.0` value means the finite bound was present in the target selector, not that it was numerically binding. Component-wise, even the tightest tested case used only about 0.304% of its own largest active component ratio. So dx5/dx10/dx20 did not materially shape the target trajectory in this run; they mainly restored a finite recursive-feasibility-style bound.

## Recommendation

Keep `dx_rate_scale = 5` as the only default option.

Reasons:

- It matches dx10/dx20 on reward, RMSE, contraction, governor activity, and target mismatch.
- It is the tightest finite $x_s$ rate bound among the tested options.
- It keeps the proof story stronger than dx10/dx20 without degrading closed-loop performance.
- It is still loose enough that the present run did not show over-filtering from the $x_s$ rate bound.

The next diagnostic run should use this single dx5 setting with `eps = 1e-4`. If the 1e-4 run breaks or causes sudden contraction/tracking artifacts, the failure is more likely from the tighter Lyapunov contraction tolerance interacting with disturbance-estimate motion than from dx5 being too restrictive.
