# Direct Lyapunov Three-Method Contraction-Factor Sensitivity Report

This report replaces the older two-case settling note with the latest
three-method nominal single-setpoint study from `2026-05-01`.

The main new finding is that the Lyapunov contraction factor is not a minor
detail. It is a first-order design parameter for the direct formulation. Across
the latest sweeps, changing `rho_lyap` from `0.99` to `0.95` changes the same
controller family from near-exact nominal tracking to sustained fallback-target
operation and repeated first-step contraction failures.

## Objective

The updated question is:

How do the three direct bounded-hard methods change when the same nominal
single-setpoint run is repeated with different Lyapunov contraction factors?

The three methods are:

1. `bounded_hard`
2. `bounded_hard_u_prev_0p1`
3. `bounded_hard_xs_prev_0p1`

All four selected runs use:

- one physical setpoint: `[4.5, 324.0]`
- nominal plant
- one episode
- 2000 control steps
- raw-setpoint tracking in the online MPC objective
- bounded target construction
- hard Lyapunov constraint with zero slack activation in the saved results

## Bundles Used

Representative bundles:

| `rho_lyap` | Bundle |
| --- | --- |
| `0.95` | `Data/debug_exports/direct_lyapunov_mpc_bounded_three_scenario_single_setpoint_nominal/20260501_001425` |
| `0.98` | `Data/debug_exports/direct_lyapunov_mpc_bounded_three_scenario_single_setpoint_nominal/20260501_003638` |
| `0.985` | `Data/debug_exports/direct_lyapunov_mpc_bounded_three_scenario_single_setpoint_nominal/20260501_002805` |
| `0.99` | `Data/debug_exports/direct_lyapunov_mpc_bounded_three_scenario_single_setpoint_nominal/20260501_001948` |

Duplicate note:

- `20260501_000956` is a duplicate export of the `rho_lyap = 0.98` run. The
  metrics match `20260501_003638`, so the later bundle is used as the
  representative artifact.

The `rho_lyap` labels were verified directly from the saved step tables. For the
first saved step in each run,

$$
V_{\mathrm{bound},k} = \rho_{\mathrm{lyap}} V_k
$$

to numerical precision, so `V_bound / V_k` identifies the contraction factor
without relying on notebook memory or filenames.

## Why This Parameter Matters

The direct controller enforces first-step contraction around the current steady
target:

$$
V(x_{1|k} - x_s) \le \rho_{\mathrm{lyap}} V(\hat x_k - x_s).
$$

Lower `rho_lyap` means stricter contraction. In this sweep:

- `rho_lyap = 0.95` is the most restrictive run.
- `rho_lyap = 0.99` is the least restrictive run.

Because the saved runs have zero Lyapunov slack activation, the differences
below are not caused by slack softening. They come from the actual interaction
between the target construction and the hard one-step contraction requirement.

## Executive Findings

1. `rho_lyap` must be reported alongside the direct-controller result. Without
   it, the performance story is incomplete.
2. `rho_lyap = 0.95` is too aggressive for this nominal direct bounded setup.
   All three methods spend almost the entire episode in the bounded least-squares
   fallback target stage, and the output RMSE degrades sharply.
3. `rho_lyap = 0.99` is the strongest nominal operating point in this sweep.
   All three methods become nearly indistinguishable, with output RMSE near
   `0.11` and solver success at or above `99.85%`.
4. `rho_lyap = 0.985` gives the clearest method separation. The `x_s`-smoothing
   method is best there, while the input-anchor method degrades substantially.
5. `rho_lyap = 0.98` still separates the methods, but in a different way:
   the input-anchor method remains strong while plain `bounded_hard` and
   `x_s` smoothing are noticeably worse.

## Method 1: `bounded_hard`

### Results

| `rho_lyap` | Reward mean | Output RMSE mean | Solver success | Hard contraction | Violation steps | Bounded-LS steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.95` | -46.759 | 1.481 | 90.25% | 90.20% | 196 | 2000 |
| `0.98` | -11.801 | 0.857 | 97.95% | 97.95% | 41 | 1152 |
| `0.985` | -2.948 | 0.396 | 96.50% | 96.50% | 70 | 1263 |
| `0.99` | -0.362 | 0.113 | 99.85% | 99.85% | 3 | 221 |

Lyapunov and target notes:

- The strict `0.95` run is effectively always in bounded-LS target mode
  (`2000/2000` steps), so the method never reaches a clean exact-target regime.
- The contraction margin minimum improves from `-132.46` at `rho = 0.95` to
  `-1.78` at `rho = 0.99`.
- Plain `bounded_hard` benefits the most from relaxing `rho_lyap`. Its nominal
  weakness is not intrinsic; it is strongly tied to over-restrictive contraction.

![Bounded hard contraction ratio across rho](figures/2026-05-01_direct_three_method_rho_sensitivity/bounded_hard_contraction_ratio_by_rho.svg)

Interpretation:

- At `rho = 0.95`, the method is over-constrained and performs worst in the
  three-method set.
- At `rho = 0.99`, it becomes fully competitive and nearly tied with the other
  two methods.
- So plain `bounded_hard` is highly sensitive to contraction-factor choice.

## Method 2: `bounded_hard_u_prev_0p1`

### Results

| `rho_lyap` | Reward mean | Output RMSE mean | Solver success | Hard contraction | Violation steps | Bounded-LS steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.95` | -10.659 | 0.726 | 95.85% | 95.80% | 84 | 1986 |
| `0.98` | -0.467 | 0.132 | 100.00% | 100.00% | 0 | 311 |
| `0.985` | -9.880 | 0.766 | 98.60% | 98.60% | 28 | 1570 |
| `0.99` | -0.370 | 0.110 | 100.00% | 100.00% | 0 | 221 |

Lyapunov and target notes:

- This is the most robust method at `rho = 0.98`. It is the only case in that
  run with both perfect solver success and zero contraction violations.
- The method is not uniformly monotone in `rho`. It performs well at `0.98` and
  `0.99`, but degrades strongly at `0.985`.
- The target-input anchoring does reduce `||u_s-u_{\mathrm{ref}}||`, but at
  `rho = 0.985` it still leaves the run in bounded-LS target mode for `1570`
  steps, which is far too long for good nominal tracking.

![Bounded hard plus input anchor contraction ratio across rho](figures/2026-05-01_direct_three_method_rho_sensitivity/bounded_hard_u_prev_0p1_contraction_ratio_by_rho.svg)

Interpretation:

- This method is best at `rho = 0.95` and `rho = 0.98`.
- It is also marginally best on nominal RMSE at `rho = 0.99`, although the
  difference there is very small.
- The surprising point is the `rho = 0.985` regression. So the input-anchor
  method is not just sensitive to strict versus loose contraction; it is also
  sensitive to how the contraction factor interacts with the steady-target move
  regularization.

## Method 3: `bounded_hard_xs_prev_0p1`

### Results

| `rho_lyap` | Reward mean | Output RMSE mean | Solver success | Hard contraction | Violation steps | Bounded-LS steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.95` | -28.395 | 1.195 | 95.00% | 94.95% | 101 | 1999 |
| `0.98` | -2.002 | 0.277 | 99.85% | 99.85% | 3 | 510 |
| `0.985` | -0.448 | 0.127 | 99.90% | 99.90% | 2 | 201 |
| `0.99` | -0.362 | 0.113 | 99.85% | 99.85% | 3 | 227 |

Lyapunov and target notes:

- This method collapses under `rho = 0.95`. It is almost always in bounded-LS
  mode (`1999` steps) and the mean target-reference mismatch rises to `2.312`.
- Once `rho` is relaxed to `0.985` or `0.99`, the `x_s`-smoothing method
  becomes one of the two best performers in the entire sweep.
- The best single run for this method is `rho = 0.985`, not `0.99`. That is the
  clearest evidence that the state-smoothing term is helpful, but only when the
  contraction factor is not too strict.

![Bounded hard plus x_s smoothing contraction ratio across rho](figures/2026-05-01_direct_three_method_rho_sensitivity/bounded_hard_xs_prev_0p1_contraction_ratio_by_rho.svg)

Interpretation:

- `x_s` smoothing is the most sensitive method at the strict end of the sweep.
- But it is also the method that gains the most at `rho = 0.985`.
- So the correct statement is not that `x_s` smoothing is universally better or
  worse. It is conditionally strong when the contraction factor is chosen well.

## Cross-Method Comparison

The cross-rho comparison is summarized below.

![Method metrics by rho](figures/2026-05-01_direct_three_method_rho_sensitivity/method_metrics_by_rho.svg)

Best method by contraction factor:

| `rho_lyap` | Best method on output RMSE | Practical interpretation |
| --- | --- | --- |
| `0.95` | `bounded_hard_u_prev_0p1` | The input anchor is the least damaged by the strict contraction, but all three methods are still poor. |
| `0.98` | `bounded_hard_u_prev_0p1` | Input anchoring gives the cleanest feasible nominal behavior at this contraction level. |
| `0.985` | `bounded_hard_xs_prev_0p1` | `x_s` smoothing is the best compromise between target stability and tracking quality here. |
| `0.99` | effectively tied; smallest RMSE is `bounded_hard_u_prev_0p1` | At this loose contraction factor, method choice matters much less than at `0.98-0.985`. |

Cross-method comparison points:

- The worst case in the whole sweep is not tied to one regularizer. It appears
  whenever `rho_lyap` is too strict.
- The nominal direct formulation does not need slack to work well here. The good
  runs at `0.985-0.99` achieve their behavior with zero Lyapunov slack
  activation.
- The three methods are not being ranked on different targets. The large
  differences in performance coincide with large differences in bounded-LS usage,
  target-reference mismatch, and contraction-violation count.

## What Should Be Stated

The report should state the following clearly:

1. The Lyapunov contraction factor is a major tuning variable for the direct
   controller and must be reported with every result.
2. In the current nominal single-setpoint study, `rho_lyap = 0.95` is too
   restrictive and materially harms all three methods.
3. The strong operating region is `rho_lyap = 0.985-0.99`.
4. The best method depends on `rho_lyap`:
   - input anchoring is strongest at `0.95-0.98`
   - `x_s` smoothing is strongest at `0.985`
   - all three methods are nearly tied at `0.99`
5. Therefore, conclusions about "which method is best" are not valid unless the
   contraction factor is fixed and reported.

## Summary

The most important new conclusion is not just that one method won one run. The
more important conclusion is that `rho_lyap` changes the direct-controller
regime.

- At `rho_lyap = 0.95`, the hard contraction is so aggressive that the direct
  controller spends almost the whole episode in bounded-LS target mode.
- At `rho_lyap = 0.98`, input anchoring is clearly the most reliable choice.
- At `rho_lyap = 0.985`, `x_s` smoothing becomes the best method.
- At `rho_lyap = 0.99`, the direct formulation is well-conditioned enough that
  all three methods are nearly equivalent on this nominal task.

That is why the contraction-factor sweep should replace the earlier single-run
report as the main statement for this experiment.

## Limitations And Next Step

These are still nominal single-setpoint results. The next experiment should keep
the same three methods and test:

1. `rho_lyap = 0.99` as the nominal default because it gives the cleanest
   overall baseline.
2. `rho_lyap = 0.985` as the discriminating stress point because it separates
   `x_s` smoothing from the input-anchor method most clearly.
3. disturbed and multi-setpoint runs before making broader claims about the best
   regularization strategy.
