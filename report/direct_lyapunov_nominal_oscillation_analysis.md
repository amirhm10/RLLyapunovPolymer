# Direct Lyapunov Nominal Oscillation Analysis

This report revisits the nominal oscillation issue after the latest
ten-scenario direct frozen-output-disturbance Lyapunov MPC export:

`Data/debug_exports/direct_lyapunov_mpc_ten_scenario/20260423_234338`

The earlier four-scenario export remains useful because it captured the
failure mechanism clearly:

`Data/debug_exports/direct_lyapunov_mpc_four_scenario/20260423_221822`

The short answer is now sharper:

The oscillation is not caused by an external plant disturbance or by plotting
scale. It is caused by internal reference management. The output-disturbance
observer moves `d_hat` during nominal nonlinear transients, and the bounded
least-squares target selector converts that moving disturbance estimate into a
moving artificial steady target. In the unregularized bounded cases, that
target jumps between input-bound active sets while the scheduled setpoint is
flat. The MPC objective still tracks `y_sp`, but the Lyapunov constraint is
centered at the moving `x_s`. That mismatch is the oscillation engine.

The ten-scenario run confirms the diagnosis because adding only a
previous-input target regularizer largely removes the oscillation indicators.
In the flat-setpoint windows where the old bounded cases toggled target
corners 8 to 14 times, the regularized bounded cases keep the selected target
interior in the exported step tables. Solver failures, active Lyapunov slack,
mean output RMSE, and target-to-previous-input jumps all improve.

## Main Finding

The root cause is a nonsmooth target projection feeding a Lyapunov constraint.
The sensitive chain is:

```math
\hat d_k
\longrightarrow
b_k = y_{\mathrm{sp},k}-\hat d_k
\longrightarrow
u_s(k)=\Pi_{\mathcal U}^{G}(b_k)
\longrightarrow
x_s(k)
\longrightarrow
V_k(x_s(k)).
```

Even when `y_sp` is constant, this chain is not constant unless `d_hat` is
constant and the projection active set is stable. In the nominal plant, the
linear predictor is still imperfect during large nonlinear transients, so the
output-disturbance observer can move `d_hat`. That moving `d_hat` then changes
the target request seen by the bounded target selector.

The ten-scenario run proves that the target selector is the right place to
look: changing the bounded target projection with a `u_s-u_{k-1}` term improves
closed-loop behavior without changing the nominal plant, setpoint schedule,
main MPC objective, or Lyapunov formulation.

## Evidence From The Old Oscillation Windows

The old bounded target projection solved:

```math
u_s(k)
=
\arg\min_{u_{\min}\le u\le u_{\max}}
\left\|G u - \left(y_{\mathrm{sp}}(k)-\hat d_k\right)\right\|_2^2,
\qquad
x_s(k)=(I-A)^{-1}B u_s(k),
```

where:

```math
G=C(I-A)^{-1}B.
```

The direct MPC objective was:

```math
\sum_i
\left\|y_{i|k}-y_{\mathrm{sp}}(k)\right\|_{Q_y}^2
+
\sum_i
\left\|\Delta u_{i|k}\right\|_{R_{\Delta u}}^2 .
```

The first-step Lyapunov constraint was centered at the selected target:

```math
(x_{1|k}-x_s(k))^T P_x (x_{1|k}-x_s(k))
\le
\rho
(\hat x_k-x_s(k))^T P_x(\hat x_k-x_s(k))
+\epsilon .
```

So the optimizer was balancing two references:

```math
\text{output objective center: } y_{\mathrm{sp}},
\qquad
\text{Lyapunov center: } x_s,u_s .
```

That is manageable if `x_s` moves slowly. It becomes dangerous when the
bounded projection jumps between active input faces.

The earlier window summary shows that the suspicious behavior did not come
from the setpoint schedule. The setpoint was flat inside the windows, but the
selected target moved sharply.

| Case | Window | eta range | T range | `y_s` eta range | `y_s` T range | `u_s` Qc range | `u_s` Qm range | `u_s` corner toggles | corners | solver fails | slack active |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `bounded_hard` | 220-320 | 1.600 | 9.125 | 1.358 | 16.506 | 798.400 | 592.000 | 10 | `II IU LU UI UL UU` | 1 | 0 |
| `bounded_hard` | 450-560 | 1.258 | 4.873 | 1.003 | 12.224 | 798.400 | 592.000 | 12 | `IL IU LI LL LU UI UL UU` | 7 | 0 |
| `bounded_hard` | 580-700 | 1.756 | 6.597 | 1.427 | 14.292 | 798.400 | 592.000 | 13 | `IU LU UI UL UU` | 6 | 0 |
| `bounded_soft` | 220-320 | 2.208 | 12.317 | 1.666 | 30.707 | 798.400 | 592.000 | 8 | `II IU LU UI UL UU` | 0 | 5 |
| `bounded_soft` | 450-560 | 1.520 | 8.494 | 1.283 | 15.831 | 798.400 | 592.000 | 14 | `IL IU LI LL LU UI UL UU` | 0 | 0 |
| `bounded_soft` | 580-700 | 1.188 | 4.312 | 0.987 | 5.678 | 135.983 | 592.000 | 4 | `IU UI UL UU` | 0 | 0 |

Here `L` means that the target input is on its lower bound, `U` means upper
bound, and `I` means interior. The important column is `u_s corner toggles`:
the target was not sitting at one admissible steady point after the output
looked settled.

The most visible event was near step 500. In hard mode, the controller held
input through an infeasible segment and then the target flipped from `LU` to
`UL`, effectively asking the Lyapunov center to move toward the opposite side
of the admissible steady-input box.

![bounded_hard near step 500.](figures/direct_lyapunov_nominal_oscillation_analysis/bounded_hard_near_t500.svg)

![bounded_hard disturbance estimate near step 500.](figures/direct_lyapunov_nominal_oscillation_analysis/bounded_hard_near_t500_dhat.svg)

In soft mode, the solver avoided the hard infeasibility, but the target center
still moved sharply. Around steps 501-505, the selected target temperature rose
well above the scheduled setpoint while the scheduled setpoint was still flat.

![bounded_soft near step 500.](figures/direct_lyapunov_nominal_oscillation_analysis/bounded_soft_near_t500.svg)

![bounded_soft disturbance estimate near step 500.](figures/direct_lyapunov_nominal_oscillation_analysis/bounded_soft_near_t500_dhat.svg)

## Why Nominal Mode Still Moves `d_hat`

Nominal mode means the plant parameters were not intentionally disturbed. It
does not mean the linear prediction model is exact over every nonlinear
transient.

The direct run uses a predictor-form output-disturbance observer:

```math
\hat z_{k+1}
=
A_{\mathrm{aug}}\hat z_k
+B_{\mathrm{aug}}u_k
+L\left(y_k^{\mathrm{scaled}}-C_{\mathrm{aug}}\hat z_k\right).
```

For an output-disturbance augmentation:

```math
\hat z_k =
\begin{bmatrix}
\hat x_k\\
\hat d_k
\end{bmatrix},
\qquad
\hat d_{k+1}
=
\hat d_k
+L_d
\left(y_k^{\mathrm{scaled}}-C_{\mathrm{aug}}\hat z_k\right).
```

So `d_hat` is an output-model-mismatch state. It is not proof that the plant
received an external disturbance. If the nonlinear plant output and the linear
prediction disagree during a transient, the observer can assign that error to
`d_hat`. The target selector then sees `y_sp - d_hat`, so a constant scheduled
setpoint can become a moving internal target request.

## Evidence From The Ten-Scenario Run

The ten-scenario export added bounded target selectors with:

```math
\min_{u_{\min}\le u_s\le u_{\max}}
\left\|G u_s - \left(y_{\mathrm{sp},k}-\hat d_k\right)\right\|_2^2
+
\lambda_{\mathrm{prev}}
\left\|u_s-u_{k-1}\right\|_2^2 ,
```

with:

```math
\lambda_{\mathrm{prev}}\in\{0.1,\;1.0,\;10.0\}.
```

This one change greatly reduces target movement relative to the previously
applied input.

| Case | `lambda_prev` | Solver success | Hard contraction | Slack active | Mean `||u_s-u_prev||_inf` | Mean state-target error | Mean output RMSE | Reward mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bounded_hard` | 0 | 96.75% | 96.75% | 0 | 11.67 | 96.28 | 1.146 | -26.44 |
| `bounded_soft` | 0 | 97.38% | 96.38% | 16 | 12.06 | 85.94 | 1.358 | -33.56 |
| `bounded_hard_u_prev` | 0.1 | 99.50% | 99.50% | 0 | 0.524 | 44.40 | 0.644 | -11.64 |
| `bounded_soft_u_prev` | 0.1 | 100.00% | 99.69% | 5 | 0.990 | 38.22 | 0.821 | -15.99 |
| `bounded_hard_u_prev_1p0` | 1.0 | 99.81% | 99.81% | 0 | 0.430 | 34.94 | 0.567 | -7.694 |
| `bounded_soft_u_prev_1p0` | 1.0 | 100.00% | 100.00% | 0 | 0.518 | 21.30 | 0.371 | -3.598 |
| `bounded_hard_u_prev_10p0` | 10.0 | 99.31% | 99.31% | 0 | 0.432 | 35.46 | 0.487 | -6.568 |
| `bounded_soft_u_prev_10p0` | 10.0 | 100.00% | 100.00% | 0 | 0.320 | 22.39 | 0.405 | -4.183 |

The soft `lambda_prev = 1.0` case is the best overall nominal result. Relative
to unregularized `bounded_soft`, it:

- reduces mean `||u_s-u_prev||_inf` from `12.06` to `0.518`;
- reduces mean state-target error from `85.94` to `21.30`;
- reduces mean physical output RMSE from `1.358` to `0.371`;
- improves reward mean from `-33.56` to `-3.598`;
- removes active Lyapunov slack;
- reaches 100% solver success and 100% hard contraction diagnostics.

The `lambda_prev = 10.0` soft case is also clean, but it is slightly worse than
`lambda_prev = 1.0` in reward and RMSE. That is an important tuning lesson:
stronger target anchoring is not monotonically better. At some point, the
selector resists useful target adaptation.

![Ten-scenario output RMSE comparison](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_output_rmse.png)

![Ten-scenario target residual and bounded activity](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_target_residual_bounded_activity.png)

## Window Retest After Regularization

Parsing the ten-scenario step tables in the same flat-setpoint windows gives
the decisive pattern:

| Window | Unregularized hard | Unregularized soft | Soft `lambda_prev = 1.0` | Soft `lambda_prev = 10.0` |
| --- | --- | --- | --- | --- |
| 220-320 | 10 target-corner toggles, 1 solver fail | 8 target-corner toggles, 5 slack-active steps | 0 target-corner toggles, 0 solver fails, 0 slack-active steps | 0 target-corner toggles, 0 solver fails, 0 slack-active steps |
| 450-560 | 12 target-corner toggles, 7 solver fails | 14 target-corner toggles, 0 solver fails | 0 target-corner toggles, 0 solver fails, 0 slack-active steps | 0 target-corner toggles, 0 solver fails, 0 slack-active steps |
| 580-700 | 13 target-corner toggles, 6 solver fails | 4 target-corner toggles, 0 solver fails | 0 target-corner toggles, 0 solver fails, 0 slack-active steps | 0 target-corner toggles, 0 solver fails, 0 slack-active steps |

This does not mean the plant becomes perfectly still. It means the pathological
mechanism is removed: the Lyapunov center is no longer jumping between active
input-bound corners inside flat-setpoint windows.

The full ten-scenario overlays now show that the regularized cases are the
only credible supervisor candidates.

![Ten-scenario output overlay](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_outputs_overlay.png)

![Ten-scenario input overlay](figures/direct_lyapunov_mpc_frozen_output_disturbance/comparison_inputs_overlay.png)

## Why Unbounded And Unregularized Cases Are Not Acceptable

The unbounded cases are useful diagnostics but bad controllers:

- `unbounded_hard` has exact target residual but 0% solver success because the
  exact steady input is inadmissible at every step.
- `unbounded_soft` has lower output RMSE than the unregularized bounded cases,
  but it needs active slack on 1131 of 1600 steps and has the worst reward
  because it is softening a fundamentally infeasible Lyapunov center.

The unregularized bounded cases are admissible but still problematic:

- the bounded projection frequently selects input-bound corners;
- corner flips move the Lyapunov center sharply;
- hard mode can become infeasible and hold the previous input;
- soft mode solves more often, but can still chase a moving artificial target.

The regularized bounded cases are different because the target problem gains
memory. The stationarity condition for inactive bounds becomes:

```math
\left(G^\top G+\lambda_{\mathrm{prev}}I\right)u_s
=
G^\top\left(y_{\mathrm{sp},k}-\hat d_k\right)
+
\lambda_{\mathrm{prev}}u_{k-1}.
```

That extra positive curvature makes weakly determined directions less sensitive
to small changes in `d_hat` and discourages active-set jumps unless the output
residual improvement is worth the movement.

## Literature-Grounded Interpretation

The literature points to the same diagnosis.

Offset-free MPC commonly uses estimated disturbances and a steady-state target
calculation to remove steady offset. Muske and Badgwell emphasize that the
disturbance model, estimator, target calculation, and dynamic controller must
work together; the disturbance estimate is part of the controller design, not a
physical disturbance measurement by itself. Pannocchia and Rawlings give
general zero-offset conditions for disturbance-augmented MPC, especially when
the number of measured and controlled variables does not make offset-free
tracking automatic.

The tracking MPC literature is even more directly relevant. Limon, Alvarado,
Alamo, and Camacho propose adding an artificial steady state and input as
decision variables, penalizing the distance from that artificial target to the
desired target, and using a terminal invariant set for tracking. Their key
point maps directly to this project: if the requested steady target is not
admissible, the controller should steer to the closest admissible steady state,
not chase an infeasible reference or repeatedly recenter itself outside the
optimization.

The constrained MPC stability literature, represented by Mayne, Rawlings, Rao,
and Scokaert, explains why terminal ingredients must be consistent with the
reference being tracked. The Lyapunov-based predictive control literature,
including Mhaskar, El-Farra, and Christofides, likewise emphasizes feasibility
regions, auxiliary bounded control, and careful treatment of hard versus soft
constraints.

Reference-governor literature gives the same practical message in a different
language: when constraints make a requested command unsafe or infeasible, modify
the reference evolution so constraints are respected. The `u_s-u_prev`
regularizer is acting like a simple implicit reference governor for the
artificial steady input. It smooths target evolution before the Lyapunov
constraint sees it.

## Best Next Step

The best next step is not another isolated plot fix and not only a wider weight
sweep. The best literature-aligned design is to move from:

```text
external target selector -> Lyapunov MPC around selected target
```

to:

```text
one tracking MPC with artificial steady state/input as decision variables
```

The next controller should solve for the artificial target and the control
trajectory together. A practical formulation is:

```math
\min_{\{u_i,x_i\},x_a,u_a}
\sum_{i=1}^{N_p}
\left\|y_{i|k}-y_a\right\|_Q^2
+
\sum_{i=0}^{N_c-1}
\left\|\Delta u_{i|k}\right\|_R^2
+
\left\|y_a-y_{\mathrm{sp},k}\right\|_{Q_o}^2
+
\lambda_a\left\|u_a-u_{k-1}\right\|_2^2
+
\lambda_{\Delta a}\left\|u_a-u_{a,k-1}\right\|_2^2 .
```

subject to:

```math
x_{i+1|k}=Ax_{i|k}+Bu_{i|k},
\qquad
y_{i|k}=Cx_{i|k}+\hat d_k,
```

```math
(I-A)x_a-Bu_a=0,
\qquad
y_a=Cx_a+\hat d_k,
\qquad
u_{\min}\le u_a\le u_{\max},
```

plus input constraints and either:

```math
(x_{1|k}-x_a)^T P (x_{1|k}-x_a)
\le
\rho(\hat x_k-x_a)^T P(\hat x_k-x_a)+\epsilon+s_k,
```

or a terminal tracking set centered at `(x_a,u_a)`.

This is the Limon-style artificial-reference idea adapted to the current
frozen-output-disturbance Lyapunov path. It has three advantages over the
current two-stage target selector:

1. The MPC objective, Lyapunov center, and admissible artificial target are
   optimized in one problem.
2. If `y_sp` is not admissible, the optimizer chooses a nearby admissible
   target instead of letting an external projection jump between corners.
3. The target regularization becomes an explicit design term, not a patch
   hiding in a pre-solve selector.

## Why One-Stage Is Better For The RL Safety Layer

The final project goal is not simply to make a standalone MPC controller. The
goal is an RL controller with a safety layer that can guarantee stability or
certify when an RL action is safe. That changes the interpretation of the
one-stage proposal.

The one-stage artificial-target MPC is not meant to replace RL. It is a better
form for the safety layer. The RL actor should still propose the preferred
input `u_{\mathrm{RL}}`; the safety layer should then solve the smallest
necessary correction while choosing the admissible target and Lyapunov center
consistently.

The current two-stage safety architecture is:

```text
target selector computes x_s,u_s
        ↓
RL or MPC proposes u_cand
        ↓
safety filter checks/corrects u_cand around the preselected x_s,u_s
```

That structure is simple and useful for diagnosis, but the oscillation study
shows its main weakness. The target selector can choose a target that is
mathematically admissible as a steady state but poor for the current state,
current input, RL proposal, terminal set, or first-step Lyapunov contraction.
The safety filter then inherits that target. If the target moves sharply, the
filter is forced to either:

- reject or heavily modify the RL action;
- accept slack;
- hold the previous input;
- fall back to another controller;
- certify stability around a center that may move again at the next step.

That is not the cleanest guarantee. It is a conditional guarantee around an
externally selected target.

For the RL safety layer, the one-stage structure should instead be:

```text
RL proposes u_RL
        ↓
one optimization chooses u_safe and the artificial target x_a,u_a together
        ↓
return u_safe closest to u_RL subject to admissibility and Lyapunov conditions
```

A safety-layer version can be written as:

```math
\min_{u_0,\ldots,u_{N_c-1},x_a,u_a}
\left\|u_0-u_{\mathrm{RL}}\right\|_{R_{\mathrm{RL}}}^2
+
\sum_{i=0}^{N_c-1}\left\|\Delta u_i\right\|_R^2
+
\sum_{i=1}^{N_p}\left\|y_{i|k}-y_a\right\|_Q^2
+
\left\|y_a-y_{\mathrm{sp}}\right\|_{Q_o}^2
+
\lambda_a\left\|u_a-u_{k-1}\right\|^2 .
```

subject to the plant prediction model, input limits, artificial steady-state
equations, and the Lyapunov safety condition:

```math
(x_{1|k}-x_a)^TP(x_{1|k}-x_a)
\le
\rho(\hat x_k-x_a)^TP(\hat x_k-x_a)+\epsilon+s_k .
```

The key difference is that `u_RL` is in the objective, while admissibility,
input limits, and Lyapunov contraction are in the constraints. That is exactly
what a safety layer should do: preserve the learned policy whenever possible,
but override it when the stability certificate requires a correction.

This is better than the two-stage method for six reasons.

1. The target and action corrections are no longer competing projections.
   In the two-stage method, the target selector first projects the setpoint to
   an admissible steady target, and the safety filter then projects the RL
   action onto the safe set around that target. These two projections do not
   commute. A target that minimizes output residual can require a large action
   correction, while a nearby target with slightly worse output residual could
   allow a much smaller, stable correction. One-stage optimization sees that
   tradeoff directly.

2. The Lyapunov certificate uses the same target that the optimizer selected.
   Stability claims are only meaningful relative to the Lyapunov center. If
   `x_s` is chosen outside the correction problem, the safety layer can be
   forced to certify an RL action around a center that was not chosen with
   current feasibility in mind. In the one-stage design, the certificate,
   terminal ingredients, artificial target, and corrected action are one
   consistent object.

3. The safety layer becomes minimally invasive to RL.
   The current filter can accept `u_cand`, solve a correction, or fall back,
   but it cannot ask whether a slightly different artificial target would allow
   a smaller change to the RL action. A one-stage safety layer can directly
   minimize `||u_safe-u_RL||` while still enforcing contraction. That is the
   right behavior for an RL supervisor: intervene only as much as needed.

4. It reduces target-induced chattering.
   The oscillation study showed that the two-stage bounded target can jump
   between active-set faces. Adding `u_s-u_prev` regularization helped because
   it gave the target selector memory. In a one-stage safety layer, that memory
   can be formalized with penalties on `u_a-u_{k-1}` and
   `u_a-u_{a,k-1}`, and with optional target-rate constraints. These terms
   become part of the certificate-producing optimization rather than a
   pre-filter patch.

5. It handles infeasible setpoints more honestly.
   If `y_sp` is not exactly reachable under input constraints and the current
   disturbance estimate, the two-stage method must decide on a target before
   seeing the safety correction problem. The one-stage method can say:
   "this is the closest admissible target that also lets me keep the RL action
   as much as possible and satisfy Lyapunov contraction." That is closer to the
   constrained tracking MPC literature and closer to what a supervisor should
   do in deployment.

6. It gives cleaner diagnostics.
   If the one-stage problem fails, the failure has a precise meaning: no
   combination of admissible artificial target and near-RL action satisfied the
   chosen constraints and slack policy. In the two-stage method, failure can
   come from the target selector, target/action mismatch, terminal-set mismatch,
   Lyapunov infeasibility, or fallback policy. The current logs are useful, but
   the guarantee is harder to explain.

The practical implication is:

- keep the current two-stage safety filter as a diagnostic baseline;
- keep `bounded_soft_u_prev_1p0` as the best current nominal reference case;
- build the next safety layer as a one-stage artificial-target correction MPC
  that takes `u_RL` as an input and returns `u_safe`;
- report both the intervention size `||u_safe-u_RL||` and the Lyapunov margin.

This preserves the project direction. The RL policy remains the performance
controller. The MPC layer becomes the certificate-producing shield.

There is an important wording distinction here. In the one-stage design, the
safety layer is enabled every step in the sense that it checks or certifies the
RL action every step. It does not have to be active every step in the sense of
changing the RL action. If `u_RL` is already feasible and satisfies the
Lyapunov condition for an admissible artificial target, the one-stage problem
should return:

```math
u_{\mathrm{safe}} = u_{\mathrm{RL}}
```

up to numerical tolerance. That step should be logged as:

```text
safety checked, RL action accepted, no intervention
```

not as a safety intervention.

This is conceptually the same pass-through behavior as the current two-stage
filter when a candidate action is acceptable. The difference is how the
certificate is produced. In the current two-stage filter, the pass/fail check
is made around a target selected before seeing the correction problem. In the
one-stage safety layer, the pass-through decision is made while the admissible
artificial target, Lyapunov center, and candidate action are considered
together. Therefore the one-stage design can still be minimally invasive: it
should intervene only when preserving `u_RL` would violate constraints,
Lyapunov contraction, or the chosen slack policy.

A practical implementation can use a two-level computational workflow:

1. Fast certification: test whether `u_RL` is safe with the current/best
   artificial target and Lyapunov ingredients.
2. One-stage correction: only if the fast test fails, solve the full
   artificial-target safety projection for `u_safe`.

The guarantee still requires a safety certificate every step. The intervention
rate should be measured separately from the certification rate.

## Engineering Plan

Use the current best case as the baseline while building the structural fix.

### Step 1: Keep The Current Best Supervisor Candidate

Use `bounded_soft_u_prev_1p0` as the nominal baseline:

- best reward mean: `-3.598`;
- best mean physical output RMSE: `0.371`;
- 100% solver success;
- 100% hard and relaxed contraction diagnostics;
- zero active Lyapunov slack.

Keep `bounded_soft_u_prev_10p0` as the over-anchoring comparison and keep both
hard `1.0` and hard `10.0` cases as no-slack diagnostics.

### Step 2: Inspect The Remaining Hard-Mode Failure Cases

`bounded_hard_u_prev_10p0` has the best hard-mode reward and RMSE, but it has
11 held steps and a lower success rate than hard `lambda_prev = 1.0`. Before
using it as a strict backup, inspect those held steps:

- local target residual;
- active input bounds;
- first-step contraction margin;
- terminal alpha and whether the terminal set was skipped;
- previous input versus selected `u_s`;
- whether failures cluster near setpoint transitions or flat-setpoint windows.

### Step 3: Run A Focused Weight Sweep

Do not sweep blindly. The ten-scenario result already says the useful range is
around `lambda_prev = 1.0`. Run:

```python
lambda_prev_grid = [0.25, 0.5, 1.0, 2.0, 5.0]
```

Track:

- mean and max `||u_s-u_prev||_inf`;
- target corner toggles inside flat-setpoint windows;
- target residual max and mean;
- output RMSE in physical units;
- slack active steps;
- solver failures and contraction failures.

### Step 4: Implement Artificial-Target Tracking MPC

Add a new direct-controller variant rather than replacing the current one
immediately. The purpose is to test the literature-recommended structure while
keeping the existing two-stage controller as a comparison. For the final RL
project, this variant should be implemented as a safety-layer projection that
takes an RL action as the preferred input.

Suggested name:

`direct_artificial_target_lyapunov_mpc`

Core design:

- decision variables include `x_a,u_a` or equivalent reduced steady-input
  variables;
- objective penalizes `u_0-u_RL` when an RL candidate is supplied;
- objective tracks `y_a` over the horizon and penalizes `y_a-y_sp`;
- include `u_a-u_prev` and optionally `u_a-u_{a,prev}` penalties;
- keep the frozen `d_hat` target equation for offset-free behavior;
- center the Lyapunov or terminal constraint on the same artificial target
  chosen inside the MPC solve.

The report metrics for the RL safety-layer version should include:

- accepted RL action rate;
- mean and max intervention `||u_safe-u_RL||`;
- Lyapunov margin after the returned action;
- active slack count and slack magnitude;
- artificial target motion `||u_a-u_{a,k-1}||`;
- output RMSE and reward, so safety and performance are not conflated.

### Step 5: Revisit The Disturbance Model After The Target Fix

Do this after the artificial-target experiment, not before. The target-jump
mechanism is already confirmed, so observer tuning alone is unlikely to be the
best first fix. Still, the literature says disturbance-model design matters.
After target management is stable, test:

- lower observer gain or filtered `d_hat` for target calculation only;
- state/input disturbance alternatives to pure output disturbance;
- measurement-corrected target timing, where `\hat z_{k|k}` is used for target
  selection instead of the one-step predictor state;
- disturbed-plant runs to make sure the nominal fix does not destroy
  offset-free behavior.

### Step 6: Add A Reference Governor If Setpoint Infeasibility Persists

If artificial-target MPC still shows aggressive behavior near infeasible
setpoints, add a lightweight reference governor on `y_sp` or on the artificial
target offset. This is the reference-governor version of what the target
regularizer is already doing: slow or reshape the command so the constrained
closed loop sees an admissible reference evolution.

## Final Diagnosis

The oscillation is an internal target-motion problem. The nominal plant is not
being disturbed, but the output-disturbance observer moves `d_hat` during
nominal nonlinear transients. The unregularized bounded target selector then
maps `y_sp-d_hat` through a nonsmooth constrained projection. When the active
set changes, the selected `x_s,u_s` can jump between input-bound corners. The
Lyapunov constraint follows that moving center, while the MPC objective still
tracks the scheduled setpoint.

The ten-scenario result is the strongest evidence because adding
previous-input target regularization removes the corner-flip signature and
dramatically improves the closed-loop metrics. The current best practical
choice is `bounded_soft_u_prev_1p0`. The best research/engineering next step is
to implement a Limon-style artificial steady-state tracking MPC variant so the
admissible target, offset penalty, tracking objective, and Lyapunov center are
chosen consistently in one optimization problem.

## References

- D. Limon, I. Alvarado, T. Alamo, and E. F. Camacho, "MPC for tracking
  piecewise constant references for constrained linear systems," *Automatica*,
  2008. https://doi.org/10.1016/j.automatica.2008.01.023
- D. Q. Mayne, J. B. Rawlings, C. V. Rao, and P. O. M. Scokaert,
  "Constrained model predictive control: Stability and optimality,"
  *Automatica*, 2000. https://doi.org/10.1016/S0005-1098(99)00214-9
- K. R. Muske and T. A. Badgwell, "Disturbance modeling for offset-free linear
  model predictive control," *Journal of Process Control*, 2002.
  https://doi.org/10.1016/S0959-1524(01)00051-8
- G. Pannocchia and J. B. Rawlings, "Disturbance models for offset-free
  model-predictive control," *AIChE Journal*, 2003.
  https://doi.org/10.1002/aic.690490213
- P. Mhaskar, N. H. El-Farra, and P. D. Christofides, "Stabilization of
  nonlinear systems with state and control constraints using Lyapunov-based
  predictive control," *Systems & Control Letters*, 2006.
  https://doi.org/10.1016/j.sysconle.2005.09.014
- E. Garone, S. Di Cairano, and I. V. Kolmanovsky, "Reference and command
  governors for systems with constraints: A survey on theory and applications,"
  *Automatica*, 2017. https://doi.org/10.1016/j.automatica.2016.08.013
