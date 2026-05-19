# Direct Lyapunov Settled-Then-Weird Root-Cause Analysis 2026-05-19

## Objective

This note investigates the direct Lyapunov lexicographic no-RL runs in `results/direct_lyap_ch2_lex/` to answer two questions:

1. Is the post-settling oscillation mainly caused by very small `lyap_eps`?
2. Why can the Lyapunov controller appear to behave strangely after the outputs have already reached and briefly settled near the requested setpoint?

This update also incorporates the standalone truth-test notebook:

- `DirectLyapunov_TargetSelectorTruthTest.ipynb`
- saved notebook outputs analyzed here:
  - `results/target_selector_truth_test/20260518_222128/`

## Files And Bundles Inspected

- `results/direct_lyap_ch2_lex/20260518_224211/`
- `results/direct_lyap_ch2_lex/20260518_232630/`
- `results/direct_lyap_ch2_lex/20260518_233233/`
- `results/direct_lyap_ch2_lex/20260518_233701/`
- `results/target_selector_truth_test/20260518_222128/target_selector_truth_test.json`
- `results/target_selector_truth_test/20260518_222128/verdict.txt`
- `DirectLyapunov_TargetSelectorTruthTest.ipynb`
- [Lyapunov/direct_lyapunov_mpc.py](/c:/Users/hamediaa/OneDrive%20-%20McMaster%20University/PythonProjects/Lyapunov_polymer/Lyapunov/direct_lyapunov_mpc.py:642)
- [Lyapunov/lyapunov_core.py](/c:/Users/hamediaa/OneDrive%20-%20McMaster%20University/PythonProjects/Lyapunov_polymer/Lyapunov/lyapunov_core.py:315)

## Method Reminder

The direct controller enforces the first-step Lyapunov condition

$$
V_{k+1|k} \le \rho_{\mathrm{lyap}} V_k + \varepsilon_{\mathrm{lyap}},
$$

with

$$
V_k = (x_k - x_s(k))^\top P_x (x_k - x_s(k)).
$$

The important subtlety is that the Lyapunov center is not the raw setpoint. It is the selected steady target `x_s(k), y_s(k), u_s(k)` returned by the target selector at the current step.

So if the target selector changes from one step to the next, then a state that is well-settled relative to the old center can become far from the new center immediately. In that case the controller is not oscillating around a fixed equilibrium. It is contracting toward a moving equilibrium candidate.

## Run-Level Comparison

The four runs below use the same `lyap_mix_u0p1_x0p1_lex` case. The `rho` and `eps` values were inferred directly from the saved `V_bound = rho V_k + eps` arrays.

| Bundle | Inferred `rho` | Inferred `eps` | RMSE mean | Reward mean | Mean `|y_s-y_sp|_inf` | Exact steps in final segment |
|---|---:|---:|---:|---:|---:|---:|
| `20260518_224211` | 0.99 | `1e-6` | 0.397 | -4.228 | 0.645 | 28 / 1600 |
| `20260518_232630` | 0.99 | `1e-6` | 0.418 | -4.830 | 0.837 | 28 / 1600 |
| `20260518_233233` | 0.99 | `1e-3` | 0.178 | -0.934 | 0.131 | 1282 / 1600 |
| `20260518_233701` | 0.98 | `1e-3` | 0.178 | -0.934 | 0.131 | 1282 / 1600 |

Two immediate points follow.

- The latest two good runs on May 18, 2026 are effectively identical in closed-loop performance even though one uses `rho=0.99` and the other uses `rho=0.98`.
- The big separation is between the `eps approx 1e-6` runs and the `eps approx 1e-3` runs, but that does not yet prove that `eps` is the root cause. It only shows correlation.

## Final-Segment Evidence

The clearest behavior separation appears in the last setpoint segment, steps 4800 to 6399.

| Bundle | Tail-200 mean `|y-y_sp|_inf` | Tail-200 mean `|y_s-y_sp|_inf` | Tail-200 mean `target_rate_inf` | Steps above `|y-y_sp|_inf > 0.5` after first entering the 0.1 band |
|---|---:|---:|---:|---:|
| `20260518_224211` | 0.789 | 1.063 | 0.0193 | 603 |
| `20260518_232630` | 0.144 | 0.138 | 0.0154 | 601 |
| `20260518_233233` | 0.003 | 0.000 | 0.000023 | 0 |

The key separation is not only raw output error. It is target quality:

- In the bad fourth-latest run `20260518_224211`, the target itself stays far from the requested setpoint late in the run.
- In the good run `20260518_233233`, the selected target collapses onto the requested setpoint and stays there.

![Final segment root-cause comparison](figures/2026-05-19_direct_lyap_settled_weird_root_cause/final_segment_root_cause_comparison.png)

Interpretation:

- The bad run does not spend the final segment settled around one fixed target and then chatter inside a tiny Lyapunov tube.
- Instead, it spends almost the entire final segment in `bounded_ls`, with only six short exact-target episodes of lengths `[2, 9, 1, 6, 2, 8]`.
- The good run enters `exact_bounded` once and then stays there for 1282 consecutive steps.

This is the first main root-cause clue.

## Why It Looks Weird After Settling

The weird behavior comes from target recentering, not from a fixed-equilibrium oscillation.

### Bad run: `20260518_224211`

In the final segment, the run first reaches the band `|y-y_sp|_inf < 0.1` at local step 235, that is global step 5035. But that settling is not robust.

At global steps 5116 to 5150:

- the target selector briefly reports `frozen_output_disturbance_exact_bounded`
- `target_quality_mismatch_inf = 0.0`
- `V_k` is small, about `0.041` to `0.075`
- the contraction margin is essentially active, around floating-point zero

Then the mode flips back:

- at global step 5151, the target stage returns to `frozen_output_disturbance_bounded_ls`
- `target_quality_mismatch_inf` jumps immediately to `0.652`
- `V_k` jumps from `0.075` to `2.657`

Two steps later:

- at global step 5153, `target_quality_mismatch_inf = 2.119`
- `V_k = 25.847`

So the controller did not start by oscillating around a tiny steady-state neighborhood. It first lost the exact target, then the bounded least-squares target moved away, then the Lyapunov center moved, then `V_k` exploded relative to the new center.

### Good run: `20260518_233233`

At the same region:

- the target remains `frozen_output_disturbance_exact_bounded`
- `target_quality_mismatch_inf` stays at `0.0`
- `V_k` stays around `0.02`
- there is no exact-to-bounded fallback cascade

This is the second main root-cause clue.

## Why Small `lyap_eps` Is Not The Primary Root Cause

The evidence does support a secondary role for `lyap_eps`, but not a primary one.

### What `lyap_eps` can plausibly do

When `V_k` is tiny, the bound

$$
\rho V_k + \varepsilon_{\mathrm{lyap}}
$$

is much tighter for `eps=1e-6` than for `eps=1e-3`. That can make the first-step contraction almost an equality constraint near equilibrium and may increase sensitivity to:

- solver conditioning
- target-stage switching
- small target movements
- exact-to-bounded fallback events

So a larger `lyap_eps` can help the controller remain on the calm exact-target branch indirectly.

### Why `lyap_eps` is not enough to explain the bad behavior

At the onset of the bad excursion in `20260518_224211`:

- global step 5151 has `V_k = 2.657`
- global step 5153 has `V_k = 25.847`

At those values, changing `eps` from `1e-6` to `1e-3` changes `V_bound` by only `0.000999`. That is negligible relative to `\rho V_k`, which is already order `1` to `25`.

So by the time the visible weirdness starts, the additive epsilon is too small to be the dominant physical explanation. The bad event is already being driven by target mismatch and target-center movement.

### Best interpretation

The current evidence supports this layered view:

1. The primary root cause is target-center motion caused by target-selector mismatch and repeated exact-to-bounded fallback.
2. Small `lyap_eps` is a secondary amplifier because it makes near-equilibrium contraction more brittle, which may reduce the chance of staying on the exact-target branch once the target solve becomes delicate.
3. The visible oscillation is therefore not best described as "Lyapunov chatters near zero because `eps` is tiny." It is better described as "the controller keeps being asked to contract toward a moving or biased target center."

## Truth-Test Notebook Results

The truth-test notebook gives an important independent piece of evidence because it removes the closed-loop rollout and asks a simpler question:

> If the disturbance estimate is frozen at zero, is the linear target selector a faithful target object for the nominal nonlinear plant?

The saved verdict from `results/target_selector_truth_test/20260518_222128/verdict.txt` is:

> The linear target-selector model is the bottleneck for at least one nominal setpoint: the nonlinear plant can reach the setpoint within the configured tolerance, but the linear selector cannot.

In fact, the saved JSON shows both tested setpoints were classified as:

- `linear_says_unreachable_but_nonlinear_reachable`

### Per-setpoint truth-test summary

| Setpoint | Linear selector physical inf error | Best nonlinear physical inf error | Linear `u_s` | Best nonlinear `u` |
|---|---:|---:|---|---|
| SP 1: `[4.5, 324.0]` | 0.604 | 0.000013 | `[393.5, 670.0]` | `[621.0, 498.6]` |
| SP 2: `[3.4, 321.0]` | 2.397 | 0.000004 | `[870.0, 78.0]` | `[494.7, 262.0]` |

![Truth-test linear versus nonlinear](figures/2026-05-19_direct_lyap_settled_weird_root_cause/truth_test_linear_vs_nonlinear.png)

Interpretation:

- The linear selector often pushes to an input bound and still misses the requested output badly.
- The nonlinear steady search finds an interior feasible input that hits the same setpoint almost exactly.
- This means the selector mismatch is real even before any Lyapunov rollout, RL policy, or observer drift is considered.

That truth-test result is fully consistent with the final-segment behavior in `direct_lyap_ch2_lex`:

- if the target object is biased or structurally wrong,
- then the Lyapunov controller can settle relative to that wrong target,
- and when the target solve changes stage or re-centers, the controller can suddenly appear strange from the raw-setpoint point of view.

## Root-Cause Statement

The root cause of the settled-then-weird behavior is not primarily that `lyap_eps` is tiny.

The root cause is:

1. The target selector can produce biased steady targets for setpoints that the nonlinear plant can actually reach.
2. In the bad runs, the final segment repeatedly falls back from exact target solves to bounded least-squares target solves.
3. Those fallback events move the Lyapunov center `x_s, y_s`.
4. The controller then contracts toward a moving center rather than the raw requested setpoint.
5. A very small `lyap_eps` likely makes this regime more brittle near equilibrium, but it is not the main mechanism once the large excursion starts.

In short:

$$
\text{main issue} = \text{moving / biased target center},
$$

not

$$
\text{fixed-center Lyapunov chatter caused only by tiny } \varepsilon_{\mathrm{lyap}}.
$$

## Practical Implications

The calm latest runs should not be interpreted as proof that only `lyap_eps` needed fixing.

What the latest runs more likely show is:

- larger `lyap_eps` helped the closed loop remain on the exact-target branch,
- once on that branch, `y_s` stayed close to `y_sp`,
- then the Lyapunov controller looked normal because its center stopped moving.

That is a much stronger and more useful interpretation than "epsilon bigger solved oscillation."

## Recommended Next Experiments

### 1. Controlled `lyap_eps` sweep with identical target logic

Hold everything else fixed and run:

- `lyap_eps = 1e-6`
- `lyap_eps = 1e-5`
- `lyap_eps = 1e-4`
- `lyap_eps = 1e-3`

Track:

- fraction of final-segment exact-target steps
- number and length of exact-to-bounded fallback episodes
- post-settling steps with `|y-y_sp|_inf > 0.5`
- tail mean `|y_s-y_sp|_inf`

This isolates the secondary role of epsilon directly.

### 2. Add exact/bounded hysteresis

Once `exact_bounded` is active and `|y-y_sp|_inf < 0.1`, do not drop back to `bounded_ls` unless the exact target becomes clearly infeasible for several consecutive steps.

This directly targets the mode-flip mechanism seen in the bad runs.

### 3. Freeze or slow the target center near settling

When the controller is inside a small raw-setpoint band and the exact target is good, limit `target_rate_inf` or freeze `x_s, y_s` temporarily.

This tests whether the visible weirdness disappears when the Lyapunov center is prevented from moving.

### 4. Use the truth-test notebook as a selector regression test

The notebook already provides a high-signal unit-style check:

- if the nonlinear steady search can reach a setpoint but the linear selector cannot,
- the target object is not healthy enough for hard Lyapunov authority.

This should become a standard diagnostic before trusting improved closed-loop settling claims.

## Final Conclusion

The May 18, 2026 `direct_lyap_ch2_lex` evidence and the truth-test notebook tell the same story.

Small `lyap_eps` is not the main root cause of the settled-then-weird behavior. The deeper issue is that the Lyapunov controller is centered on a target generated by a selector that can be biased, can switch between exact and bounded least-squares modes, and can move significantly even after the raw outputs have already reached the requested setpoint. The strange late behavior is therefore best understood as contraction toward a moving or wrong target center, with small `lyap_eps` acting as a secondary brittleness factor near equilibrium rather than the primary driver of the excursions.
