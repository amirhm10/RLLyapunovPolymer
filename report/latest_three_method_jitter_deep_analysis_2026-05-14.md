# Latest Three-Method Jitter Deep Analysis

## Scope

This report analyzes the latest completed runs for the three active direct-Lyapunov experiment families:

- pretrained RL gate: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260513_232640`
- cold-start RL gate: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260513_232632`
- direct no-RL baseline: `results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260513_191433`

The active comparison cases are:

- no selector anchoring: `bounded_hard`
- mixed anchoring: `u_prev + x_s_prev`

Important note:

- the direct no-RL export still uses the stale case label `bounded_hard_u_prev_0p1_xs_prev_0p1`
- the saved arrays confirm the actual active weights are `u_ref_weight = 0.25` and `x_ref_weight = 0.25`

The focus here is **jitter only**: input chatter, target motion, fallback switching, and algorithm inconsistencies that create oscillatory behavior.

## Method

I analyzed the saved `arrays.npz`, `step_table.csv`, and summary bundles for all six latest case bundles.

The main window of interest is the **second 800-step constant-setpoint plateau** of the final 1600-step block:

- local plateau window: steps `319200:320000`
- deepest steady-state tail window: steps `319800:320000`

This isolates jitter after the setpoint has already been constant for a long time.

For each case I computed:

- mean applied-input motion: `$\\|\\Delta u_{apply}\\|_\\infty$`
- mean target-input motion: `$\\|\\Delta u_s\\|_\\infty$`
- output error to the certified target: `$\\|y-y_s\\|_\\infty$`
- target mismatch: `$\\|y_s-y_{sp}\\|_\\infty$`
- mode-switch counts
- fallback and solver-status counts

For the RL bundles, the output deviation was reconstructed as

$$
y_k = e_k + y_{sp,k},
$$

from the saved `e_store` and `y_sp`.

For the direct no-RL bundles, the output deviation was reconstructed as

$$
y_k = (y_k - y_{sp,k}) + y_{sp,k},
$$

from the saved `y_minus_y_sp_store` and `y_sp`.

## The Main Result

The latest jitter is not one phenomenon. It has two layers:

1. a **core direct-method jitter** that already exists without RL
2. an **extra RL-wrapper jitter** added on top of that baseline

The clearest proof is this:

- the latest direct no-RL no-anchor case jitters even with `200/200` tail steps solved as `optimal`
- therefore RL fallback switching and `optimal_inaccurate` cannot be the root cause by themselves

The root cause is a combination of:

- moving admissible targets in the bounded least-squares selector
- tracking raw `y_sp` while certifying contraction around `y_s`
- RL accepted actions that satisfy only one-step contraction and hug the Lyapunov boundary
- no hysteresis around RL/fallback switching

## Tail Jitter Summary

### Motion Metrics

| Case | Tail mean `$\\|\\Delta u_{apply}\\|_\\infty$` | Tail mean `$\\|\\Delta u_s\\|_\\infty$` | Tail mean `$\\|\\Delta y\\|_\\infty$` | Tail mean `$\\|\\Delta y_s\\|_\\infty$` |
| --- | ---: | ---: | ---: | ---: |
| Pretrained RL, no anchor | 0.650 | 3.870 | 0.0485 | 0.221 |
| Pretrained RL, mixed 0.25 | 0.197 | 1.631 | 0.0238 | 0.150 |
| Cold-start RL, no anchor | 0.507 | 2.559 | 0.0523 | 0.270 |
| Cold-start RL, mixed 0.25 | 0.231 | 1.366 | 0.0157 | 0.0866 |
| Direct MPC, no anchor | 0.233 | 1.666 | 0.138 | 0.311 |
| Direct MPC, mixed 0.25 | 0.0254 | 0.0250 | 0.0149 | 0.0209 |

### Error And Switching Metrics

| Case | Tail mean `$\\|y-y_s\\|_\\infty$` | Tail mean `$\\|y-y_{sp}\\|_\\infty$` | Tail mean `$\\|y_s-y_{sp}\\|_\\infty$` | Tail switches |
| --- | ---: | ---: | ---: | ---: |
| Pretrained RL, no anchor | 0.328 | 0.175 | 0.402 | 29 |
| Pretrained RL, mixed 0.25 | 0.242 | 0.0612 | 0.258 | 20 |
| Cold-start RL, no anchor | 0.441 | 0.174 | 0.490 | 18 |
| Cold-start RL, mixed 0.25 | 0.193 | 0.0806 | 0.216 | 15 |
| Direct MPC, no anchor | 1.467 | 1.807 | 2.307 | 0 |
| Direct MPC, mixed 0.25 | 0.196 | 0.709 | 0.621 | 0 |

## First High-Level Conclusion

Mixed anchoring is the strongest anti-jitter change in all three experiment families.

The evidence is strongest in the direct no-RL baseline:

- direct no-anchor tail input jitter: `0.233`
- direct mixed tail input jitter: `0.025`

So mixed anchoring cuts direct-method tail input motion by about `9.2x`.

That matters because it shows the selector itself is a major source of jitter, independent of RL.

## Second High-Level Conclusion

RL adds extra jitter on top of the direct-method baseline, even in the mixed-anchor case.

Relative to the direct mixed baseline:

- pretrained mixed RL tail jitter is about `7.8x` larger
- cold-start mixed RL tail jitter is about `9.1x` larger

So mixed anchoring alone is not enough. The RL wrapper is re-introducing motion that the direct baseline no longer has.

## Figure 1: Input Motion And Target Motion

Figure 1 shows the second constant-setpoint plateau for all six latest cases. It compares applied-input motion against target-input motion. For the RL cases, green shading marks fallback direct-MPC intervention and red lines mark `optimal_inaccurate`.

![Latest three-method input motion](figures/2026-05-14_three_method_jitter_analysis/latest_three_method_seg2_input_motion.png)

Two visual facts stand out:

- direct no-anchor already has persistent motion even without RL switching
- direct mixed is nearly flat, while RL mixed is not

That isolates the extra RL jitter very clearly.

## Figure 2: Output Error Decomposition

Figure 2 shows the same plateau but decomposes the output error into:

- raw-setpoint error, `$\\|y-y_{sp}\\|_\\infty$`
- certified-target error, `$\\|y-y_s\\|_\\infty$`
- target mismatch, `$\\|y_s-y_{sp}\\|_\\infty$`

![Latest three-method error decomposition](figures/2026-05-14_three_method_jitter_analysis/latest_three_method_seg2_error_decomposition.png)

This figure shows the core inconsistency:

- the controller is often closer to raw `y_sp` than to the admissible target `y_s`
- yet the Lyapunov certificate is built around `y_s`

## Figure 3: Tail Summary

Figure 3 summarizes the final 200-step tail numerically.

![Latest three-method tail summary](figures/2026-05-14_three_method_jitter_analysis/latest_three_method_tail_summary.png)

## What The Direct No-RL Baseline Proves

The direct no-RL baseline is the most important control experiment in the latest data.

In the final 200-step tail:

- all `200/200` no-anchor steps use `direct_lyapunov_mpc`
- all `200/200` steps have solver status `optimal`
- all `200/200` steps use `frozen_output_disturbance_bounded_ls`
- there are no RL accepted/fallback switches because RL is absent

Yet the case still has:

- tail mean `$\\|\\Delta u_{apply}\\|_\\infty = 0.233`
- tail mean `$\\|\\Delta u_s\\|_\\infty = 1.666`
- tail mean `$\\|y_s-y_{sp}\\|_\\infty = 2.307`

Therefore:

- `optimal_inaccurate` is not required for jitter
- RL switching is not required for jitter
- the direct method itself contains a structural jitter mechanism

## Core Direct-Method Inconsistency

The structural inconsistency is in the objective split:

- target selection produces an admissible steady target `(x_s,u_s,y_s)`
- first-step Lyapunov contraction is checked around `(x_s,u_s)`
- but the direct tracking problem still tracks raw `y_sp` when `use_target_output_for_tracking=False`

This is explicit in `Lyapunov/direct_lyapunov_mpc.py`:

```python
y_target = y_s.copy() if use_target_output_for_tracking else y_sp_k.copy()
```

The latest direct and RL notebooks are still running with:

- `direct_tracking_use_target_output=False`

So the direct method is solving:

$$
\\text{contract around } x_s
\\quad \\text{while simultaneously tracking } y_{sp}.
$$

That is a genuine formulation inconsistency, not just a tuning issue.

It explains why direct no-anchor can jitter even with perfect solver success.

## Why The No-Anchor Selector Jitters

The no-anchor selector solves a bounded least-squares target problem with no regularization toward previous `u_s` or previous `x_s`.

The current direct target preparation in `Lyapunov/direct_lyapunov_mpc.py:651-664` calls:

- `solve_output_disturbance_target(...)`
- with `u_ref = u_prev_dev`
- with `x_ref = x_target_prev_success`

But when the configured weights are zero, those references are inactive and the bounded least-squares solution can slide along the admissible manifold as the observer state changes.

This is exactly what the latest numbers show:

- direct no-anchor tail mean `$\\|\\Delta u_s\\|_\\infty = 1.666`
- direct mixed tail mean `$\\|\\Delta u_s\\|_\\infty = 0.025`

So the target motion is a first-order jitter source.

## Why Bounded-LS By Itself Is Not The Problem

All six latest cases spend most or all of the jitter window in:

- `frozen_output_disturbance_bounded_ls`

Yet they are not equally jittery.

For example:

- direct mixed: all `200/200` tail steps are bounded-LS, but the loop is nearly flat
- pretrained no-anchor: `191/200` tail steps are bounded-LS and the loop is visibly jittery

So the key distinction is not:

- exact bounded target versus bounded-LS target

It is:

- moving bounded-LS target versus stable bounded-LS target

## What RL Adds On Top Of The Direct Baseline

The RL wrapper adds three more inconsistencies.

### 1. The policy is trained and evaluated on `y_sp`, not on `y_s`

The RL state is built in `utils/helpers.py` as:

$$
s_k = [xhatdhat_k,\; y_{sp,k},\; u_{k-1}]
$$

There is no `y_s` or `y_s-y_{sp}` in the observation.

So the actor is asked to act for raw `y_sp`, but the gate judges it relative to `(x_s,u_s,y_s)`.

This is a structural mismatch between the policy objective and the safety objective.

### 2. Behavioral cloning teaches the same mismatch

The BC teacher in `Simulation/run_rl_lyapunov.py:1011-1050` is generated by the same direct Lyapunov solver with:

- `use_target_output_for_tracking=direct_tracking_use_target_output`

and the latest RL notebooks still pass:

- `direct_tracking_use_target_output=False`

So the teacher itself is not a pure `y_s`-tracking controller. The mismatch is learned directly during BC.

### 3. Accepted RL actions hug the Lyapunov boundary

In the latest RL mixed cases, accepted candidate steps in the final 200-step tail satisfy contraction with extremely small margin:

- pretrained mixed accepted-step max margin: about `-4.5e-06`
- cold-start mixed accepted-step max margin: about `-2.0e-05`

By contrast, the direct mixed baseline in the same tail has:

- applied-step max contraction margin: about `-2.87e-03`

So the RL gate is accepting many actions that are only barely contractive relative to the moving target. That allows the policy to slide along the admissible manifold instead of settling.

This is why RL mixed still jitters even though direct mixed is almost calm.

## RL-Specific Chattering Mechanism

The RL gate is memoryless at the decision level:

- every step independently decides accepted RL versus fallback direct MPC
- there is no dwell time
- there is no hysteresis
- there is no "stay in fallback for N steps after takeover" rule

That creates classic mode chatter when the candidate margin crosses zero repeatedly.

In the final 200-step tail:

- pretrained no-anchor: `29` switches
- pretrained mixed: `20` switches
- cold-start no-anchor: `18` switches
- cold-start mixed: `15` switches

The fallback counts are not huge, but the switching is enough to inject extra input motion on top of the base selector motion.

## Why `optimal_inaccurate` Is Not The Main Cause

The latest six-case comparison makes this point stronger than before.

1. Direct no-anchor jitters with `200/200` tail steps solved as `optimal`.
2. Pretrained no-anchor still jitters strongly even when the final tail contains no `optimal_inaccurate`.
3. Mixed RL still shows residual jitter even when `optimal_inaccurate` is absent in the same tail.

So `optimal_inaccurate` is a numerical-warning symptom, not the primary jitter mechanism.

## Why `rho = 0.99` Is Not The Main Differentiator

All six latest cases use the same:

- first-step contraction logic
- hard Lyapunov mode
- `rho_lyap = 0.99`

Yet their jitter levels differ dramatically.

So `rho = 0.99` may still be permissive, but it is not what separates calm from jittery behavior in the latest results. The decisive variables are:

- target motion
- target/objective mismatch
- RL boundary-hugging accepted actions
- switching without hysteresis

## Bugs And Inconsistencies To Fix

### 1. Core formulation mismatch

The direct solver certifies around `y_s` but tracks `y_sp`.

Fix:

- rerun the direct baseline first with `direct_tracking_use_target_output=True`
- then propagate that choice consistently into the RL fallback and BC teacher

### 2. RL observation mismatch

The policy does not observe `y_s` even though the gate judges actions relative to `y_s`.

Fix:

- for cold-start RL, add a `y_s`-aware observation mode such as `[xhatdhat, y_sp, y_s, u_prev]` or `[xhatdhat, y_sp, y_s-y_sp, u_prev]`
- for pretrained RL, do not retrofit the old checkpoint blindly; create a separate experiment with consistent pretraining

### 3. No hysteresis around RL/fallback switching

Fix:

- add fallback dwell or hysteresis
- examples: stay in fallback for `N=3` to `5` steps after takeover, or require a stronger recovery margin before returning to RL

### 4. Boundary-hugging acceptance

The accepted RL actions are often only barely contractive.

Fix:

- require a stricter negative margin near steady state
- or add a secondary acceptance check on output-target improvement or input-motion suppression

### 5. Replay-boundary state inconsistency

In `Simulation/run_rl_lyapunov.py`, the replay `next_state` intentionally reuses `y_sp_k` instead of `y_sp_{k+1}` at setpoint boundaries.

The code comment explains the intent, but algorithmically this means the stored transition is not the literal environment next state at a task boundary.

Fix options:

- mark setpoint-change boundaries as terminal in replay
- or store the actual `y_sp_{k+1}` in `next_state`
- or keep the current convention but split the dataset explicitly by task boundary

This is not the main steady-tail jitter source, but it is a genuine learning-loop inconsistency.

### 6. Missing selector status logging

The RL debug tables do not preserve a usable selector solver-status field, which makes it impossible to attribute a CVXPY warning cleanly to:

- the target selector
- the fallback direct MPC
- or another solver path

Fix:

- log selector solver name, status, and message into the RL step table for every step

### 7. Result-label bug

The latest direct no-RL run still labels the mixed case as `0p1` even though the stored weight arrays are `0.25`.

Fix:

- rerun the direct baseline after the naming patch so exported case names match the actual weights

## Recommended Fix Order

The fix order should follow the mechanism hierarchy.

1. Fix the core direct-method mismatch:
   set `direct_tracking_use_target_output=True` and re-evaluate the direct no-RL baseline first.

2. Keep mixed anchoring as the default selector mode:
   `u_prev + x_s_prev`.

3. Add switching hysteresis in the RL gate.

4. Tighten the RL acceptance rule near steady state.

5. Introduce a `y_s`-aware state representation for cold-start RL.

6. Clean up replay-boundary semantics and selector logging.

## Bottom Line

The latest jitter is not caused mainly by solver warnings.

The deep mechanism is:

$$
\\text{moving admissible target}
\\; + \\;
\\text{tracking/certification mismatch}
\\; + \\;
\\text{boundary-hugging RL actions}
\\; + \\;
\\text{memoryless accepted/fallback switching}
$$

The direct no-RL baseline proves that the first two terms are already enough to create jitter. The RL wrapper then adds the last two terms on top.

## Figure Links

- [latest_three_method_seg2_input_motion.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_seg2_input_motion.png>)
- [latest_three_method_seg2_error_decomposition.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_seg2_error_decomposition.png>)
- [latest_three_method_tail_summary.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-14_three_method_jitter_analysis/latest_three_method_tail_summary.png>)
