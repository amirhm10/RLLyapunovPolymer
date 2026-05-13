# Pretrained Direct Safety Gate `rho=0.99` Last-Episode Target And Offset Analysis

Date: 2026-05-13

## Objective

This note analyzes the latest pretrained direct safety-gate RL study with `rho_lyap = 0.99`:

- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb)
- export run:
  [Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313)

The main questions are:

1. Why can the loop oscillate after it has already reached the setpoint region?
2. Why can we still see offset if `$y_{\mathrm{sp}}$` is used in the steady-state target calculation?
3. In the last episode, is `$y_s = y_{\mathrm{sp}}$` or not for each scenario?

## Files inspected

- [Simulation/run_rl_lyapunov.py](../Simulation/run_rl_lyapunov.py)
- [Lyapunov/direct_lyapunov_mpc.py](../Lyapunov/direct_lyapunov_mpc.py)
- [Lyapunov/lyapunov_core.py](../Lyapunov/lyapunov_core.py)
- [Lyapunov/frozen_output_disturbance_target.py](../Lyapunov/frozen_output_disturbance_target.py)
- [analysis/steady_state_debug_analysis.py](../analysis/steady_state_debug_analysis.py)
- latest comparison summary:
  [comparison_summary.json](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313/comparison_summary.json)
- latest case summaries:
  [bounded_hard summary](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313/sf_5aabb97c/summary.json)
  [bounded_hard_u_prev_0p1 summary](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313/sf_ce75a180/summary.json)
  [bounded_hard_xs_prev_0p1 summary](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313/sf_e31c7b57/summary.json)
  [bounded_hard_u_prev_0p1_xs_prev_0p1 summary](../Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260512_071313/sf_41d4b0ae/summary.json)
- new focused figures:
  [Episode 200 second segment outputs vs targets](./figures/2026-05-13_pretrained_rho99_last_episode/episode200_seg2_outputs_vs_targets.png)
  [Episode 200 tail 100 outputs vs targets](./figures/2026-05-13_pretrained_rho99_last_episode/episode200_seg2_tail100_outputs_vs_targets.png)

## 1. Current method reminder

The current pretrained run uses:

- RL observation built from `xhat_aug`, raw `y_sp`, and previous input
- `projection_backend = "direct_accept_or_fallback"`
- direct target selector from the frozen output-disturbance path
- direct fallback tracking with `direct_tracking_use_target_output = False`
- one-step Lyapunov acceptance with `rho = 0.99` and `eps_lyap = 1e-9`

So the actor still thinks in terms of raw `y_sp`, while the Lyapunov check is evaluated around the selected admissible target `(x_s, u_s)` with output `y_s`.

## 2. Whole-run four-scenario summary

| Case | Reward mean | Output RMSE mean | Accepted rate | Fallback rate |
| --- | ---: | ---: | ---: | ---: |
| `bounded_hard` | -6.782 | 0.417 | 0.935 | 0.0597 |
| `bounded_hard_u_prev_0p1` | -4.712 | 0.394 | 0.984 | 0.0143 |
| `bounded_hard_xs_prev_0p1` | -5.697 | 0.394 | 0.938 | 0.0593 |
| `bounded_hard_u_prev_0p1_xs_prev_0p1` | -4.719 | 0.383 | 0.985 | 0.0134 |

Main high-level conclusion:

- the best overall cases are still the ones with previous-input anchoring
- `x_s`-only regularization improves some tracking metrics, but it does not suppress fallback or oscillation as effectively as `u_prev`
- the combined case is the cleanest overall, but its late offset is not zero

## 3. Episode 200 target-stage breakdown

Episode 200 is the last saved episode and contains two 400-step setpoint segments.

The direct target selector has two relevant solve stages:

- `frozen_output_disturbance_exact_bounded`
  this means the exact steady target was within bounds, so the solver returned the exact target
- `frozen_output_disturbance_bounded_ls`
  this means the exact steady target was not within bounds, so the solver returned the bounded least-squares compromise target

### Episode 200 stage counts

| Case | Exact-bounded steps | Bounded-LS steps | Tail-100 exact steps |
| --- | ---: | ---: | ---: |
| `bounded_hard` | 32 | 768 | 4 |
| `bounded_hard_u_prev_0p1` | 211 | 589 | 53 |
| `bounded_hard_xs_prev_0p1` | 118 | 682 | 7 |
| `bounded_hard_u_prev_0p1_xs_prev_0p1` | 18 | 782 | 0 |

This already answers most of the `$y_s = y_{\mathrm{sp}}$` question:

- if the selector is in `exact_bounded`, then `$y_s = y_{\mathrm{sp}}$`
- if it is in `bounded_ls`, then `$y_s$` is only the best admissible compromise and is not guaranteed to equal `$y_{\mathrm{sp}}$`

## 4. Last-episode tail diagnostics

The second 400-step segment of episode 200 is the most useful place to inspect the final offset, because it is the last setpoint in the whole run. The table below focuses on the last 100 steps of that segment.

Definitions:

- `mean |y-y_sp|_inf`: mean per-step raw-setpoint error
- `mean |y-y_s|_inf`: mean per-step tracking error relative to the selected target
- `mean |y_s-y_sp|_inf`: mean per-step target mismatch

### Tail-100 summary

| Case | `mean |y-y_sp|_inf` | `mean |y-y_s|_inf` | `mean |y_s-y_sp|_inf` | Tail fallback count |
| --- | ---: | ---: | ---: | ---: |
| `bounded_hard` | 0.268 | 0.655 | 0.770 | 7 |
| `bounded_hard_u_prev_0p1` | 0.056 | 0.157 | 0.162 | 11 |
| `bounded_hard_xs_prev_0p1` | 0.167 | 0.424 | 0.471 | 7 |
| `bounded_hard_u_prev_0p1_xs_prev_0p1` | 0.135 | 0.001 | 0.135 | 3 |

This is the key separation:

- `bounded_hard_u_prev_0p1_xs_prev_0p1`:
  the loop tracks `$y_s$` almost perfectly in the tail, so the remaining offset is almost entirely `$y_s - y_{\mathrm{sp}}$`
- `bounded_hard`:
  the loop is not only offset from `$y_{\mathrm{sp}}$`; it is also still moving around `$y_s$`
- `bounded_hard_xs_prev_0p1`:
  same basic problem as `bounded_hard`, although somewhat milder
- `bounded_hard_u_prev_0p1`:
  this is the closest case to raw-setpoint settling; the remaining tail offset is small and split between a modest target mismatch and a modest residual tracking error

## 5. Is `$y_s = y_{\mathrm{sp}}$` in the last episode?

Short answer: not consistently, and in the cleanest combined case the answer is essentially no in the final tail.

### `bounded_hard`

- In episode 200, 768 of 800 steps use bounded least squares.
- In the last 100 steps, only 4 steps are exact-bounded.
- Tail mismatch stays large:
  `mean |y_s-y_sp|_inf = 0.770`.

Conclusion:

- for this case, `$y_s \neq y_{\mathrm{sp}}$` for most of the last episode

### `bounded_hard_u_prev_0p1`

- In episode 200, 211 of 800 steps are exact-bounded.
- In the last 100 steps, 53 of 100 steps are exact-bounded.
- Tail mismatch is much smaller:
  `mean |y_s-y_sp|_inf = 0.162`.

Conclusion:

- this case spends a substantial fraction of the tail with `$y_s = y_{\mathrm{sp}}$`
- but not all of it, especially on the second output

### `bounded_hard_xs_prev_0p1`

- In episode 200, 682 of 800 steps are bounded least squares.
- In the last 100 steps, only 7 of 100 steps are exact-bounded.
- Tail mismatch remains significant:
  `mean |y_s-y_sp|_inf = 0.471`.

Conclusion:

- for most of the last episode, `$y_s \neq y_{\mathrm{sp}}$`

### `bounded_hard_u_prev_0p1_xs_prev_0p1`

- In episode 200, 782 of 800 steps are bounded least squares.
- In the last 100 steps, all 100 steps are bounded least squares.
- Tail mismatch is stable and nonzero:
  `mean |y_s-y_sp|_inf = 0.135`.
- At the same time, `mean |y-y_s|_inf = 0.001`.

Conclusion:

- in the final tail, `$y_s \neq y_{\mathrm{sp}}$`
- but the plant is tracking `$y_s$` almost exactly
- so the residual steady-state offset is a target-selection issue, not an oscillation issue

## 6. Why can the loop oscillate after reaching the setpoint region?

There are two distinct reasons.

### 6.1 One-step contraction is not the same as non-oscillatory output settling

The gate checks

$$
V_{k+1}^{\mathrm{cand}} \le \rho V_k + \varepsilon_{\mathrm{lyap}}
$$

with `$V_k = (\hat x_k - x_s)^\top P_x (\hat x_k - x_s)$`.

That means:

- the certificate is about physical-state contraction around `$x_s$`
- it is not a direct monotonicity certificate on each output
- it is not a direct penalty on alternating inputs
- with `rho = 0.99`, the gate allows fairly slow decay

So the RL action can still chatter or oscillate in output space while remaining Lyapunov-admissible in this one-step sense.

### 6.2 The actor tracks raw `$y_{\mathrm{sp}}$`, but the gate certifies around `$y_s$`

The actor state still contains raw `$y_{\mathrm{sp}}$`.
The fallback direct MPC also tracks raw `$y_{\mathrm{sp}}$`.
But the safety gate is built around the selected target `(x_s, u_s)` and output `$y_s$`.

If `$y_s \neq y_{\mathrm{sp}}$`, then the controller stack is solving two different objectives:

1. RL and fallback tracking try to reduce raw setpoint error.
2. The Lyapunov gate certifies contraction around the admissible target.

This mismatch creates a persistent forcing mechanism near the setpoint region.

### 6.3 What the last-episode numbers say

For `bounded_hard` in the last 100 steps:

- accepted rate is 100%
- `mean |y-y_s|_inf = 0.655`
- `input_std` is still large
- `mean_abs_du` is still large

So the late oscillation there is not because the gate is inactive.
It is because the RL policy is still moving aggressively, the gate still accepts those moves, and the selected target is itself shifted away from raw `$y_{\mathrm{sp}}$`.

For the combined case, those symptoms disappear:

- `mean |y-y_s|_inf = 0.001`
- `input_std` is almost zero
- `mean_abs_du` is almost zero

So that case is not oscillating in the tail. It is simply converging to the wrong steady output target.

## 7. Why do we get offset even though `$y_{\mathrm{sp}}$` is in the target calculator?

Because in bounded mode, `$y_{\mathrm{sp}}$` is the desired steady output, not a hard equality constraint once the exact bounded target is unavailable.

The exact target equations are

$$
(I-A)x_s - Bu_s = 0,
$$

$$
Cx_s = y_{\mathrm{sp}} - \hat d,
$$

$$
d_s = \hat d,
$$

which imply

$$
y_s = y_{\mathrm{sp}}
$$

only when the exact steady input is admissible.

But when the exact target violates bounds, the code switches to bounded least squares:

$$
\min_{x_s,u_s}
\left\|
\begin{bmatrix}
(I-A)x_s - B u_s \\
C x_s - (y_{\mathrm{sp}} - \hat d)
\end{bmatrix}
\right\|_2^2
+
\|u_s-u_{\mathrm{ref}}\|_{W_u}^2
+
\|x_s-x_{\mathrm{ref}}\|_{W_x}^2
$$

subject to

$$
u_{\min} \le u_s \le u_{\max}.
$$

So in that stage:

- output matching is softened into a least-squares term
- the solver is allowed to compromise on `$y_s$`
- the compromise is affected by input bounds and the active anchor weights

That is exactly what happens in the late tail of the combined case.

## 8. Direct answers to the three questions

### Why does it oscillate after reaching the setpoint region?

- Because the gate only enforces one-step contraction in `$V(x-x_s)$`, not monotone output settling.
- Because `rho=0.99` is loose enough to allow slow decay with alternating moves.
- Because RL and fallback still push on raw `$y_{\mathrm{sp}}$`, while the gate certifies around `$y_s$`.
- In the unanchored and `x_s`-only cases, the input motion remains visibly active near the end.

### Why is there offset if `$y_{\mathrm{sp}}$` is in the target calculator?

- Because bounded target selection is not a hard equality solve once the exact bounded steady target is unavailable.
- In `bounded_ls`, `$y_s$` is the best admissible compromise target, not necessarily the raw setpoint.

### In the last episode, is `$y_s = y_{\mathrm{sp}}$`?

- `bounded_hard`: mostly no
- `bounded_hard_u_prev_0p1`: partly, and this is the closest case to yes
- `bounded_hard_xs_prev_0p1`: mostly no
- `bounded_hard_u_prev_0p1_xs_prev_0p1`: essentially no in the final tail, but the plant then tracks `$y_s$` almost perfectly

## 9. Most important interpretation

The cleanest late-episode case is actually the combined anchored case, not because it eliminates offset, but because it reveals what the offset really is.

In the final 100 steps of `bounded_hard_u_prev_0p1_xs_prev_0p1`:

- the plant is almost exactly at `$y_s$`
- the input is almost no longer moving
- the remaining offset is therefore almost exactly `$y_s - y_{\mathrm{sp}}$`

So the final issue in that case is not RL oscillation anymore. It is target mismatch.

By contrast, `bounded_hard` still has both problems at once:

- target mismatch
- and residual oscillatory tracking around that target

## 10. Recommended next checks

1. Export a compact last-episode table with `$y$`, `$y_s$`, `$y_{\mathrm{sp}}$`, `$u$`, `$u_s$`, and `target_stage` for every case.
2. Run the same four pretrained cases with `direct_tracking_use_target_output=True` to isolate raw-target mismatch from RL oscillation.
3. Add a stagnation or offset-triggered override on top of the one-step Lyapunov check, because contraction alone is not enough to detect near-setpoint chattering.
4. For the pretrained setup, keep the actor observation unchanged, but log whether accepted RL actions are repeatedly moving away from raw-setpoint settling while still satisfying contraction.
