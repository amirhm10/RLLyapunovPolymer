# Latest RL Jitter Root-Cause Analysis

## Scope

This note analyzes the latest completed RL direct-Lyapunov exports from May 13, 2026:

- pretrained RL: `Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260513_191437`
- cold-start RL: `Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260513_191435`

Both runs contain two active selector variants:

- `bounded_hard`: no selector anchoring
- `bounded_hard_u_prev_0p1_xs_prev_0p1`: stale label, but the actual active weights in these runs are `u_ref_weight = 0.25` and `x_ref_weight = 0.25`

The main question is whether the controller jitter is caused by the `optimal_inaccurate` warning path, or by another mechanism in the RL-plus-direct-Lyapunov loop.

## Method

I analyzed the saved `step_table.csv` and `arrays.npz` bundles and focused on the last episode. The last episode contains two constant-setpoint plateaus of 800 steps each, so the second plateau is the cleanest place to study true steady behavior.

For each case I computed:

- step-to-step safe input motion, `$\\|\\Delta u_{\\mathrm{safe}}\\|_\\infty$`
- step-to-step target-input motion, `$\\|\\Delta u_s\\|_\\infty$`
- output error to the admissible target, `$\\|y-y_s\\|_\\infty$`
- admissible-target mismatch, `$\\|y_s-y_{sp}\\|_\\infty$`
- fallback rate and mode-switch count
- `optimal_inaccurate` counts

The output deviation signal was reconstructed from the stored tracking error as

$$
y_k = e_k + y_{sp,k},
$$

using the saved `e_store` and `y_sp` arrays.

## Main Finding

The latest RL jitter is **not** primarily caused by `optimal_inaccurate`.

The dominant mechanism is:

1. the no-anchor selector keeps moving the admissible steady target `(x_s,u_s,y_s)` even when the raw setpoint is constant,
2. the RL policy and fallback tracking problem still act on raw `y_sp`, not `y_s`,
3. this creates repeated RL/fallback switching, and the fallback moves are much larger than the accepted RL moves.

`optimal_inaccurate` is at most a secondary amplifier.

## Strongest Evidence

The clearest evidence comes from the **final 200 steps of the second constant-setpoint plateau** of the last episode.

| Case | Tail mean `$\\|\\Delta u_{safe}\\|_\\infty$` | Tail mean `$\\|\\Delta u_s\\|_\\infty$` | Tail mean `$\\|y-y_s\\|_\\infty$` | Tail mean `$\\|y_s-y_{sp}\\|_\\infty$` | Tail fallback rate | Tail `optimal_inaccurate` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pretrained, no anchor | 0.605 | 3.028 | 0.392 | 0.443 | 7.5% | 0 |
| Pretrained, mixed 0.25 | 0.0000005 | 0.0000005 | 0.00010 | 0.0429 | 0.0% | 0 |
| Cold start, no anchor | 0.505 | 3.143 | 0.486 | 0.529 | 8.5% | 2 |
| Cold start, mixed 0.25 | 0.180 | 1.088 | 0.178 | 0.178 | 8.0% | 0 |

Two conclusions follow immediately:

- The pretrained no-anchor case still jitters at the end even though it has **zero** `optimal_inaccurate` steps in that final 200-step window.
- The pretrained mixed-anchor case settles almost perfectly even though its target stage is still `frozen_output_disturbance_bounded_ls` in all 200 tail steps.

So:

- the warning path is not the main cause,
- and bounded-LS target mode by itself is not the problem.

The problem is whether the bounded-LS target keeps moving.

## Mechanism 1: The No-Anchor Selector Produces A Moving Target

The no-anchor case keeps changing the admissible target even on a constant setpoint plateau.

In the same final 200-step window:

- pretrained no-anchor: `$\\text{mean }\\|\\Delta u_s\\|_\\infty = 3.028$`
- pretrained mixed: `$\\text{mean }\\|\\Delta u_s\\|_\\infty \\approx 5 \\times 10^{-7}$`

That is the most important difference in the latest results.

The target-selector code explains why. In `Lyapunov/frozen_output_disturbance_target.py`, the bounded target is computed by `solve_bounded_steady_state_least_squares(...)`. In `analysis/steady_state_debug_analysis.py:763-833`, the bounded least-squares problem minimizes residual plus optional regularization terms toward `u_ref` and `x_ref`. If the selector weights are zero, the bounded solution is free to slide along the feasible least-squares manifold as the observer estimate changes.

With mixed anchoring, those regularizers are active, so the selector stops wandering:

- input anchor term toward `u_prev`
- state anchor term toward `x_s_prev`

This is exactly what the latest data show.

## Mechanism 2: The Gate Certifies Contraction Around `y_s`, But The RL/Fallback Loop Still Tracks `y_sp`

The implementation is still split across two objectives:

- the candidate RL action is accepted or rejected using first-step Lyapunov contraction around the current admissible target `(x_s,u_s)`, via `evaluate_candidate_action(...)` in `Simulation/run_rl_lyapunov.py:1408-1421`
- if the RL action is rejected, the fallback direct MPC is solved in `solve_direct_tracking_from_target(...)`

However, the fallback tracking target is set by:

```python
y_target = y_s.copy() if use_target_output_for_tracking else y_sp_k.copy()
```

in `Lyapunov/direct_lyapunov_mpc.py:802`, and the active RL notebooks still run with:

- `direct_tracking_use_target_output=False`

as shown in:

- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

So the latest RL loop still does this:

- safety certificate: contract toward `x_s`
- fallback objective: track raw `y_sp`
- policy training and BC: also oriented around raw `y_sp`

That mismatch is visible numerically in the no-anchor case. On the final 200 steps of the pretrained no-anchor run:

- `$\\text{mean }\\|y-y_s\\|_\\infty = 0.392$`
- `$\\text{mean }\\|y_s-y_{sp}\\|_\\infty = 0.443$`
- `$\\text{mean }\\|y-y_{sp}\\|_\\infty = 0.176$`

So the plant is noticeably closer to raw `y_sp` than to the admissible target `y_s`. This means the loop is not actually settling around the Lyapunov target being used in the contraction test.

By contrast, on the pretrained mixed-anchor tail:

- `$\\text{mean }\\|y-y_s\\|_\\infty = 1.0 \\times 10^{-4}$`
- `$\\text{mean }\\|y_s-y_{sp}\\|_\\infty = 4.29 \\times 10^{-2}$`
- `$\\text{mean }\\|y-y_{sp}\\|_\\infty = 4.30 \\times 10^{-2}$`

There the plant is essentially sitting on `y_s`, and `y_s` itself is nearly fixed, so the mismatch stops causing chatter.

## Mechanism 3: Jitter Is A Switching Problem, Not Just A Solver-Accuracy Problem

The no-anchor case shows short repeated alternations between accepted RL and fallback MPC.

Final 200 steps of the second plateau:

| Case | Tail switches | Tail fallback run mean | Accepted-step mean `$\\|\\Delta u_{safe}\\|_\\infty$` | Fallback-step mean `$\\|\\Delta u_{safe}\\|_\\infty$` |
| --- | ---: | ---: | ---: | ---: |
| Pretrained, no anchor | 20 | 1.5 | 0.392 | 3.236 |
| Pretrained, mixed 0.25 | 0 | 0.0 | 0.0000005 | n/a |
| Cold start, no anchor | 26 | 1.55 | 0.357 | 2.194 |
| Cold start, mixed 0.25 | 15 | 2.0 | 0.117 | 0.896 |

This shows two distinct contributions:

- accepted RL actions create the persistent background wobble,
- fallback MPC creates the larger spikes.

For pretrained no-anchor, the tail move budget splits roughly as:

- accepted RL moves: 59.9% of total `$\\sum \\|\\Delta u_{safe}\\|_\\infty$`
- fallback MPC moves: 40.1%

So the jitter is not a pure fallback artifact. Most steps are still accepted RL steps, but the target is moving underneath them, and the fallback corrections add visible bursts on top.

## Why `optimal_inaccurate` Is Not The Main Cause

The data reject that explanation directly.

1. In the final 200 steps of the pretrained no-anchor run, `optimal_inaccurate = 0`, but the controller still jitters strongly.
2. The cold-start mixed case has `optimal_inaccurate = 0` in the same window, but still has some residual switching and motion.
3. Over the full last episode, the inaccurate counts are small relative to the total:
   - pretrained no-anchor: 3 of 1600
   - cold-start no-anchor: 5 of 1600
4. The fallback moves on ordinary `optimal` solves are already large enough to generate visible spikes.

Therefore the warning can still matter as a numerical-quality flag, but it is not the root cause of the observed end-of-run jitter.

## Why `rho = 0.99` Is Not The Main Differentiator

Both no-anchor and mixed-anchor runs use the same:

- first-step contraction check
- hard Lyapunov mode
- `rho_lyap = 0.99`

Yet the pretrained mixed-anchor case settles essentially perfectly in the final plateau, while the no-anchor case does not.

So `rho = 0.99` may still be permissive in an absolute sense, but it is not what separates the jittery and non-jittery latest cases. The decisive difference is target motion and target/objective mismatch.

## Exact Versus Bounded Target Stage

The latest results also show that exact-target recovery is not required for smooth settling.

On the final 200 steps of the second plateau:

- pretrained mixed-anchor: all 200 steps are `frozen_output_disturbance_bounded_ls`
- yet the target and safe input are essentially frozen, and the plant sits on `y_s`

So the relevant distinction is not:

- exact bounded target versus bounded LS target

It is:

- moving bounded-LS target versus stable bounded-LS target

## Root-Cause Statement

The latest RL jitter is best explained as:

$$
\\text{moving admissible target}
\\; + \\;
\\text{raw-setpoint tracking objective mismatch}
\\; + \\;
\\text{short repeated RL/fallback switching}
$$

not by `optimal_inaccurate`.

In the current implementation, no-anchor target selection lets `(x_s,u_s,y_s)` wander. The RL policy and fallback controller keep acting against raw `y_sp`, while the gate certifies only one-step contraction around the moving admissible target. That combination creates repeated mode switches and visible input jitter.

## What To Fix First

The data suggest this priority order.

1. Keep mixed anchoring as the default selector configuration.
   It is the single clearest stabilizing change in the latest runs.

2. Align the fallback tracking objective with the certified target.
   The current implementation still uses `direct_tracking_use_target_output=False`, which preserves the `y_sp` versus `y_s` mismatch.

3. Add a hysteresis or dwell rule around fallback entry/exit.
   The tail switch counts and short fallback bursts show classic chattering behavior.

4. Only after that, revisit solver-accuracy tuning.
   `optimal_inaccurate` is worth monitoring, but the latest runs show it is not the first-order issue.

## Figures

Figure 1 shows the second constant-setpoint plateau of the last episode. It separates target motion, safe-input motion, fallback intervals, and the rare `optimal_inaccurate` events. The key visual result is that the no-anchor cases remain active even when `optimal_inaccurate` is absent, while the pretrained mixed-anchor case becomes essentially flat.

![Latest RL jitter mechanism timeseries](figures/2026-05-13_rl_jitter_root_cause/latest_rl_seg2_mechanism_timeseries.png)

Figure 2 summarizes the same tail window numerically. It highlights that the strongest reduction in jitter comes from freezing the selector target motion, not from reducing inaccurate-solver counts.

![Latest RL tail summary](figures/2026-05-13_rl_jitter_root_cause/latest_rl_seg2_tail_summary.png)

Direct file links:

- [latest_rl_seg2_mechanism_timeseries.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-13_rl_jitter_root_cause/latest_rl_seg2_mechanism_timeseries.png>)
- [latest_rl_seg2_tail_summary.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/figures/2026-05-13_rl_jitter_root_cause/latest_rl_seg2_tail_summary.png>)
