# RL Settling And Target Selector Next Steps

This report analyzes the latest saved direct safety-gated RL runs and the latest direct no-RL reference runs available in `results/` as of May 18, 2026.

Important scope note: I found latest saved pretrained and cold-start RL bundles only for the disturbed plant. Both RL notebooks currently set `plant_mode = "disturb"`. I did not find a saved nominal RL bundle in `results/`; the only latest nominal bundle I found is the direct no-RL run at `results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260517_213932/`.

## Result Files

RL disturbed runs:

- Pretrained: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260517_223040/sf_41d4b0ae_230655/`
- Cold-start: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260517_223033/sf_41d4b0ae_231425/`

Direct no-RL context:

- Nominal, raw setpoint tracking: `results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260517_213932/`
- Disturbed, raw setpoint tracking: `results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260517_205300/`
- Disturbed, selected target tracking: `results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260517_213039/`

Generated analysis artifacts:

- Metrics by episode: `figures/2026-05-18_rl_settling_target_selector/rl_tail_metrics_by_episode.csv`
- Metrics summary: `figures/2026-05-18_rl_settling_target_selector/rl_tail_metrics_summary.csv`
- Direct context summary: `figures/2026-05-18_rl_settling_target_selector/direct_context_summary.csv`

## Current Method

The current direct safety-gated RL loop is:

1. observe the augmented estimate `xhatdhat`;
2. solve the direct frozen-output-disturbance target selector for the current raw setpoint;
3. let the TD3 actor propose a control move;
4. accept the actor move if it passes the direct Lyapunov gate around the selected target;
5. otherwise solve the direct Lyapunov MPC fallback;
6. train TD3 from the executed behavior.

The important implementation detail is that the Lyapunov certificate is centered on the selected target, but the direct tracking fallback still uses the raw setpoint because `direct_tracking_use_target_output = False` in the RL notebooks.

This is the right choice for the current disturbed target selector. The latest direct no-RL comparison showed that using `ys` as the tracking reference can make the controller track a bad selected target very accurately while missing the requested setpoint badly.

## Episode Trends

![RL episode trends](figures/2026-05-18_rl_settling_target_selector/rl_episode_trends.png)

The cold-start run is better than the pretrained run in the latest disturbed comparison.

| Run | RMSE mean | Reward mean | Accepted | Fallback |
| --- | ---: | ---: | ---: | ---: |
| Pretrained disturbed | 0.512 | -11.225 | 97.0% | 2.8% |
| Cold-start disturbed | 0.255 | -3.059 | 96.5% | 3.3% |

The pretrained agent is surprisingly worse. This likely means the old checkpoint is mismatched to the current direct target/fallback setup, reward shaping, setpoint schedule, or disturbance scenario. The cold-start agent learns a better policy for this exact gate, even though it begins without the previous checkpoint.

## Tracking Versus Maintaining

![RL last two subepisodes tracking](figures/2026-05-18_rl_settling_target_selector/rl_last_two_subepisodes_tracking.png)

The latest RL runs do often reach the setpoint region. The issue is more specific: they do not settle into a clean low-effort regulation mode. In the last 20 subepisodes:

| Run | Tail eta RMSE | Tail T RMSE | Tail move mean | Fallback rate |
| --- | ---: | ---: | ---: | ---: |
| Pretrained disturbed | 0.020 | 0.132 | 5.834 | 4.6% |
| Cold-start disturbed | 0.027 | 0.134 | 11.358 | 7.1% |

So the late tracking error is not catastrophic. But the maintenance behavior is still not controller-like enough:

- fallback use increases late in training;
- accepted actor actions dominate, so the gate is mostly checking safety, not performance;
- tail input movement remains high;
- the selected target remains offset from the raw setpoint in many tail windows;
- the cold-start run tracks better overall but moves the input more in the tail.

This supports your observation: RL can get near the setpoint, but it has not learned a stable regulation habit around the setpoint.

## Target Mismatch

![RL tail tracking and target gap](figures/2026-05-18_rl_settling_target_selector/rl_tail_tracking_and_target_gap.png)

The direct target selector remains a major source of ambiguity. In the disturbed direct no-RL runs:

| Direct case | Tracks | RMSE mean | Mean raw error | Mean target error |
| --- | --- | ---: | ---: | ---: |
| Nominal latest | raw `ysp` | 0.355 | 0.408 | 0.408 |
| Disturbed latest | raw `ysp` | 0.436 | 0.564 | 0.564 |
| Disturbed latest | selected `ys` | 1.218 | 3.132 | 0.045 |

![Direct context errors](figures/2026-05-18_rl_settling_target_selector/direct_context_errors.png)

The selected-target run proves the problem: tracking `ys` is not automatically robust. If the target selector drifts, the controller can be Lyapunov-consistent and still scientifically wrong for raw setpoint tracking.

## Why Nominal Is Not Automatically Easy

I expected nominal RL to behave better too, but I cannot verify that from saved RL nominal data because I did not find a nominal RL export.

The latest direct nominal run is not perfect either: it has RMSE mean 0.355 and mean raw error 0.408. That means the two-setpoint schedule plus constraints and target selection are already nontrivial even without parameter drift.

For RL specifically, nominal conditions still may not settle well if:

- the reward mainly rewards getting inside a band but does not strongly reward quiet maintenance;
- exploration noise remains active near the setpoint;
- the safety gate accepts actions that satisfy Lyapunov contraction but increase tracking cost or input movement;
- the actor is trained on many transient samples and too few near-setpoint regulation samples;
- the pretrained checkpoint was learned under a different objective or target convention.

## Making The Target Selector More Robust

The target selector should be treated as a certificate anchor first and a tracking reference only when it passes a quality test.

Recommended target-selector changes:

1. Add a target quality gate.
   Use `ys` as a tracking reference only if `|ys - ysp|` is below a physical tolerance and the target residual is small. Otherwise keep tracking raw `ysp` and use the selected target only for the Lyapunov certificate.

2. Use a lexicographic target solve.
   First minimize output setpoint mismatch. Then, among solutions with nearly the same output mismatch, minimize input movement, state movement, and previous-target movement. This avoids a smoothing penalty pulling the selected target away from the requested setpoint.

3. Penalize target drift conditionally.
   Keep `u_prev` and `xs_prev` regularization, but make it secondary to setpoint fidelity. Regularization should stabilize the selector, not redefine the control objective.

4. Add target rate limits and backup logic.
   If the new selected target jumps too far from the previous selected target, reject it unless it also improves raw setpoint mismatch. This prevents target chasing.

5. Improve the disturbance model.
   The current frozen output-disturbance target is weak for parameter drift in `Qi`, `Qs`, and `hA`. Test Rawlings-style state disturbance through `B`, mixed `B/I` disturbance, or online model re-identification before expecting `ys` to be reliable.

6. Log target admissibility separately from target quality.
   A target can be feasible, bounded, and Lyapunov usable while still being a poor approximation of the requested setpoint. The report tables should always separate target success from raw target quality.

## RL Next Step

The next experiment should isolate the three causes instead of changing everything at once.

### Step 1: Run A Clean Four-Case Matrix

Run and save:

| Agent | Plant | Purpose |
| --- | --- | --- |
| Cold-start | nominal | check if learned regulation works without model mismatch |
| Cold-start | disturbed | current best disturbed RL baseline |
| Pretrained | nominal | test whether checkpoint helps in the easy case |
| Pretrained | disturbed | current pretrained disturbed baseline |

Use separate study names for nominal and disturbed exports so the result folder name does not hide the plant mode.

### Step 2: Add A Maintenance-Aware Reward

Keep the current transient reward, but add a near-setpoint maintenance term:

- when inside the physical tolerance band, penalize input movement more strongly;
- penalize output jitter inside the band;
- add a small dwell bonus for staying inside the band for consecutive steps;
- reduce or disable exploration noise inside the band.

This should make the policy learn "arrive, then hold" instead of "arrive, keep acting."

### Step 3: Add A Performance Guard To The Safety Gate

The current gate is mostly a Lyapunov/safety acceptance test. Add a tracking-cost guard:

- accept the RL action only if predicted one-step raw-setpoint error is no worse than the fallback by a tolerance;
- or accept only if the candidate does not increase a short-horizon raw tracking cost near the setpoint.

This keeps RL from injecting unnecessary motion while still letting it improve transients.

### Step 4: Prefer Residual RL Around Direct MPC

Instead of letting TD3 choose the full action, let it choose a residual around the direct MPC action:

`u = u_direct + delta_u_RL`

Then shrink the residual authority near the setpoint. This matches the empirical situation: direct MPC knows how to stabilize, RL mainly needs to improve transient aggressiveness or compensate residual mismatch.

## Bottom Line

The target selector needs a quality gate and probably a better disturbance model before `ys` can be trusted as a tracking target. For now, raw `ysp` should remain the tracking and reward reference.

The RL problem is not simply "cannot reach the setpoint." The latest data show it often reaches the setpoint region, especially cold-start. The real weakness is maintenance: excessive tail movement, increasing fallback rate, and accepted actions that are safe but not necessarily better for regulation.

The next clean experiment should be a nominal/disturbed, cold/pretrained matrix with a maintenance-aware reward and a tracking-cost guard. That will tell us whether nominal RL is truly failing, or whether the apparent failure is caused by the disturbed target selector and current gate accepting too much near-setpoint motion.
