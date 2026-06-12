# Online TD3 Next-Step Strategy

Date: 2026-06-12

## Executive Recommendation

The best next move is not a full critic reset and not a larger network yet. The
most suited path for this online training process is:

1. Run the newly cleaned BC/handoff schedule once as the new baseline.
2. Add a **pretrained actor + critic-only online recalibration** variant.
3. If the critic remains unstable or the policy gradient looks mis-scaled, try a
   **critic last-layer reset + critic-only recalibration**.
4. Add **visited-state teacher relabeling** with decayed BC strength, especially
   for LMPC-pretrained safety-gate runs.
5. Treat full critic reset and third-critic designs as later ablations.

Reason: the current results suggest two different problems:

- The **critic value scale** is likely mismatched because offline pretraining used
  one-step MPC quadratic rewards, while online TD3 uses shaped closed-loop
  rewards and safety penalties.
- The **actor imitation distribution** is likely mismatched because offline LMPC
  labels are sampled over a broad synthetic box, while online rollouts visit a
  narrower and policy-dependent state distribution.

So the strongest practical strategy is to keep the useful pretrained actor, adapt
the critic to the online reward before actor policy-gradient updates dominate,
and keep teaching the actor on states it actually visits.

## Current Implementation Facts

The current online loop already supports several useful pieces:

- BC phase executes a teacher policy and stores the executed transition in the
  critic replay buffer.
- Actor demo targets are stored separately with `push_actor_demo(...)`.
- During BC, `train_step(actor_update=False)` updates the critic only.
- During BC, `train_actor_bc_step()` trains the actor toward the clean teacher
  action.
- `bc_teacher_gap_inf` is logged whenever a teacher action is available.
- The new schedule uses clean or tiny-noise BC and small policy-side handoff
  noise before full-RL exploration starts.

What is not yet implemented:

- A pure actor-frozen critic recalibration phase with actor BC disabled.
- Critic last-layer reset or full critic reset controls.
- DAgger-style relabeling after BC/handoff.
- Decayed BC weight during full RL.
- Automatic re-enable of teacher relabeling when `bc_teacher_gap_inf` grows.

## Why Critic Recalibration Matters

Offline TD3 pretraining trains the critic on rewards of the form

$$
r_k^{\mathrm{offline}}
= -\left(
\|y_{k+1}-r_k\|_{Q_y}^2
+ \|u_k-u_{k-1}\|_{R_{\Delta u}}^2
\right),
$$

with $Q_y=[5,1]$ and $R_{\Delta u}=[1,1]$.

Online TD3 instead uses the shaped reward family:

$$
r_k^{\mathrm{online}}
= r_{\mathrm{tracking}}
+ r_{\mathrm{band}}
- r_{\mathrm{move}}
- r_{\mathrm{fallback}},
$$

where the fallback penalty is active only for safety-gate runners. This changes
both the scale and local gradients seen by the actor. A pretrained critic can
therefore be useful as a feature extractor but still produce a misleading actor
gradient immediately after loading.

Critic-only recalibration asks the critic to learn

$$
Q^\pi(s_k,a_k)
\approx r_k^{\mathrm{online}}
+ \gamma Q^{\pi}_{\mathrm{target}}(s_{k+1},\pi_{\mathrm{target}}(s_{k+1}))
$$

before allowing the actor to chase that critic.

## Why DAgger-Style Relabeling Matters

Offline LMPC labels were generated from broad random synthetic states. Online,
the actor visits a different distribution because its own errors, the safety gate,
disturbances, and setpoint transitions shape the trajectory.

DAgger-style relabeling fixes this by collecting teacher labels on visited states:

$$
\mathcal{D}_{\mathrm{demo}}
\leftarrow
\mathcal{D}_{\mathrm{demo}}
\cup
\{(s_k,\mu_T(s_k))\},
$$

where $\mu_T$ is the selected teacher. The actor loss becomes

$$
J_{\pi}
=
-\mathbb{E}_s[Q_1(s,\pi(s))]
+ \lambda_{\mathrm{BC}}(k)
\mathbb{E}_{(s,a_T)\in\mathcal{D}_{\mathrm{demo}}}
\|\pi(s)-a_T\|^2 .
$$

Then $\lambda_{\mathrm{BC}}(k)$ decays over time instead of abruptly switching
from supervised control to pure TD3.

Important fairness detail: for no-gate comparisons, the online teacher should
remain OF-MPC unless the run is explicitly labeled as an LMPC-supervised no-gate
ablation. Otherwise Direct LMPC would no longer be diagnostic-only.

## Candidate Ranking

### Best First: Pretrained Critics Plus Critic-Only Recalibration

This is the fastest and least disruptive option.

Use when:

- The pretrained actor is useful but online reward scale is different.
- The safety gate or BC teacher can keep early rollouts stable.
- We want to preserve offline critic features while adapting the value scale.

Expected signature if it works:

- Lower early critic loss after BC.
- Less policy degradation after handoff.
- Better `reward_no_penalty` without increasing input movement.
- `bc_teacher_gap_inf` stays flat or shrinks during the transition.

Implementation need:

- Add an explicit phase where actor policy-gradient and actor BC are disabled.
- Execute teacher or current policy depending on the runner design.
- Store online-reward transitions and update critic only.
- Then begin the existing low-noise BC/handoff schedule.

### Strong Second: Critic Last-Layer Reset Plus Recalibration

This is the best compromise if pretrained critic features help but the Q output
scale is wrong.

Use when:

- Pretrained critic-only recalibration still shows poor early actor gradients.
- Q values appear badly scaled relative to realized online returns.
- We want to keep state-action feature extraction but remove the old output map.

The current critic has `q1_network.output_layer` and `q2_network.output_layer`,
so this is a small implementation. Reset both online and target critic output
layers, reinitialize the critic optimizer, then run the same critic-only
recalibration.

### Best For Distribution Shift: Visited-State Relabeling With Decayed BC

This is the most scientifically targeted fix for the LMPC actor gap.

Use when:

- The actor tracks the teacher on offline/random labels but behaves poorly online.
- `bc_teacher_gap_inf` shrinks during BC but grows again in full RL.
- LMPC-pretrained safety-gate runs oscillate or trigger frequent fallback after
  handoff.

Recommended rule:

- During BC and handoff, always store visited-state teacher labels.
- During full RL, keep a decaying BC coefficient.
- If the rolling 95th percentile of `bc_teacher_gap_inf` increases, re-enable
  teacher relabeling for a short window.

Teacher assignment should follow the existing comparison logic:

| Runner family | Relabel teacher |
|---|---|
| LMPC-pretrained safety | Direct LMPC |
| OF-MPC-pretrained safety | OF-MPC |
| cold-start safety | Direct LMPC |
| LMPC-pretrained no-gate | OF-MPC |
| OF-MPC-pretrained no-gate | OF-MPC |
| cold-start no-gate | OF-MPC |

### Later: Full Critic Reset

Full critic reset is clean but slower. It discards offline critic features and
asks the online run to learn the value function almost from scratch.

Use when:

- Pretrained critic and last-layer reset both underperform.
- Critic losses or actor gradients remain unstable after recalibration.
- We can afford longer early training before judging the run.

This is a good diagnostic, but not the first option.

### Experimental Later: Third Fresh Critic

A third fresh critic could help as an uncertainty or disagreement signal, but it
adds complexity before the simpler hypotheses are exhausted.

Use later for:

- estimating disagreement between offline-pretrained and online-fresh critics
- conservative actor updates
- detecting out-of-distribution policy actions

It should not be the next implementation unless the first three options fail.

## Proposed Attempt Sequence

### Attempt 0: New Low-Noise BC/Handoff Baseline

Purpose: isolate whether the previous poor behavior was caused by noisy BC and
handoff.

Run:

- all six online TD3 runners
- same 300 episodes and 400-step blocks
- current bounded-mixed selector
- current clean/tiny BC and small handoff exploration

Decision:

- If safety-gate runs improve and no-gate runs remain strong, keep this schedule.
- If LMPC-pretrained still becomes aggressive after handoff, proceed to Attempt 1.

### Attempt 1: Actor-Frozen Critic Recalibration

Purpose: test whether online value-scale mismatch is the main issue.

Phase design:

1. Load pretrained actor and critic.
2. Freeze actor policy-gradient updates.
3. Disable actor BC updates during recalibration.
4. Execute clean teacher or tiny teacher-noise behavior.
5. Train critic only on online reward transitions.
6. Resume BC/handoff/full-RL schedule.

Recommended starting length:

- pretrained: 5 to 10 cycles
- cold-start: not needed, since there is no pretrained critic to recalibrate

Decision:

- If reward improves and input movement drops, value-scale mismatch was important.
- If there is little change, try last-layer reset.

### Attempt 2: Critic Last-Layer Reset

Purpose: keep critic features but remove stale offline reward scale.

Phase design:

1. Load pretrained actor and critic.
2. Reset `q1_network.output_layer` and `q2_network.output_layer`.
3. Hard-update critic target from critic.
4. Reinitialize critic optimizer.
5. Run actor-frozen critic recalibration.
6. Resume BC/handoff/full RL.

Decision:

- If this beats Attempt 1, the old critic output scale was harmful.
- If it is worse, pretrained critic output was useful and should be kept.

### Attempt 3: DAgger Relabeling With Decayed BC

Purpose: fix online distribution shift.

Phase design:

1. At every visited state where teacher is available, store
   `(state, clean_teacher_action)` in demo buffer.
2. Keep actor BC active with a decaying coefficient:

   $$
   \lambda_{\mathrm{BC}}(k)
   =
   \lambda_0 \exp(-k/\tau_{\mathrm{BC}}).
   $$

3. Continue TD3 critic and actor updates.
4. Re-enable a short relabeling window if `bc_teacher_gap_inf` gets worse.

Decision:

- If actor gap shrinks and fallback rate drops, the problem was distribution
  shift.
- If actor gap shrinks but tracking worsens, teacher action may be too
  conservative for the online reward.

### Attempt 4: Full Critic Reset

Purpose: remove all offline critic assumptions.

Run only if Attempt 1 and Attempt 2 do not improve behavior.

### Attempt 5: Larger Actor/Critic Network

Purpose: test representational capacity.

Only try after attempts above, because the current evidence points first to
reward-scale and distribution-shift effects. If used, compare `[256,256,256]`
against `[512,512,512]` with the same recalibration/relabeling schedule.

## Metrics To Track

Use `reward_no_penalty` for fair control-performance comparison across safety
and no-gate runs.

Primary metrics:

- mean RMSE and IAE for both outputs
- `reward_no_penalty`
- actual fallback/intervention count for safety-gate runners
- diagnostic would-activate count for no-gate runners
- mean input movement
- input saturation or clipping rate

Learning diagnostics:

- critic loss during recalibration
- actor loss and actor BC loss
- `bc_teacher_gap_inf` median and 95th percentile
- `handoff_candidate_gap_inf`
- `behavior_exploration_sigma`

Useful new diagnostic to add:

- mean and standard deviation of critic Q values
- TD target scale
- realized return estimate over the same block

Those would make the critic reset decision much less guessy.

## Final Choice

For the next implementation after the current low-noise rerun, I would implement
Attempt 1 and Attempt 2 together behind a runner flag or preset:

- `critic_recalibration_mode="keep_pretrained"`
- `critic_recalibration_mode="reset_last_layer"`

Then run the LMPC-pretrained safety and OF-MPC-pretrained safety runners first.
Those two will tell us whether the main issue is LMPC actor distribution shift,
critic reward-scale mismatch, or both.

If those runs show the actor still drifts away from the teacher after handoff,
then implement DAgger-style relabeling with decayed BC. That is the most natural
next fix for the LMPC actor gap, but it should be tested after the critic scale
hypothesis because critic recalibration is cheaper and less invasive.
