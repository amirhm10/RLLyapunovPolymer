# Pretrained Handoff Critic Calibration

## Summary

Pretrained online TD3 runners now use a longer, actor-frozen handoff phase. The executed behavior is still the teacher-policy blend, but the TD3 policy-gradient actor update is delayed until after handoff so the freshly reset critic can calibrate on online blended-action transitions first.

## Motivation

The pretrained critic-reset batch showed that BC was not the weak point: during BC, replay receives executed teacher-plus-noise transitions, the critic is trained with TD targets, and the actor is supervised toward the clean teacher action. The localized OF-MPC handoff spike was more consistent with enabling full TD3 actor-gradient updates too early, while the critic was still adapting to the handoff distribution.

## Code Changes

- Pretrained online runners now use `handoff_episodes = 10`.
- Cold-start runners keep the previous 5-episode handoff for comparability.
- Added explicit phase metadata:
  - `bc_update_mode = "critic_td_plus_actor_bc"`
  - pretrained `handoff_update_mode = "critic_td_plus_actor_bc"`
  - cold-start `handoff_update_mode = "td3_full"`
  - pretrained `handoff_actor_bc_updates_per_step = 1`
- During pretrained handoff:
  - replay stores the executed blended action
  - critic TD learning remains active
  - actor BC toward the clean teacher action remains active
  - full TD3 actor-gradient updates are disabled
- Step-table exports now include:
  - `critic_td_update_active`
  - `actor_bc_update_active`
  - `td3_full_update_active`
  - `actor_bc_updates_per_step`

## Expected Effect

The handoff phase should become a calibration bridge:

$$
u_k^{\mathrm{exec}} =
\alpha_k u_k^{\mathrm{teacher}} + (1-\alpha_k)u_k^{\pi},
\qquad
\alpha_k \downarrow 0.
$$

During this bridge, the critic learns from the online reward and the blended behavior distribution, while the actor remains anchored by supervised teacher actions. TD3 actor-gradient updates begin only after the handoff distribution has been added to replay.

## Validation

Use low-cost validation first:

```powershell
python -m py_compile Simulation/run_rl_lyapunov.py Lyapunov/safety_debug.py utils/online_disturbance_runner.py
```

Then smoke-test one pretrained safety-gate and one pretrained no-gate runner:

```powershell
python OnlineTD3_LMPCPretrained_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
python OnlineTD3_OFMPCPretrained_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
```

Acceptance checks:

- pretrained configs show `handoff_episodes = 10`
- pretrained configs show `handoff_update_mode = "critic_td_plus_actor_bc"`
- handoff rows in `step_table.csv` show `td3_full_update_active = False`
- handoff rows show `critic_td_update_active = True` and `actor_bc_update_active = True`
- full-RL rows after handoff show `td3_full_update_active = True`

