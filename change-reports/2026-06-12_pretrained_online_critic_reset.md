# Pretrained Online Critic Reset And Tiny BC Noise

## Summary
Updated the online TD3 disturbance runners so pretrained agents keep their loaded actor but start online training with a fresh critic. Pretrained online BC now executes teacher actions with a tiny Gaussian exploration standard deviation of `1e-4`.

## Rationale
- The pretrained actor may still be useful because it encodes LMPC/OF-MPC action structure.
- The pretrained critic is less trustworthy online because it was trained on offline synthetic transitions and offline reward targets, while online training uses closed-loop shaped rewards and safety-gate penalties when applicable.
- Resetting only the critic isolates the useful policy prior from a potentially miscalibrated Q-function prior.
- Tiny BC noise gives the critic nonzero local action variation while the actor-demo target remains the clean teacher action.

## Implementation Notes
- Pretrained runners reset `critic` and `critic_target` immediately after loading the checkpoint.
- The critic optimizer is reinitialized after the reset.
- `--keep-pretrained-critic` is available on pretrained online runners for old-style comparison runs.
- Cold-start online runners are unchanged; their critic is already fresh.
- Run configs and summaries now record `pretrained_critic_reset`, `critic_reset_scope`, and `actor_loaded_from_checkpoint`.
- Safety-gate step tables now export `behavior_exploration_sigma` so BC/handoff/full-RL exploration schedules can be audited directly.

## Validation
- Static validation should compile `TD3Agent/agent.py`, `utils/online_disturbance_runner.py`, and the six online root runners.
- Smoke validation should confirm pretrained BC logs teacher Gaussian behavior with `behavior_exploration_sigma` near `0.0001`.
