# Align LMPC TD3 Pretraining Hyperparameters

## Summary

Aligned the LMPC TD3 pretraining helper with the OF-MPC TD3 pretraining helper so the next scale-up can compare expert-label source rather than mixed TD3 optimizer settings.

## Changed

- Updated `utils/lmpc_td3_workflow.py` so `make_lmpc_td3_agent(...)` now uses the same core TD3 defaults as `utils/of_mpc_td3_workflow.py`:
  - `gamma = 0.995`
  - `actor_lr = 1e-4`
  - `critic_lr = 1e-4`
  - `policy_delay = 4`
  - `target_policy_smoothing_noise_std = 0.2`
  - `noise_clip = 0.5`
- Updated the OF-MPC and LMPC process reports to document the aligned defaults and clarify that `policy_delay` is not active inside the offline `pretrain_from_buffer(...)` actor-BC and frozen-actor critic warm-up loops.

## Rationale

The June 9 pilot comparison used equal sample counts and equal `[256, 256, 256]` architectures, but the OF-MPC and LMPC checkpoints used different TD3 optimizer and target settings. Aligning LMPC to the OF-MPC settings makes future OF-MPC-versus-LMPC pretraining runs a cleaner comparison of the expert-label source.

## Validation

- `python -m py_compile utils/of_mpc_td3_workflow.py utils/lmpc_td3_workflow.py`
