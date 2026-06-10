# Tune TD3 Pretraining Critic Defaults

## Summary

Updated both OF-MPC and LMPC TD3 pretraining helpers to use a faster critic warm-up learning rate and less aggressive TD3 target-policy smoothing before the next larger pretraining runs.

## Changed

- `utils/of_mpc_td3_workflow.py`
  - `critic_lr`: `1e-4` to `3e-4`
  - `target_policy_smoothing_noise_std`: explicit `0.05`
  - `noise_clip`: explicit `0.1`
- `utils/lmpc_td3_workflow.py`
  - `critic_lr`: `1e-4` to `3e-4`
  - `target_policy_smoothing_noise_std`: explicit `0.05`
  - `noise_clip`: explicit `0.1`
- Updated the OF-MPC and LMPC process reports with the shared forward-looking defaults.

## Rationale

The June 9 pilot showed that actor behavioral cloning was effective, while critic warm-up behavior was more sensitive. The new critic learning rate is still moderate but should adapt faster than `1e-4`. The target-smoothing settings `0.05/0.1` are less aggressive than the generic TD3 defaults `0.2/0.5`, which is more appropriate for bounded MPC action labels in this scaled process-control setting.

## Discount-Factor Note

The discount factor remains unchanged in this update. The offline pretraining helpers currently use `gamma = 0.995`; the active online Direct Lyapunov safety-gate RL runners use `GAMMA = 0.99`. A later change can align offline pretraining to online `0.99` if the next comparison prioritizes online reward/value consistency over the older OF-MPC pretraining convention.

## Validation

- `python -m py_compile utils/of_mpc_td3_workflow.py utils/lmpc_td3_workflow.py`
