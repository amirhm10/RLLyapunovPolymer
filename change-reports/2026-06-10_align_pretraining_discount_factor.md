# Align TD3 Pretraining Discount Factor

## Summary

Changed the OF-MPC and LMPC TD3 pretraining helpers from `gamma = 0.995` to `gamma = 0.99`.

## Changed

- `utils/of_mpc_td3_workflow.py`
  - `gamma`: `0.995` to `0.99`
- `utils/lmpc_td3_workflow.py`
  - `gamma`: `0.995` to `0.99`
- Updated the OF-MPC and LMPC process reports to document the new shared discount factor.

## Rationale

The active online Direct Lyapunov safety-gate RL runners already use `GAMMA = 0.99`. Aligning offline pretraining to the online value horizon makes the pretrained critic handoff more consistent with online training and reduces sensitivity to long-horizon bootstrapping from model/observer-state transitions.

The actor behavioral-cloning stage is unaffected by the discount factor. The change mainly affects frozen-actor critic TD warm-up and saved checkpoint metadata.

## Validation

- `python -m py_compile utils/of_mpc_td3_workflow.py utils/lmpc_td3_workflow.py`
