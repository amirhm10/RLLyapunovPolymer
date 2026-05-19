# 2026-05-11 RL Direct Tracking Use ys

## What changed

Updated both direct safety-gate RL notebooks so the direct Lyapunov MPC tracking stage uses the admissible target output `y_s` instead of the raw setpoint `y_sp`.

Files changed:

- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

## Why

The RL safety-gate direct-tracking path was still passing:

- `direct_tracking_use_target_output=False`

which means the direct tracking MPC stage was solving against raw `y_sp` even when the target generator had already produced an admissible `y_s`.

Switching this to `True` aligns the direct tracking stage with the admissible target used by the direct Lyapunov formulation.

## Numerical interpretation

Small tolerance effects can contribute to whether contraction is marked satisfied near the boundary:

- `optimal`: tolerance `1e-7`
- `optimal_inaccurate`: tolerance `1e-5`

So there can be some numerical chatter when the contraction margin is very close to zero.

But that is unlikely to be the main cause of the previously observed non-settling when the controller is still tracking raw `y_sp` while the admissible target differs. The tracking-target mismatch is a stronger structural explanation than tolerance alone.

## Validation

- Notebook JSON remained valid after the edits.
- Both notebooks now contain `direct_tracking_use_target_output=True`.
