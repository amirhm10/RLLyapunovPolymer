# Tune Online Reward Bonus And Band

## Objective

Increase the near-setpoint reward incentive for the online TD3 two-phase studies and expose the reward components needed to diagnose whether the shaped reward is behaving as intended.

## Changes

- Updated the online disturbance reward configuration:
  - `band_floor_phys`: `[0.003, 0.035]` to `[0.004, 0.06]`
  - `beta`: `1.0` to `5.0`
- Added reward component exports to safety debug step records and arrays:
  - `reward_tracking_cost`
  - `reward_move_cost`
  - `reward_bonus`
  - `reward_w_in`

## Interpretation

The shaped reward remains:

$$
r = -(\ell_y + \ell_u) + b
$$

where `reward_tracking_cost` is $\ell_y$, `reward_move_cost` is $\ell_u$, and `reward_bonus` is $b$. The wider physical band makes the near-setpoint gate less restrictive, and the larger `beta` increases the positive bonus when the output is inside that band.

## Validation

- `python -X pycache_prefix=... -m py_compile utils/online_disturbance_runner.py Lyapunov/safety_debug.py`
- Reward configuration check in the `rl` conda environment confirmed:
  - `band_floor_phys = [0.004, 0.06]`
  - `beta = 5.0`
  - perfect-tracking maximum step reward is approximately `0.18` to `0.20` for the current phase setpoints.
