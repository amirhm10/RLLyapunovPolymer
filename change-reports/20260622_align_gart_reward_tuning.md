# Align GART-LMPC Reward Tuning With Online TD3

## Objective

Make the GART-LMPC two-phase baseline use the same shaped reward settings as the four online TD3 runners before starting the comparison runs.

## Changes

- Updated `experiments/run_gart_target_selector_study.py` reward settings:
  - `band_floor_phys = [0.004, 0.06]`
  - `beta = 5.0`
- Recorded GART reward components in each step:
  - `reward_tracking_cost`
  - `reward_move_cost`
  - `reward_bonus`
  - `reward_w_in`
- Added those reward component fields to the direct Lyapunov/GART compact step-table and NPZ exports.

## Validation

- `python -X pycache_prefix=... -m py_compile experiments/run_gart_target_selector_study.py Lyapunov/direct_lyapunov_mpc.py utils/online_disturbance_runner.py Lyapunov/safety_debug.py`
- Runtime config check in the `rl` conda environment confirmed:
  - TD3 reward config: `beta = 5.0`, `band_floor_phys = [0.004, 0.06]`
  - GART reward config: `beta = 5.0`, `band_floor_phys = [0.004, 0.06]`

The five two-phase runners now use the same reporting reward definition.
