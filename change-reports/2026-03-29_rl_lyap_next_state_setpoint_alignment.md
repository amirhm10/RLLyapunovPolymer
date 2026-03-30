# RL Lyapunov Next-State Setpoint Alignment

## Why
- The RL transition stored in `Simulation/run_rl_lyapunov.py` used `y_sp_k` for the current state and reward, but used `y_sp_kp1` for the replay `next_state`.
- At setpoint-change boundaries, that mixes two different tasks in one TD3 transition:
  - action and reward belong to the old setpoint,
  - next-state label belongs to the new setpoint.
- The plain RL runner `Simulation/run_rl.py` already uses the current setpoint for both state and next-state in the replay tuple.

## What Changed
- Updated both TD3 replay `next_state` constructions in `Simulation/run_rl_lyapunov.py` to use `y_sp_k` instead of `y_sp_kp1`.
- Left `y_sp_kp1` available for diagnostics such as `e_next`, so this change is limited to replay semantics.

## Effect
- The TD3 replay tuple in the Lyapunov RL runner is now logically consistent:
  - `s_t = [x_t, y_sp_t, u_{t-1}]`
  - `a_t`
  - `r_t`
  - `s_{t+1} = [x_{t+1}, y_sp_t, u_t]`
- This matches the interpretation that the setpoint is constant over the control step and only changes at block boundaries.

## Files
- `Simulation/run_rl_lyapunov.py`

## Validation
- `python -m py_compile Simulation/run_rl_lyapunov.py`
