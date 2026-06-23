# Offset-Free MPC Phase-2 Online Cycle Probe

## Objective

Update the offset-free MPC disturbance runner so it can test the same Phase-2 continuation setpoint cycle used by the online two-phase runners. This helps check whether the current disturbance profile makes the first setpoint cycle infeasible even for plain offset-free MPC.

## Changes

- Updated `OffsetFreeMPC_DisturbanceRunner.py` to use the online two-setpoint schedule:
  - `(4.5, 324.0)` for 400 samples
  - `(3.4, 321.0)` for 400 samples
- Restored full-cycle reporting:
  - `REPORTING_WINDOW_STEPS = 800`
  - `rollout_set_points_len = 400`
- Set the probe to the same Phase-2 duration as the online runners:
  - `PHASE2_EPISODES = 50`
  - `PHASE2_STEPS = 40000`
- Renamed the case to `offset_free_mpc_phase2_online_setpoint_cycle`.

## Validation

- `python -X pycache_prefix=... -m py_compile OffsetFreeMPC_DisturbanceRunner.py`
- Profile construction check confirmed:
  - `phase2_episodes = 50`
  - `n_profile_steps = 40000`
  - `reporting_window_steps = 800`
  - `rollout_n_tests = 50`
  - `rollout_set_points_len = 400`
  - disturbance starts at the online Phase-2 initial values:
    - `Qi = 102.6`
    - `Qs = 481.95`
    - `hA = 966000.0`
  - disturbance ends at:
    - `Qi = 113.4`
    - `Qs = 436.05`
    - `hA = 924000.0`

This runner still probes Phase 2 only. It starts from the final Phase-1 disturbance value and then applies the Phase-2 disturbance ramp.
