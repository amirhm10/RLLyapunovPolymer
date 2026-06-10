# Online Exploration End Std And Disturbance Profile

Date: 2026-06-10

## Summary

Updated the disturbance-only online TD3 runner exploration schedule so the full-RL Gaussian exploration standard deviation decays to `0.005`. Added a reproducible disturbance-profile plotting script and embedded the generated figure in the online runner algorithm audit report.

## Changes

- Updated `utils/online_disturbance_runner.py`:
  - `STD_END = 0.005`
  - `full_rl_exploration_std_end = 0.005`
- Added `analysis/plot_online_disturbance_profile.py`.
- Generated `report/figures/2026-06-10_online_disturbance_runner/disturbance_profile.png`.
- Updated `report/online_disturbance_runner_algorithm_audit_2026-06-10.md` with:
  - BC actor/critic behavior
  - TD3 discount factor `0.99`
  - TD3 policy delay `2`
  - Lyapunov `rho=0.99`, `eps=5e-3`, and tolerance `1e-10`
  - disturbance profile figure and schedule description

## Validation

- `python -m py_compile utils/online_disturbance_runner.py analysis/plot_online_disturbance_profile.py`
- `python analysis/plot_online_disturbance_profile.py`

