# Allow GART-LMPC On Held Targets With Current Disturbance

## Summary
- Fixed the cold-start GART-LMPC deadlock where `hold_previous` targets forced the controller to hold the previous input.
- Held-target fallback now keeps the previous steady `x_s,u_s` but recomputes `y_s = C x_s + C_d d_cert` using the current certified disturbance.
- If that held steady package is accepted, it remains usable for GART-LMPC so the LMPC solver attempts a control move instead of immediately returning hold-previous.

## Why
- The earlier hold-command fix recertified `alpha = 0` governed-reference candidates, but the separate stale-target fallback path still marked `usable_for_lmpc=False`.
- Online TD3 cold-start safety could then get stuck with repeated `gart_target_not_usable_hold_prev` rows.

## Validation
- Compile:
  ```powershell
  python -m py_compile Lyapunov/gart_target.py Simulation/run_rl_lyapunov.py tests/test_gart_target.py
  ```
- Targeted regression:
  ```powershell
  pytest tests/test_gart_target.py
  ```
