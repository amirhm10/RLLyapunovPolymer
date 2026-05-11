# Direct Four-Method Disturbance Shaped Reward

Date: 2026-05-11

## Summary

Updated [DirectLyapunovMPC_FourMethodDisturbance.ipynb](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/DirectLyapunovMPC_FourMethodDisturbance.ipynb>) to use the shaped reward function instead of the quadratic reward.

## Changes

- Replaced `make_reward_fn_mpc_quadratic` with `make_reward_fn_relative_QR`.
- Added the same shaped-reward configuration used in the other direct notebooks:
  - `k_rel = [0.003, 0.0003]`
  - `band_floor_phys = [0.006, 0.07]`
  - `tau_frac = 0.7`
  - `gamma_out = 0.5`
  - `gamma_in = 0.5`
  - `beta = 7.0`
  - `gate = "geom"`
  - `lam_in = 1.0`
  - `bonus_kind = "exp"`
  - `bonus_k = 12.0`
  - `bonus_p = 0.6`
  - `bonus_c = 20.0`
- Kept the notebook's local `Qy_diag` and `Rdu_diag` weights unchanged.

## Validation

- Verified the notebook remains valid JSON.
- Compiled each code cell in memory after the edit.
