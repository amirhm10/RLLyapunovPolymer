# GART Section 16 Online TD3 Runner Update

## Summary

Implemented the Section 16 GART-aware online TD3 path for the cold-start safety-gate runner and made the four GART online TD3 runners editable from their root files.

## Technical Changes

- Added `rl_observation_mode="standard" | "gart"` to `Simulation/run_rl_lyapunov.py`.
- Added `projection_backend="gart_section16_projection"` for the cold-start active-gate case.
- Reused the existing GART observation helper:

  $$
  o_k^{GART} =
  [\hat z_k, d^c_k, y_{sp,k}, u_{k-1}, r_k, y_{s,k}, u_{s,k}, m_{probe,k}]
  $$

- Reused the existing CVXPY Lyapunov safety-filter machinery for the Section 16 projection:

  $$
  u_k^{safe}
  =
  \arg\min_{u \in U}
  \|u-u_k^{RL}\|_{W_c}^2
  \quad
  \text{s.t.}
  \quad
  V(x_{k+1}(u)-x_{s,k})
  \le
  \rho V(\hat x_k-x_{s,k}) + \epsilon
  $$

- Kept TD3 replay training on the executed action while logging the actor proposal, projected action, fallback action, projection status, and Lyapunov margins.
- Added certificate-aware exploration scaling for the Section 16 backend.
- Added visible top-level parameters to:
  - `OnlineTD3_ColdStart_SafetyGate.py`
  - `OnlineTD3_ColdStart_NoSafetyGate.py`
  - `OnlineTD3_OFMPCPretrained_SafetyGate.py`
  - `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`

## Defaults

- `OnlineTD3_ColdStart_SafetyGate.py` now defaults to:
  - `RL_OBSERVATION_MODE = "gart"`
  - `PROJECTION_BACKEND = "gart_section16_projection"`
- `OnlineTD3_ColdStart_NoSafetyGate.py` keeps `RL_OBSERVATION_MODE = "standard"` by default for continuity.
- OF-MPC-pretrained runners keep `RL_OBSERVATION_MODE = "standard"` so existing checkpoints remain dimension-compatible.

## Validation

- AST compile check passed for the touched Python files.
- Root runner import smoke check passed.
- Observation-dimension smoke check passed:
  - standard TD3 observation: 10 features
  - GART Section 16 observation: 19 features
- CVXPY projection smoke could not be completed in the default `python` environment because `cvxpy` is not installed. The runtime code keeps the intended fallback behavior: if projection is unavailable or unverified, the active cold-start gate calls GART-LMPC.

## Notes

- `python -m py_compile` could not be used directly because this OneDrive workspace denied bytecode writes to `__pycache__`.
- A temporary `.validation-pyc/` directory was created during validation and Windows/OneDrive denied deletion because it was materialized as a reparse point. It is untracked and not part of this change.
