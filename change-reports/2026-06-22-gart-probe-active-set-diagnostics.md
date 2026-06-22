# GART Probe Active-Set Speedup And No-Gate Diagnostics

## Context

The online TD3 GART safety-gate runner was interrupted inside CVXPY while building the per-step GART `contraction_probe` problem. The traceback showed a `KeyboardInterrupt` during CVXPY canonicalization, not a controller infeasibility or solver exception.

The no-safety-gate runners also need to keep reporting the diagnostic "would have activated" count because that is the main comparison against the active safety-gate runs.

## Changes

- Replaced the GART `contraction_probe` CVXPY solve with a small NumPy active-set solver for the same box-constrained quadratic probe.
- Kept the probe objective and margin definition unchanged:

  $$\min_{u_{\min} \le u \le u_{\max}} (A\hat{x} + Bu - x_s)^T P_x (A\hat{x} + Bu - x_s).$$

- Restored cold-start no-gate diagnostics as the default by setting `PROJECTION_BACKEND` to `mpc_only_diagnostic` unless `FAST_NO_DIAGNOSTIC=True`.
- Left the active safety-gate runner on `direct_accept_or_fallback`.

## Expected Impact

- Safety-gate and no-gate GART runs avoid repeated CVXPY canonicalization for the contraction probe.
- No-gate runs continue to report `diagnostic_unsafe_count`, while active safety-gate runs report actual interventions.
- The fast no-diagnostic cold no-gate path remains available for quick learning checks, but it is no longer the default comparison path.

## Validation

- `python -m py_compile Lyapunov/gart_target.py OnlineTD3_ColdStart_NoSafetyGate.py OnlineTD3_ColdStart_SafetyGate.py utils/online_disturbance_runner.py Simulation/run_rl_lyapunov.py`
- Compared the new active-set contraction probe against CVXPY on 25 random box-QP cases. Maximum objective gap was `4.097e-08`.
- Checked online TD3 backend normalization:
  - cold safety gate: `direct_accept_or_fallback`
  - cold no gate: `mpc_only_diagnostic`
  - OF-MPC-pretrained no gate: `mpc_only_diagnostic`
