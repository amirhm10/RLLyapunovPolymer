# Promote Governed-Reference Defaults

Date: 2026-05-23

## Summary

Governed-reference target selection was promoted from the temporary test runner into the default target selector for the active direct Lyapunov MPC and safety-gate RL runners.

## Changes

- Added shared governed-reference target config helpers in `utils/direct_lyapunov_study.py`.
- Updated `DirectLyapunovMPC.py` to use governed-reference for the direct Lyapunov case and `mpc_only` diagnostics.
- Updated `DirectLyapunovSafetyGateRL_ColdStart.py` and `DirectLyapunovSafetyGateRL_Pretrained.py` to use governed-reference safety-gate targets and governed-reference `mpc_only` diagnostics.
- Preserved raw setpoint tracking with `use_target_output_for_tracking = False` and `direct_tracking_use_target_output = False`.
- Archived the temporary root test runner as `archive/DirectLyapunovMPC_GovernedReference.py`.
- Extended the governed-reference methodology report with the adopted default and configuration rationale.

## Adopted Target Defaults

```python
target_mode = "governed_reference"
lambda_cmd_move = 1.0
Qr_diag = Qy_diag
W_r_diag = Qy_diag
u_ref_weight = 0.1
x_ref_weight = 0.1
input_headroom_frac = 0.03
one_step_probe = True
```

## Validation

- Passed `python -m py_compile` on the three active runners, `Lyapunov/direct_lyapunov_mpc.py`, `Lyapunov/governed_reference_target.py`, `Simulation/run_rl_lyapunov.py`, and `utils/direct_lyapunov_study.py`.
- Passed lightweight config checks confirming governed-reference is the selected target helper in all three active runners, bounded case specs are no longer selected by default, raw setpoint tracking remains active, and `u_ref_weight = x_ref_weight = 0.1`.
- Confirmed the temporary governed-reference runner was removed from the root and archived under `archive/`.

Full training and long direct MPC runs were intentionally not part of implementation validation.
