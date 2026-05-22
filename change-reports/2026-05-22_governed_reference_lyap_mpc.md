# Governed-Reference LyapMPC Proposal 01

Date: 2026-05-22

## Summary

Implemented Proposal 01 as a separate direct Lyapunov MPC test path. The existing `DirectLyapunovMPC.py` runner and the existing bounded/unbounded target selector behavior are not changed.

## Files Added

- `DirectLyapunovMPC_GovernedReference.py`
- `Lyapunov/governed_reference_target.py`
- `report/governed_reference_lyap_mpc_methodology_2026-05-22.md`

## Files Updated

- `Lyapunov/direct_lyapunov_mpc.py`

## Technical Changes

- Added a new `target_mode="governed_reference"` branch in the direct LMPC preparation path.
- Added a two-stage governed-reference target layer:
  - first solve an admissible command `r_cmd` near the raw setpoint,
  - then solve the steady Lyapunov target around `r_cmd`.
- Added optional one-step Lyapunov feasibility probe metadata.
- Added saved diagnostics for:
  - `r_cmd`,
  - `r_cmd_minus_y_sp`,
  - `y_s_minus_r_cmd`,
  - `governor_active`,
  - `governor_probe_margin`,
  - `input_headroom_min`,
  - `command_move_inf`.
- Added bundle arrays, summary metrics, step CSV fields, and a governed-reference diagnostic plot.

## Validation

Completed:

```powershell
python -m py_compile DirectLyapunovMPC_GovernedReference.py Lyapunov/governed_reference_target.py Lyapunov/direct_lyapunov_mpc.py
```

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile DirectLyapunovMPC_GovernedReference.py Lyapunov/governed_reference_target.py Lyapunov/direct_lyapunov_mpc.py
```

The implementation passed a synthetic target-layer check in `rl-env`:

- feasible raw setpoint gives `r_cmd` near `y_sp`;
- infeasible raw setpoint modifies `r_cmd`;
- old `target_mode="bounded"` remains on the old solver path.

`git diff --check` completed with only existing CRLF-normalization warnings.

## Notes

This change only creates the method and diagnostics. It does not claim improved performance before the governed-reference run is executed.
