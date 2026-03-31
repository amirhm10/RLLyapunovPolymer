## Summary

Rebuilt the first-step-contraction experiment around the MPC-upstream safety architecture instead of the old standard-Lyapunov solver stack.

## Why

The previous implementation was architecturally wrong for the intended experiment. It used the separate `Lyapunov/` standard tracking solver path, while the intended controller should have:

- the same refined selector and effective-target logic as the MPC safety notebook,
- the same upstream offset-free MPC objective,
- no QCQP projection,
- one hard first-step Lyapunov contraction inequality added directly to the upstream MPC solve.

## What Changed

- Added a constrained upstream-MPC helper in [Lyapunov/upstream_controllers.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/upstream_controllers.py):
  - `solve_offset_free_mpc_candidate_with_first_step_contraction(...)`
  - same baseline MPC objective
  - same prediction model
  - one nonlinear inequality on the first predicted physical-state step
- Added a dedicated runner in [run_mpc_first_step_contraction.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Simulation/run_mpc_first_step_contraction.py):
  - same selector config surface as the safety notebook
  - same current-target / last-valid-target reuse logic
  - same tracking-target policy logic
  - ordinary fallback MPC when the constrained solve cannot be used
  - no QCQP / no projection path
- Repurposed [StandardLyapMPCFirstStepContraction.ipynb](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/StandardLyapMPCFirstStepContraction.ipynb):
  - mirrors the safety notebook structure
  - uses `MpcSolver`
  - uses the new constrained-MPC runner
  - reuses the safety debug/export pipeline
- Extended [safety_debug.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/safety_debug.py):
  - stores `V_next_first`
  - stores `contraction_margin`
  - stores `first_step_contraction_satisfied`
  - generates `first_step_contraction_diagnostics.png`
- Updated [standard_lyap_first_step_contraction.md](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/report/standard_lyap_first_step_contraction.md) so the method note now matches the corrected architecture.

## Behavior After Change

- The notebook now uses the same refined selector flow and config style as the safety MPC notebook.
- The MPC objective is unchanged from baseline MPC.
- There is no projection / QCQP correction stage.
- The only Lyapunov enforcement is the first-step contraction constraint inside the upstream MPC optimization.
- If that constrained solve fails, the controller falls back to ordinary offset-free MPC and logs whether that fallback happened to satisfy the Lyapunov test.

## Validation

- `python -m py_compile Lyapunov\\upstream_controllers.py Simulation\\run_mpc_first_step_contraction.py Lyapunov\\safety_debug.py`
- Notebook JSON validation for `StandardLyapMPCFirstStepContraction.ipynb`
- Static grep confirmed the repurposed notebook no longer imports the old standard-Lyapunov first-step solver path

## Rollback

Revert this commit to restore the previous notebook/runner path that used the separate standard-Lyapunov solver stack.
