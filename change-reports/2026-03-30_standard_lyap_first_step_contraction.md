# Standard Lyapunov MPC With First-Step Contraction

## Why
- We wanted a standard Lyapunov MPC variant that reuses the current refined target selector but does not use the safety-filter architecture.
- The requested design keeps the existing standard tracking MPC objective unchanged and adds one hard Lyapunov contraction constraint only on the first predicted step.

## What Changed
- Added a new sibling solver in `Lyapunov/lyapunov_core.py`:
  - `FirstStepContractionTrackingLyapunovMpcSolver`
- Added a new sibling rollout in `Lyapunov/run_lyap_mpc.py`:
  - `run_standard_tracking_lyapunov_mpc_first_step_contraction(...)`
- Added a new import-driven notebook:
  - `StandardLyapMPCFirstStepContraction.ipynb`
- Added a new implementation-facing math note:
  - `report/standard_lyap_first_step_contraction.md`

## Controller Semantics
- The refined target selector is still computed every step.
- The MPC objective is unchanged relative to the current standard solver.
- The new hard constraint is:
  - only on the first predicted physical-state step
  - centered on the selected steady physical state `x_s`
  - evaluated with the existing `P_x`
  - bounded by `rho * V_k + eps_lyap`

## Behavior Preservation
- The existing standard solver class and standard rollout were left intact.
- No safety-filter code was changed.
- Existing notebook workflows were not modified.

## Validation
- `python -m py_compile Lyapunov\\lyapunov_core.py Lyapunov\\run_lyap_mpc.py`
- notebook JSON validation for `StandardLyapMPCFirstStepContraction.ipynb`
