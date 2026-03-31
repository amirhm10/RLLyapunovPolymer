## Summary

Aligned the new first-step-contraction standard Lyapunov MPC path with the same refined target-selector config surface used in the safety-filter MPC notebook.

## Why

The new notebook was exposing the older low-level selector knobs (`Qs_tgt_diag`, `Ru_tgt_diag`, `Qx_tgt_diag`, etc.), while the maintained safety notebook uses the higher-level refined selector config (`alpha_u_ref`, `alpha_du_sel`, `alpha_dx_sel`, `alpha_x_ref`, `x_weight_base`, `use_output_bounds_in_selector`). That mismatch made the new notebook look inconsistent with the canonical selector workflow.

## What Changed

- Updated [Lyapunov/run_lyap_mpc.py](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/Lyapunov/run_lyap_mpc.py) so `run_standard_tracking_lyapunov_mpc_first_step_contraction(...)` now accepts:
  - `target_selector_config`
  - `selector_warm_start`
  - `selector_H`
- The new runner now uses `prepare_filter_target(...)` when `target_selector_config` is provided, so it goes through the same refined selector config path as the safety notebook.
- Preserved backward compatibility for the new runner by keeping the older low-level selector arguments as a fallback when `target_selector_config` is not provided.
- Updated [StandardLyapMPCFirstStepContraction.ipynb](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/StandardLyapMPCFirstStepContraction.ipynb) so it now uses:
  - `run_config["target_selector_config"]`
  - `run_config["selector_warm_start"]`
  - `run_config["selector_H"]`
- Replaced the notebook selector config block with the same parameter shape used in [LyapunovSafetyFilterMPC.ipynb](C:/Users/HAMEDI/Desktop/Lyapunov_polymer/LyapunovSafetyFilterMPC.ipynb):
  - `alpha_u_ref`
  - `alpha_du_sel`
  - `alpha_dx_sel`
  - `alpha_x_ref`
  - `x_weight_base`
  - `use_output_bounds_in_selector`

## Behavior After Change

- The first-step-contraction notebook now exposes the refined selector the same way as the safety notebook.
- The new controller path still keeps the MPC objective unchanged and only adds the first-step Lyapunov contraction constraint.
- Existing older standard paths were not changed.

## Validation

- `python -m py_compile Lyapunov\\run_lyap_mpc.py`
- Notebook JSON parse for `StandardLyapMPCFirstStepContraction.ipynb`

## Rollback

If needed, revert this commit to restore the notebook to the older low-level selector interface for the first-step-contraction path.
