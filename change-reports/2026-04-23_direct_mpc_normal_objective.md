# Direct MPC Normal Objective

## Summary

Changed the direct frozen-output-disturbance Lyapunov MPC objective so it matches the normal MPC objective requested for this study:

- output tracking to the scheduled setpoint
- input move suppression through `delta_u`

The direct objective no longer penalizes:

- `u - u_s`
- terminal `x_N - x_s`

The steady target `(x_s, u_s)` is still computed and retained for Lyapunov contraction and terminal admissibility checks.

## Main Changes

- Added explicit direct-solver objective switches:
  - `objective_steady_input_cost=False`
  - `objective_terminal_cost=False`
- Updated hard-mode direct solves to zero the inherited steady-input and terminal objective weights while solving.
- Updated soft-mode direct solves to omit the steady-input and terminal objective terms directly.
- Exposed the objective switches in `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`.
- Set the notebook's visible `terminal_cost_scale = 0.0`.
- Updated the Markdown and LaTeX reports so the MPC objective is documented as `y - y_sp` plus `delta_u`.

## Clarification

With the notebook default `use_target_output_for_tracking=False`, the solver variable named `y_target` is the scheduled setpoint `y_sp`, not the steady target output `y_s`.

## Validation

- `python -m py_compile Lyapunov\direct_lyapunov_mpc.py`
- Notebook JSON parse and objective-setting check

## Not Run

- Full notebook execution was not run in this sandbox because the default Python environment does not have the scientific stack installed.
