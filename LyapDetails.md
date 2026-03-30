# Standard Lyapunov MPC Details

## Scope
This document describes the exact standard Lyapunov MPC procedure currently implemented in:

- `Lyapunov/target_selector.py`
- `Lyapunov/lyapunov_core.py`
- `Lyapunov/run_lyap_mpc.py`
- `utils/lyapunov_utils.py`

This is the canonical Lyapunov path in the repository now.

## High-Level Structure
The controller is split into three stages:

1. compute a feasible steady-state target
2. solve a finite-horizon tracking MPC around that target
3. apply the first input, step the plant, and update the observer

The Lyapunov component is standard terminal-set / terminal-cost MPC. It is not a hard pathwise contraction filter.

## State and Coordinate Conventions
- `xhatdhat` is the augmented observer state
- `x` is the physical state part
- `d` is the disturbance-state part
- `u` is the input in scaled deviation coordinates inside the controller
- `y` is the output in scaled deviation coordinates inside the controller
- physical plant inputs and outputs are converted with min-max scaling helpers
- `ss_inputs` and `y_ss` are the steady-state anchors used to move between physical and deviation coordinates

## File Responsibilities

### `Lyapunov/target_selector.py`
Contains `compute_ss_target_refined_rawlings(...)`.

This solves the steady-state target problem:

- choose `x_s` and `u_s`
- keep the disturbance target fixed as the current estimate `d_hat`
- make the target output `y_s` as close as possible to the requested setpoint `y_sp`
- keep `u_s` inside tightened or untightened bounds
- optionally penalize target motion relative to the previous target

### `Lyapunov/lyapunov_core.py`
Contains:

- `design_standard_tracking_terminal_ingredients(...)`
- `compute_terminal_alpha_input_only(...)`
- `StandardTrackingLyapunovMpcRawlingsTargetSolver`

This file handles:

- terminal Riccati matrix design
- local stabilizing feedback gain
- terminal ellipsoid sizing from input headroom
- finite-horizon tracking MPC optimization

### `Lyapunov/run_lyap_mpc.py`
Contains `run_standard_tracking_lyapunov_mpc_rawlings_target(...)`.

This is the closed-loop rollout:

- builds setpoint and disturbance schedules
- calls the target selector each step
- calls the tracking MPC solver each step
- updates the plant and observer
- stores diagnostics and rewards

### `utils/lyapunov_utils.py`
Contains shared helper functions:

- diagonal PSD matrix construction
- vector defaults
- setpoint extraction by time step
- input-sequence reshape
- `du` computation
- warm-start shift for the next MPC call

## Exact Step-by-Step Procedure

## 1. Build the augmented model
The implementation assumes an augmented offset-free model:

- physical state dynamics
- disturbance states that are constant
- output equation that includes both physical state and disturbance contribution

The model is partitioned internally as:

- `A = A_aug[:n_x, :n_x]`
- `Bd = A_aug[:n_x, n_x:]`
- `B = B_aug[:n_x, :]`
- `C = C_aug[:, :n_x]`
- `Cd = C_aug[:, n_x:]`

## 2. Design the terminal ingredients
Before running the closed loop, call:

`design_standard_tracking_terminal_ingredients(...)`

This does the following:

1. extracts the physical subsystem from the augmented model
2. builds the physical-state penalty
   - `Qx = C.T @ Qy @ C + qx_eps * I`
3. builds the input penalty `Su`
   - from user-provided `Su_diag`, or
   - from input range if `Su_diag` is omitted
4. solves the discrete Riccati equation
5. computes the local stabilizing feedback
   - `K_x = -(Su + B.T P_x B)^-1 B.T P_x A`

The outputs are:

- `P_x`: terminal Lyapunov matrix
- `K_x`: local terminal controller

## 3. Construct the tracking MPC solver
Instantiate:

`StandardTrackingLyapunovMpcRawlingsTargetSolver(...)`

Important stored quantities:

- augmented model matrices
- output tracking weights `Qy`
- target-input weights `Su`
- move suppression weights `Rdu`
- prediction horizon `NP`
- control horizon `NC`
- terminal cost matrix `P_x`
- terminal feedback `K_x`
- terminal set toggle and scale

## 4. Generate the setpoint schedule
At the start of the rollout, `run_standard_tracking_lyapunov_mpc_rawlings_target(...)` calls:

`generate_setpoints_training_rl_gradually(...)`

This creates:

- the full setpoint trajectory `y_sp`
- the total number of simulation steps `nFE`
- sub-episode boundaries for logging
- disturbance schedules `qi`, `qs`, `ha`

## 5. Initialize scaled steady-state anchors
The rollout converts steady-state plant values into scaled coordinates:

- `ss_scaled_inputs`
- `y_ss_scaled`

These are used throughout the loop to:

- convert plant input to controller deviation coordinates
- convert plant output to controller deviation coordinates
- convert optimized inputs back to physical units

## 6. Initialize closed-loop storage
The rollout allocates:

- `y_mpc`
- `u_mpc`
- `yhat`
- `xhatdhat`
- `rewards`
- `avg_rewards`
- `delta_y_storage`
- `delta_u_storage`
- `lmpc_info_storage`
- `target_info_storage`

It also initializes:

- `IC_opt` for the MPC warm start
- `x_s_prev` and `u_s_prev` for target smoothing

## 7. Start one closed-loop time step
For each `step_idx`:

1. read the current augmented estimate `x0_aug`
2. scale the current physical plant input
3. convert that scaled input into deviation form
   - `u_prev_dev = scaled_current_input - ss_scaled_inputs`
4. extract the current setpoint block value `y_sp_k`
5. scale the previous measured plant output into deviation form
6. compute the current predicted output
   - `yhat_now = C_aug @ x0_aug`
7. compute the observer innovation
   - `innovation = y_prev_scaled - yhat_now`

## 8. Solve the refined target-selector problem
The rollout calls:

`compute_ss_target_refined_rawlings(...)`

This optimization solves for `x_s` and `u_s` while holding `d_s = d_hat`.

### Target-selector objective
The objective is:

- output target error penalty
  - `||y_s - y_sp||_Ty^2`
- input-centering penalty
  - `||u_s - u_nom||_Ru^2`
- state regularization
  - `||x_s||_Qx^2`
- optional target motion penalties
  - `||x_s - x_s_prev||_Qdx^2`
  - `||u_s - u_s_prev||_Rdu^2`
- optional soft output-bound slack penalties

### Target-selector constraints
The constraints are:

1. steady-state dynamics
   - `(I - A)x_s - Bu_s - Bd d_hat = 0`
2. input bounds
   - `u_lo <= u_s <= u_hi`
3. optional output bounds
   - hard or soft depending on configuration

### Target-selector outputs
The solver returns:

- `x_s`
- `u_s`
- `d_s = d_hat`
- a debug dictionary containing target quality and feasibility information

If the target selector fails, the rollout falls back to holding the previous input in deviation coordinates.

## 9. Build the target-centered Lyapunov terminal set size
If target selection succeeds, the rollout computes:

`alpha_terminal = compute_terminal_alpha_input_only(...)`

This procedure:

1. uses `P_x` and `K_x`
2. checks how much input headroom exists around `u_s`
3. computes the largest ellipsoid radius such that
   - `u = u_s + K_x e_x`
   remains inside input bounds

The result is the terminal-set radius `alpha_terminal`.

## 10. Build the tracking target
The tracking target is the steady-state output implied by the selected target:

- `x_s_aug = [x_s, d_s]`
- `y_target = C_aug @ x_s_aug`

This means the MPC tracks the admissible steady-state output selected by the target-selector, not the raw setpoint directly.

## 11. Solve the standard tracking MPC
The rollout calls:

`LMPC_obj.solve_tracking_mpc_step(...)`

This uses SLSQP and solves over the control sequence `U`.

### Internal prediction
For each prediction step:

1. propagate the augmented model
2. use the final control move after `NC` as the held input for the rest of the prediction horizon
3. compute the predicted outputs

### Tracking cost
The optimizer minimizes:

1. output tracking error over the horizon
   - `sum ||y_pred - y_target||_Qy^2`
2. input deviation from target input
   - `sum ||u - u_s||_Su^2`
3. optional move penalty
   - `sum ||du||_Rdu^2`
4. terminal cost
   - `e_N^T P_x e_N`
   where `e_N = x_N - x_s`

### Terminal constraint
If enabled, the solver also enforces:

- `e_N^T P_x e_N <= alpha_terminal`

This is the Lyapunov terminal ellipsoid.

## 12. Apply the first optimized move
If the optimizer succeeds:

1. take the first optimized deviation input
2. clip it to the allowed deviation bounds
3. warm-start the next optimization with `shift_input_guess(...)`

If the optimizer fails:

1. use `u_s` as fallback
2. tile that fallback input across the control horizon for the next warm start

## 13. Convert back to plant units
The applied deviation input is shifted back by the scaled steady-state input:

- `u_mpc[k] = u_dev_apply + ss_scaled_inputs`

Then it is converted to physical units with `reverse_min_max(...)`.

## 14. Apply disturbances and step the plant
If `mode == "disturb"`:

1. apply scheduled disturbance values
   - `hA`
   - `Qs`
   - `Qi`
2. set the plant input
3. call `system.step()`

The next measured physical output is stored in `y_mpc[k + 1]`.

## 15. Update measured output and tracking error
After stepping the plant:

1. scale the new measured output to deviation coordinates
2. compute the tracking error
   - `delta_y = y_current_scaled - y_sp_k`
3. compute the applied input move
   - `delta_u = u_mpc[k] - scaled_current_input`

## 16. Update the observer
The observer update is:

1. open-loop propagation
   - `xhat_next_openloop = A_aug x0_aug + B_aug u_dev_apply`
2. correction
   - `observer_correction = L @ innovation`
3. corrected next estimate
   - `xhatdhat[:, k + 1] = xhat_next_openloop + observer_correction`

This keeps the estimator structure close to the previous code path.

## 17. Compute reward and diagnostics
The rollout then:

1. converts the current setpoint back to physical output units
2. calls the user-provided reward function
3. stores:
   - rewards
   - output and input error traces
   - target-selector debug info
   - tracking MPC debug info

The debug storage is split into:

- `target_info_storage`
- `lmpc_info_storage`

## 18. Log sub-episode summaries
At each sub-episode boundary:

1. average reward over the recent block is computed
2. the rollout prints a compact summary including:
   - average reward
   - chosen method
   - success flag
   - terminal alpha
   - terminal margin
   - target slack info
   - solver iterations

## 19. Return final rollout data
At the end, the rollout converts `u_mpc` fully back to physical units and returns:

1. `y_mpc`
2. `u_mpc`
3. `avg_rewards`
4. `rewards`
5. `xhatdhat`
6. `nFE`
7. `time_in_sub_episodes`
8. `y_sp`
9. `yhat`
10. `delta_y_storage`
11. `delta_u_storage`
12. `lmpc_info_storage`
13. `target_info_storage`

## Failure Logic

### Target-selector failure
If the target-selector does not return a valid target:

- the controller holds the previous input in deviation coordinates
- the warm start is reset using that held input

### Tracking solver failure
If the tracking MPC fails after a valid target exists:

- the controller falls back to `u_s`
- the warm start is reset using repeated `u_s`

## What Makes This Lyapunov MPC
This implementation is Lyapunov MPC because it uses:

- a terminal Lyapunov matrix `P_x`
- a local stabilizing feedback `K_x`
- a target-centered terminal ellipsoid sized from input headroom
- a terminal cost on the final state error

It does not use:

- a hard one-step contraction filter
- pathwise Lyapunov decrease constraints at each prediction step

## Recommended Reading Order
If you want to inspect the implementation in code order, read:

1. `utils/lyapunov_utils.py`
2. `Lyapunov/target_selector.py`
3. `Lyapunov/lyapunov_core.py`
4. `Lyapunov/run_lyap_mpc.py`

## Short Practical Recipe
To use the current standard Lyapunov MPC path:

1. build or load the augmented model
2. compute `P_x` and `K_x` with `design_standard_tracking_terminal_ingredients(...)`
3. instantiate `StandardTrackingLyapunovMpcRawlingsTargetSolver(...)`
4. call `run_standard_tracking_lyapunov_mpc_rawlings_target(...)`
5. inspect `lmpc_info_storage` and `target_info_storage` for diagnostics
