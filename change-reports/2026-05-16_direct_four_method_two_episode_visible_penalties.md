# Direct Four-Method Two-Episode Visible Penalties

## What changed
- Updated `DirectLyapunovMPC_FourMethodDisturbance.ipynb` so the run length is controlled by a visible notebook variable:
  - `n_episodes = 2`
  - `n_tests = n_episodes`
- Exposed the direct target penalty weights directly in the notebook:
  - `u_prev_penalty_weight = 0.25`
  - `xs_prev_penalty_weight = 0.25`
- Routed those visible variables into `direct_four_method_case_specs(...)` instead of leaving the weights hard-coded in the function call.
- Added the visible penalty weights into the notebook `active_config` display so the run settings are recorded in the executed output.

## Why
- The notebook was previously using `DIRECT_DISTURBANCE_N_TESTS` and fixed penalty weights inside the case-spec call, which made quick experiment changes less transparent.
- Exposing the episode count and penalty weights in the notebook makes it easier to rerun short comparisons and confirm exactly which direct-reference penalties were active.

## Validation
- Executed `DirectLyapunovMPC_FourMethodDisturbance.ipynb` in place with the `rl-env` kernel.
- The executed notebook saved outputs successfully.
- Result root from this run:
  - `results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260516_202336`

## Observed outcome
- `bounded_hard`
  - `reward_mean = -11.5972`
  - `solver_success_rate = 0.9431`
  - `output_rmse_mean = 0.6541`
- `bounded_hard_u_prev_0p25_xs_prev_0p25`
  - `reward_mean = -2.9946`
  - `solver_success_rate = 1.0000`
  - `output_rmse_mean = 0.3147`

In this two-episode run, the direct method with the visible `u_prev` and `x_s` penalties outperformed the plain bounded-hard case on reward, solver success, and tracking RMSE.
