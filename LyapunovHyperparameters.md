# Lyapunov Safety Filter Hyperparameters

This note is for the current canonical implementation in:

- `Simulation/run_mpc_lyapunov.py`
- `Simulation/run_rl_lyapunov.py`
- `Lyapunov/safety_filter.py`
- `Lyapunov/target_selector.py`
- `Lyapunov/lyapunov_core.py`

The most important rule in this repository is: all controller quantities below live in scaled deviation coordinates unless stated otherwise.

- Inputs are deviation coordinates around `steady_states["ss_inputs"]`.
- Outputs and setpoints are deviation coordinates around `steady_states["y_ss"]`.
- The plant itself usually runs in physical coordinates.
- `xhatdhat` is the augmented observer state `[x_hat; d_hat]`.

If one parameter looks "wrong by a lot", first check the coordinate system before retuning weights.

## 1. Exact-MPC Equivalence Switches

These are the first settings to lock down when comparing plain MPC against the Lyapunov-filtered path.

### `mpc_target_policy`

Where:

- `Simulation/run_mpc_lyapunov.py`
- `Simulation/run_rl_lyapunov.py`

Options:

- `"raw_setpoint"`: MPC always tracks the original setpoint `y_sp`.
- `"admissible_if_available"`: MPC tracks selector output `y_s` whenever the selector succeeds.
- `"admissible_on_fallback"`: MPC tracks `y_s` only when the selector is already in fallback mode.

Effect:

- `"raw_setpoint"` gives the closest behavior to your baseline `run_mpc(...)`.
- This policy now drives three places together:
  - upstream MPC candidate target
  - MPC fallback target
  - QCQP output-tracking target
- The admissible-target modes are safer conceptually, but they change the upstream control law and can change performance a lot.

Recommendation:

- Use `"raw_setpoint"` when debugging equivalence to plain MPC.
- When this is `"raw_setpoint"`, the QCQP output term also tracks raw `y_sp` rather than `y_s`.
- The Lyapunov constraint itself is still centered on the selector equilibrium `(x_s, u_s, d_s)`.

### `target_backup_policy`

Where:

- `Simulation/run_mpc_lyapunov.py`
- `Simulation/run_rl_lyapunov.py`

Options:

- `"last_valid"`: if the current selector solve fails, reuse the previous valid target package for candidate testing and QCQP centering.

Effect:

- This reduces unnecessary fallback-MPC calls when the selector fails transiently.
- It does not change the upstream MPC target directly unless the tracking-target policy is an admissible-target mode.

Recommendation:

- Keep `"last_valid"` as the default.

### `reuse_mpc_solution_as_ic`

Where:

- `Simulation/run_mpc_lyapunov.py`
- `Simulation/run_rl_lyapunov.py`

Meaning:

- `False`: every MPC solve starts from the same fixed `IC_opt`, exactly like the baseline `run_mpc(...)`.
- `True`: the optimizer warm-starts from the previous solution.

Effect:

- `False` gives the cleanest apples-to-apples comparison against your plain MPC notebook.
- `True` can improve optimizer speed, but with `scipy.optimize.minimize` it can also move the closed-loop trajectory because the optimizer is local.

Recommendation:

- Set `False` while debugging controller differences.

### `reset_system_on_entry`

Where:

- `Simulation/run_mpc_lyapunov.py`
- `Simulation/run_rl_lyapunov.py`

Meaning:

- `True`: restore the plant object to the initial state seen on the first call.
- `False`: continue from the current object state.

Effect:

- `True` prevents repeated notebook executions from starting from a stale plant state.
- `False` is useful only when you intentionally want continuation.

Recommendation:

- Keep `True` in notebooks.
- Still recreate `PolymerCSTR(...)` before each run when doing comparisons.

## 2. Lyapunov Acceptance Parameters

The candidate action is accepted if:

`V(e_x^+) <= rho * V(e_x) + eps_lyap`

where `V(e_x) = e_x^T P_x e_x`.

### `rho_lyap`

Where:

- `run_config["rho_lyap"]` in both notebooks
- forwarded into `Lyapunov/safety_filter.py`

Effect:

- Smaller `rho_lyap` is stricter.
- Larger `rho_lyap` is more permissive.

Symptoms:

- Too many rejected candidates or too many QCQP repairs:
  increase `rho_lyap` slightly.
- Too much oscillation accepted as "safe":
  decrease `rho_lyap`.

Typical range:

- `0.95` to `0.995`

Current common value:

- `0.98`

### `lyap_eps`

Meaning:

- Additive tolerance in the Lyapunov bound.

Effect:

- Larger `lyap_eps` relaxes the decrease test.
- Smaller `lyap_eps` makes the filter more exact and more brittle.

Symptoms:

- Numerical false rejections near the boundary:
  increase `lyap_eps` a little.
- Too much acceptance near steady state:
  reduce it.

Typical range:

- `1e-10` to `1e-6`

### `lyap_tol`

Meaning:

- Numerical tolerance for bound and move checks in post-checking.

Effect:

- Mostly numerical, not conceptual.

Symptoms:

- Very small violations causing needless rejection:
  loosen slightly.

Typical range:

- `1e-10` to `1e-8`

## 3. QCQP Correction Objective Weights

When a candidate fails the Lyapunov test, the filter solves a correction problem with terms of the form:

- stay close to the candidate
- stay close to the previous applied move
- stay close to the target steady input
- optionally improve the one-step predicted output

### `w_mpc` / `w_rl`

Meaning:

- Weight on deviation from the upstream candidate.

Effect:

- Larger value: preserve the upstream action more aggressively.
- Smaller value: allow larger corrections.

Symptoms:

- QCQP makes huge jumps:
  increase `w_mpc` or `w_rl`.
- QCQP barely changes unsafe actions and still struggles:
  decrease it.

### `w_move`

Meaning:

- Weight on change from the previous applied input.

Effect:

- Larger value: smoother control, less chattering.
- Smaller value: faster correction, more aggressive moves.

Symptoms:

- Oscillatory or chattering safe input:
  increase `w_move`.
- Sluggish recovery after a setpoint change:
  decrease `w_move`.

Current commonly used value:

- `0.2`

### `w_ss`

Meaning:

- Weight pulling the corrected input toward the selector steady input `u_s`.

Effect:

- Larger value: stronger pull toward steady-state consistency.
- Smaller value: more freedom to follow candidate or move objectives.
- This term still pulls toward `u_s`, even when `mpc_target_policy="raw_setpoint"`, because there is no raw-setpoint steady input without solving the admissible target problem.

Symptoms:

- Input stays far from `u_s` after the output has reached setpoint:
  increase `w_ss`.
- Filter over-pulls toward a bad moving target:
  decrease `w_ss`.

Current commonly used value:

- `0.1`

### `w_track`

Meaning:

- Weight on the one-step predicted output error `y_{k+1} - y_s`.

Effect:

- Larger value: correction is more output-focused.
- Smaller value: correction is more input-regularization-focused.
- The output reference used here follows `mpc_target_policy`.
  - `"raw_setpoint"` means the QCQP output term tracks raw `y_sp`.
  - `"admissible_if_available"` and `"admissible_on_fallback"` make the QCQP output term track the selected admissible target output.

Symptoms:

- QCQP keeps inputs smooth but output drifts:
  increase `w_track`.
- QCQP makes abrupt input changes to fix one-step output:
  decrease `w_track`.

## 4. Diagonal Weight Vectors

These shape each channel separately.

### `Qy_track_diag`

Meaning:

- Per-output weight used inside Lyapunov ingredient design and correction output term.

Effect:

- Larger weight on one output makes the filter prioritize that output's one-step behavior.

Use when:

- One output is much more important than the other.

### `Rmove_diag`

Meaning:

- Per-input move penalty base vector.

Effect:

- Larger entry on one actuator suppresses changes on that actuator.

Use when:

- One manipulated variable is expensive or should move less.

### `Qs_tgt_diag`

Meaning:

- Weight in the refined target selector for hitting the requested setpoint.

Effect:

- Larger values push the selector harder toward the requested output target.
- If the target is not admissible, very large values can make the selector live permanently in fallback mode with large slack.

Current strong-tracking choice:

- `1e8 * MPC_obj.Q_out`

### `Ru_tgt_diag`

Meaning:

- Weight in the refined target selector on steady input magnitude or deviation.

Effect:

- Larger values prefer milder steady inputs.
- Smaller values allow more aggressive `u_s` to hit the target.

Current common choice:

- ones vector

### `w_x_tgt`

Meaning:

- Small regularization on steady state `x_s`.

Effect:

- Helps select one equilibrium when many are nearly equivalent.
- Too large a value can bias the target away from the output request.

Typical range:

- `1e-8` to `1e-4`

## 5. Trust Region and Slack Controls

### `trust_region_delta`

Meaning:

- Per-input cap on how far the QCQP correction may move away from the candidate.

Effect:

- Smaller trust region means smaller corrections and more fallback risk.
- Larger trust region gives the QCQP more room to find a certified action.

Use when:

- You want to stop large one-step projection kicks.

### `allow_trust_region_slack`

Meaning:

- Whether the trust region itself is soft when `trust_region_delta` is active.

Effect:

- `False`: the trust region is a hard cap.
- `True`: the QCQP may violate the trust region by paying slack.

Recommendation:

- Keep `False` while debugging projection kicks.

### `allow_lyap_slack`

Meaning:

- Whether the QCQP may soften the Lyapunov inequality.

Effect:

- `False`: hard safety check, better for certification.
- `True`: more feasible corrections, but weaker guarantee.

Recommendation:

- Keep `False` during scientific comparison unless you are explicitly studying soft safety.

### `lyap_acceptance_mode`

Meaning:

- How solved QCQP actions are accepted after the post-check.

Options:

- `"hard_only"`: only hard-feasible corrected actions are accepted as control.
- `"accept_slacked"`: optional research mode that can apply a slacked QCQP correction when bounds and move checks pass.

Recommendation:

- Keep `"hard_only"` for the default scientific path.

### `lyap_slack_weight`

Meaning:

- Penalty on Lyapunov slack when `allow_lyap_slack=True`.

Effect:

- Larger value makes the slack almost never used.
- Smaller value can lead to safety violations being bought too cheaply.

### `trust_region_weight`

Meaning:

- Penalty on trust-region slack when a trust region is active.

Effect:

- Larger value enforces the trust region more tightly.

## 6. Input and Move Bounds

### `u_min`, `u_max`

Meaning:

- Hard actuator bounds in scaled deviation coordinates.

Effect:

- Wrong scaling here will destroy performance immediately.

Check carefully:

- These must correspond to the scaled deviation domain expected by the MPC and Lyapunov filter.

### `du_min`, `du_max`

Meaning:

- Hard move-rate bounds.

Effect:

- Tighter move bounds reduce aggressive corrections but can make the QCQP or even the fallback policy infeasible.

Use when:

- You want rate-limited actuators or smoother trajectories.

## 7. Target Selector Warm Start Parameters

These are internal to the target selector call path:

- `prev_target`
- `x_s_prev`
- `u_s_prev`
- `Qdx_diag`
- `Rdu_diag`
- `selector_warm_start`

Effect:

- These do not change the plain MPC candidate directly when `mpc_target_policy="raw_setpoint"`.
- They do change the equilibrium used for Lyapunov checking and the `u_s` regularization term, so they can still change acceptance and QCQP corrections even when the QCQP output term is tracking raw `y_sp`.

### `selector_warm_start`

Meaning:

- Whether the selector seeds the CVXPY variables from the previous valid target and asks the solver to warm start.

Recommendation:

- Keep `True` unless you are explicitly testing solver sensitivity.

### `Qdx_tgt_diag`

Meaning:

- State-target smoothing on the fallback selector stage.

Effect:

- Small positive values discourage the fallback selector from jumping the target state too far between steps.

Current default:

- `1e-6 * ones(n_x)` in the safe-MPC and safe-RL runner defaults.

Symptoms:

- `target_stage` remains `"fallback"` for the whole run:
  inspect `Qs_tgt_diag`, `Ru_tgt_diag`, disturbance estimates, and whether the requested setpoint is actually admissible.

## 8. Solver Preferences

### `target_solver_pref`

Meaning:

- Preferred CVXPY solver order for the target selector.

### `filter_solver_pref`

Meaning:

- Preferred solver order for the QCQP correction.

Default practical interpretation:

- Keep the repository defaults unless you are debugging numerical issues.

## 9. Practical Tuning by Symptom

### Symptom: "MPC alone is stable but Lyapunov path oscillates"

Check in this order:

1. `mpc_target_policy` should be `"raw_setpoint"` for equivalence checks.
2. `reuse_mpc_solution_as_ic` should be `False`.
3. `reset_system_on_entry` should be `True`.
4. Compare `u_cand` vs `u_safe`.
5. If `u_cand == u_safe` most of the time, the oscillation is upstream MPC or observer behavior, not the QCQP.
6. If QCQP kicks are large, increase `w_move`, increase `w_mpc` or `w_rl`, or add `trust_region_delta`.

### Symptom: "Fallback MPC is active most of the time"

Check:

1. `target_success`
2. `target_stage`
3. `candidate_lyap_ok`
4. `fallback_verified`

Interpretation:

- `target_success=False` a lot means the selector is failing, not the QCQP.
- `candidate_lyap_ok=False` with successful targets means the certificate is too strict for the current policy or model mismatch.

### Symptom: "Reached the setpoint but input keeps moving"

Likely knobs:

- increase `w_ss`
- increase `w_move`
- decrease `rho_lyap` slightly if unsafe oscillatory candidates are being accepted

### Symptom: "The filter is too conservative"

Possible changes:

- increase `rho_lyap`
- increase `lyap_eps`
- decrease `w_ss`
- enlarge `trust_region_delta`

### Symptom: "The filter makes big one-step jumps"

Possible changes:

- increase `w_move`
- increase `w_mpc` or `w_rl`
- add a finite `trust_region_delta`
- optionally add move-rate bounds with `du_min`, `du_max`

## 10. Recommended Tuning Order

When debugging performance, tune in this order:

1. Lock equivalence switches:
   - `mpc_target_policy="raw_setpoint"`
   - `reuse_mpc_solution_as_ic=False`
   - `reset_system_on_entry=True`
2. Verify scaling and steady-state anchors.
3. Verify `target_success` and `target_stage`.
4. Tune `rho_lyap`, `lyap_eps`, `lyap_tol`.
5. Tune correction weights:
   - `w_move`
   - `w_ss`
   - `w_mpc` or `w_rl`
   - `w_track`
6. Only then add trust regions, move-rate limits, or slack.

## 11. Minimum Logging to Watch While Tuning

Always inspect these fields from the debug export:

- `correction_mode`
- `target_success`
- `target_stage`
- `candidate_lyap_ok`
- `fallback_verified`
- `u_cand`
- `u_safe`
- `u_s`
- `V_k`
- `V_next_cand`
- `V_bound`
- `target_error_inf`
- `target_mismatch_inf`

If those traces are not consistent with your expectation, do not tune weights yet. Fix the representation or logic mismatch first.
