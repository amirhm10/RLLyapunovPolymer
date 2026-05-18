# Direct Lyapunov Target Quality Gate And RL Guard

Date: 2026-05-18

## Purpose

This report explains the latest three direct-notebook results and documents the method changes that were implemented after the diagnosis. The main goal is to make the controller logic visible enough that the method can be reviewed, tuned, or rolled back before another long notebook run.

The active case is the polymer CSTR direct Lyapunov workflow in scaled deviation coordinates. Manipulated inputs are the coolant and monomer-related flows, usually `Qc` and `Qm`. Outputs are viscosity-like `eta` and reactor temperature `T`. The disturbed tests change process variables such as `Qi`, `Qs`, and `hA`.

The key conclusion is unchanged:

`mpc_only` is often better because it tracks the raw setpoint directly, while the Lyapunov-gated controller can enforce contraction around a target that is a poor certificate anchor under the disturbed plant.

## Result Bundles Reviewed

### No-RL Direct MPC

Bundle:

`results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260518_150230/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Lyap | 0.436 | -5.70 |
| mpc_only | 0.357 | -3.88 |

Tail tracking is the strongest evidence:

| Case | Final physical error |
|---|---:|
| Lyap | `[0.125, -0.598]` |
| mpc_only | `[0.004, -0.020]` |

Interpretation:

The `mpc_only` run nearly removes the tail offset, while the Lyapunov path keeps a clear residual error, especially in the temperature-like output. This is not an RL reward issue because this notebook does not use an RL policy. The failure has to be in the direct target, disturbance model, finite-horizon constraint interaction, or Lyapunov certificate anchor.

### RL Cold-Start

Bundle:

`results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260518_165924/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Safe gate | 0.265 | -3.209 |
| mpc_only | 0.239 | -2.225 |

Interpretation:

The safe gate improves over the worse no-RL Lyapunov behavior, but it still underperforms `mpc_only`. The likely reason is the same: actions are judged against the Lyapunov target, not directly against raw setpoint performance. A safe action can be poor if the target used for safety is poor.

### RL Pretrained

Bundle:

`results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260518_165928/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Safe gate | 0.255 | -3.036 |
| mpc_only | 0.245 | -2.249 |

Interpretation:

Pretraining narrows the gap, but it does not remove it. That means the bottleneck is not only policy quality. A stronger policy still passes through a target/certificate layer whose reference may be misaligned with the raw setpoint under disturbance.

## What The Numbers Mean

The three bundles form a consistent picture:

| Evidence | Meaning |
|---|---|
| No-RL Lyap worse than `mpc_only` | The problem exists before RL enters. |
| `mpc_only` tail error is near zero | The plant and baseline MPC can still track the disturbed two-setpoint schedule. |
| Lyap tail error remains large | The Lyapunov target/certificate path can anchor the controller away from the raw setpoint. |
| RL safe gate close but worse | RL is not the first bottleneck; the gate and target are. |
| Pretraining helps only slightly | Better policy initialization does not fix a poor target selector. |

The important separation is:

- Raw tracking quality: how well the closed loop follows the requested setpoint.
- Target quality: how close the steady target is to the requested setpoint and how small its residual is.
- Certificate quality: whether a one-step Lyapunov decrease is meaningful around that target.

The old method mixed these. It could certify contraction even when the target itself was the wrong object to contract around.

## Previous Direct Lyapunov Method

This section reconstructs the method before the new changes.

### Coordinates

Most direct-controller logic is in scaled deviation coordinates.

Important variables:

- `xhatdhat`: augmented observer state, made of physical-state estimate and disturbance estimate.
- `u_prev_dev`: previous input in scaled deviation coordinates.
- `y_sp_k`: setpoint at step `k`, also in scaled deviation coordinates.
- `u_dev_min`, `u_dev_max`: input bounds in scaled deviation coordinates.
- `x_s`, `u_s`, `d_s`, `y_s`: steady target selected for the current setpoint and disturbance estimate.

Observer update, in plain notation:

`xhatdhat(k+1) = A_aug * xhatdhat(k) + B_aug * u(k) + L * innovation(k)`

where:

`innovation(k) = y_measured_scaled(k) - C_aug * xhatdhat(k)`

### Old Target Solve

The direct target layer solved a frozen output-disturbance target. In the output-disturbance case, the disturbance estimate acts directly at the output:

`y_s = C * x_s + d_hat`

The target equations were:

`x_s = A * x_s + B * u_s`

`y_s should match y_sp`

When the exact target was outside input bounds, the old bounded fallback used a single least-squares problem. That problem combined:

- steady-state dynamic residual
- output-setpoint mismatch
- previous-input anchor, if `u_ref_weight` was nonzero
- previous-state smoothing, if `x_ref_weight` was nonzero

The problem is that these terms were all in one objective. A strong previous-input or previous-state regularizer could pull the target away from the best reachable output. That is useful for smoothness but dangerous if it redefines the output target.

### Old Direct Lyapunov MPC Step

Once the target was selected, the direct tracking MPC solved over an input sequence. The output objective tracked either raw `y_sp_k` or `y_s`, depending on config. In the direct notebooks here, the important behavior is that the Lyapunov certificate was still formed around `x_s`.

The hard first-step Lyapunov condition was effectively:

`V(x_next - x_s) <= rho * V(x_now - x_s) + eps`

If `x_s` is a poor target, this condition can reject or reshape actions that would track the raw setpoint well.

### Old RL Direct Gate

For the direct RL safety gate:

1. The actor proposed an action.
2. The proposed input was mapped to input bounds.
3. The direct target was recomputed.
4. The candidate was checked for bounds, move limits, and Lyapunov decrease around the direct target.
5. If the candidate was unsafe, the direct tracking MPC fallback was applied.

The missing piece was a performance check. A candidate could be Lyapunov-safe but still worse than a fallback or direct MPC action for raw setpoint tracking.

## Root Cause Diagnosis

### Why `mpc_only` Wins In The No-RL Disturbed Run

`mpc_only` does not enforce the direct Lyapunov certificate. It solves the offset-free tracking problem against the raw setpoint. In the latest no-RL result, this gives a final physical error of `[0.004, -0.020]`, which is essentially zero at the tail.

The Lyapunov controller solves a more constrained problem. Even if the output objective uses the raw setpoint, the hard contraction and terminal ingredients are built around `x_s`. If `x_s` is produced by a target selector that is not consistent with the disturbed plant, the optimizer can be pushed toward the certificate target instead of the raw setpoint.

So the no-RL result says:

- The baseline MPC model and horizon are good enough to nearly remove tail offset.
- The Lyapunov target/certificate layer is introducing a competing objective.
- The reward function cannot be the root cause because no RL reward is optimized in this controller.

### Why RL Does Not Fully Fix It

The RL policy can only propose actions. The gate decides what is allowed. If the gate certifies safety around a poor target, then policy improvement does not guarantee raw tracking improvement.

Pretraining helps because the actor starts closer to useful actions. But the pretrained result still loses to `mpc_only`, so the gate logic and target selector remain the bottleneck.

### Why `mpc_only` Can Still Sometimes Have Offset

When `mpc_only` has offset, that is not caused by the RL reward. `mpc_only` does not optimize the RL reward. Offset in `mpc_only` points to:

- disturbance-model mismatch
- target calculation mismatch
- observer augmentation mismatch
- finite-horizon effects
- active input constraints
- move suppression or input saturation
- plant changes not represented by a frozen output disturbance

Reward tuning can improve RL maintenance behavior, but it cannot explain offset in a pure MPC-only run.

## Implemented Method Changes

This section explains exactly what changed and where.

### Change 1: Target Quality Gate

Main file:

`Lyapunov/direct_lyapunov_mpc.py`

Key functions:

- `DEFAULT_DIRECT_TARGET_CONFIG`
- `_target_quality_config`
- `_annotate_target_quality`
- `prepare_direct_output_disturbance_step`
- `solve_direct_tracking_from_target`

New config structure:

```python
direct_target_config = {
    "target_quality": {
        "enabled": True,
        "policy": "bypass_hard_lyap",
        "max_mismatch_inf": 0.03,
        "max_residual_norm": 0.10,
        "max_rate_inf": 0.20,
    }
}
```

All thresholds are in scaled deviation coordinates.

The gate computes:

| Quantity | Meaning |
|---|---|
| `target_quality_mismatch_inf` | Infinity norm of `y_s - y_sp`. |
| `target_quality_residual_norm` | Main target residual norm, using available target diagnostics. |
| `target_rate_inf` | Infinity norm of target-state jump from previous successful target. |
| `target_quality_ok` | True if all enabled checks pass. |
| `target_quality_reason` | Text label explaining why a target was poor. |
| `target_quality_bypass` | True when a poor target should not receive hard Lyapunov enforcement. |

Step-by-step behavior:

1. The target selector solves for `x_s`, `u_s`, `d_s`, and `y_s`.
2. The quality gate compares `y_s` to raw `y_sp_k`.
3. The gate checks the target residual.
4. The gate checks whether `x_s` jumps too far from the previous target.
5. If all active checks pass, the normal Lyapunov controller is used.
6. If a target is poor and `policy="bypass_hard_lyap"`, the controller logs a bypass.
7. During bypass, the direct solver tracks raw `y_sp_k`.
8. During bypass, hard first-step contraction is disabled.
9. During bypass, the terminal set constraint is skipped.

What did not change by default:

- `target_quality.enabled` defaults to `False`.
- Existing notebooks keep the old behavior unless the config is enabled.

Important review note:

The current implementation still lets a numerically successful but poor target update `x_target_prev_success_next`. If we want poor targets not to seed the next target-smoothing reference, change the update condition from `target_success` to `target_success and target_quality_ok`. This is a reasonable follow-up if the next run shows target-quality bypasses clustered after large target jumps.

### Change 2: Lexicographic Bounded Target Solve

Main files:

- `analysis/steady_state_debug_analysis.py`
- `Lyapunov/frozen_output_disturbance_target.py`

New config:

```python
direct_target_config = {
    "solve_strategy": "lexicographic",
    "lexicographic_primary_tol_abs": 1.0e-10,
    "lexicographic_primary_tol_rel": 1.0e-8,
    "lexicographic_maxiter": 200,
    "lexicographic_ftol": 1.0e-10,
}
```

Old strategy:

`solve_strategy="legacy_ls"`

This keeps the previous single-stage least-squares behavior.

New strategy:

`solve_strategy="lexicographic"`

Stage 1:

Minimize the primary steady-target quality. In reduced form this is the output mismatch:

`primary_cost = norm(G * u_s - rhs_output)^2`

In full form it minimizes the stacked steady-state residual:

`primary_cost = norm(M * [x_s, u_s] - rhs)^2`

Stage 2:

Minimize smoothing only inside a small tolerance of the Stage 1 primary cost:

`anchor_cost = weighted_norm(u_s - u_ref)^2 + weighted_norm(x_s - x_ref)^2`

The Stage 2 constraint is:

`primary_cost <= stage1_primary_cost + tolerance`

Why this matters:

The previous target solve could sacrifice output fit to satisfy `u_ref` or `x_ref` smoothing. The new solve says: first get the best reachable target, then smooth only if smoothing does not damage target quality.

Synthetic validation:

| Strategy | u target | output residual |
|---|---:|---:|
| legacy_ls | 0.009901 | 0.990099 |
| lexicographic | 0.200000 | 0.800000 |

The synthetic case had an unreachable setpoint and a strong previous-input anchor at zero. The old solve was pulled almost to zero input. The lexicographic solve stayed at the upper bound because that minimized output mismatch first.

What can be changed:

- Use `legacy_ls` if the old behavior is needed for comparison.
- Increase `lexicographic_primary_tol_abs` or `lexicographic_primary_tol_rel` if small target-quality sacrifices are acceptable for smoother inputs.
- Keep these tolerances small for the disturbed two-setpoint runs until we confirm the target selector is no longer the bottleneck.

### Change 3: Disturbance Model Mode

Main file:

`Lyapunov/frozen_output_disturbance_target.py`

New config:

```python
direct_target_config = {
    "disturbance_model_mode": "output"
}
```

Allowed values:

| Mode | Meaning |
|---|---|
| `output` | Existing frozen output-disturbance model. |
| `state_via_B` | Use generic augmented target selector when the augmented model exposes state disturbance channels. |
| `mixed` | Use generic augmented target selector when both state and output disturbance effects are represented. |

Important limitation:

This change exposes the target-side path. It does not magically redesign the observer, augmentation matrices, or disturbance estimator. For `state_via_B` and `mixed` to be meaningful, the notebooks must pass an augmented model and observer gain whose disturbance states actually represent the disturbance channel.

Why this matters:

The disturbed tests change `Qi`, `Qs`, and `hA`. These changes do not necessarily appear as a pure additive output bias. A frozen output disturbance can compensate some offset, but it can also produce a target that is internally inconsistent with how the plant actually moved.

What can be changed:

- Keep `output` for compatibility and baseline comparisons.
- Test `state_via_B` only after confirming the augmented matrices and observer gain are built for that disturbance structure.
- Log the mode in every bundle so target diagnostics can be compared mode-by-mode.

### Change 4: Direct RL Performance Guard

Main file:

`Simulation/run_rl_lyapunov.py`

New function/config:

- `_normalize_performance_guard_config`
- `performance_guard_config`

Example:

```python
performance_guard_config = {
    "enabled": True,
    "reference_policy": "direct_mpc",
    "abs_tol": 0.0,
    "rel_tol": 0.05,
}
```

Step-by-step behavior:

1. RL proposes `u_rl`.
2. The direct Lyapunov gate checks whether `u_rl` is safe.
3. If `u_rl` is not safe, fallback MPC is used as before.
4. If `u_rl` is safe and the performance guard is disabled, `u_rl` can be accepted as before.
5. If `u_rl` is safe and the performance guard is enabled, the code computes a one-step raw tracking cost.
6. The same cost is computed for a reference action, either direct MPC fallback or hold-previous input.
7. If the RL action is worse by more than tolerance, it is rejected with `reject_reason="performance_guard"`.
8. The controller then uses the direct fallback path.

The one-step raw tracking cost is:

`cost = weighted_output_error_next + weighted_move`

where:

`weighted_output_error_next = sum(Q_i * (y_next_i - y_sp_i)^2)`

`weighted_move = sum(R_j * (u_j - u_prev_j)^2)`

Why this matters:

Safety is necessary but not sufficient. A safe RL action can be worse than the fallback for raw tracking. This guard adds the missing performance logic.

What can be changed:

- `reference_policy="direct_mpc"` is stronger but more expensive because it can solve the direct fallback for comparison.
- `reference_policy="hold_prev"` is cheaper but weaker.
- Increase `rel_tol` if the guard rejects too many actions and collapses into MPC-only behavior.
- Use the guard first for short RL smoke tests, not full training immediately.

### Change 5: Residual RL Option

Main file:

`Simulation/run_rl_lyapunov.py`

New function/config:

- `_normalize_residual_rl_config`
- `residual_rl_config`

Example:

```python
residual_rl_config = {
    "enabled": True,
    "baseline_policy": "offset_free_mpc",
    "authority_scale": 0.20,
    "shrink_error_inf": 0.05,
    "min_authority_scale": 0.10,
}
```

Old RL action interpretation:

`u = map_actor_action_to_input_bounds(action)`

New residual option:

`u = u_baseline + authority * action`

Then the input is clipped to bounds.

Step-by-step behavior:

1. Compute a baseline input.
2. The baseline is `offset_free_mpc` if available, otherwise previous input.
3. Compute residual authority.
4. Optionally shrink authority near the setpoint using `shrink_error_inf`.
5. Add the actor residual to the baseline.
6. Clip to input bounds.
7. Pass the resulting action to the safety/performance gate.

Why this matters:

The latest results show that `mpc_only` is already strong. Residual RL lets the policy improve or maintain MPC behavior instead of replacing it with a full-authority action.

What can be changed:

- Use `authority_scale=0.10` to `0.25` for conservative tests.
- Use a larger value only after the target-quality gate is working.
- Use `baseline_policy="previous_input"` only for ablations; `offset_free_mpc` is the more meaningful baseline.

### Change 6: RL Maintenance Reward Terms

Main file:

`TD3Agent/reward_functions.py`

New optional arguments:

```python
maintenance_band_scale=1.0
maintenance_move_weight=0.0
jitter_weight=0.0
dwell_bonus=0.0
```

Defaults preserve old behavior.

Step-by-step reward addition:

1. The original relative QR reward is computed.
2. The code checks whether all outputs are inside the maintenance band.
3. If inside the band, an additional move penalty can be applied.
4. If the previous output error exists, an output-jitter penalty can be applied.
5. If the output remains inside the band, the dwell counter increases.
6. A dwell bonus can be added proportional to the dwell count.

Why this matters:

The RL notebooks show maintenance and jitter concerns near the setpoint. A reward that only encourages entering the band may not sufficiently discourage small near-setpoint moves or output oscillation.

What can be changed:

- Keep all maintenance weights zero until target-quality and performance-gate fixes are tested.
- Add a small `maintenance_move_weight` first.
- Add `jitter_weight` only if the output trace still oscillates after the gate changes.
- Use `dwell_bonus` carefully, because too large a dwell bonus can encourage passive behavior.

## Export And Diagnostics Added

Direct debug exports now include:

- `target_quality_enabled`
- `target_quality_ok`
- `target_quality_reason`
- `target_quality_policy`
- `target_quality_bypass`
- `target_quality_mismatch_inf`
- `target_quality_residual_norm`
- `target_rate_inf`

RL safety debug exports now include:

- target-quality fields
- performance-guard fields
- residual-RL fields

Useful summary metrics:

| Metric | Why it matters |
|---|---|
| `target_quality_ok_rate` | Fraction of steps with acceptable target anchors. |
| `target_quality_bypass_rate` | Fraction of steps where hard Lyapunov was bypassed. |
| `target_rate_inf_max` | Largest target jump. |
| `performance_guard_ok_rate` | Fraction of checked RL actions that passed raw tracking comparison. |
| `target_quality_mismatch_inf_max` | Worst target-setpoint mismatch. |
| `target_quality_residual_norm_max` | Worst target residual. |

## Recommended Config For The Next No-RL Run

Start with the no-RL direct comparison, not RL training. The goal is to confirm that the target/certificate fix is working before touching reward design.

Suggested first config:

```python
direct_target_config = {
    "solve_strategy": "lexicographic",
    "disturbance_model_mode": "output",
    "target_quality": {
        "enabled": True,
        "policy": "bypass_hard_lyap",
        "max_mismatch_inf": 0.03,
        "max_residual_norm": 0.10,
        "max_rate_inf": 0.25,
    },
}
```

Things to inspect after the run:

- Does `mpc_only` still end near zero tail offset?
- Does Lyap stop enforcing hard contraction when `target_quality_bypass=True`?
- Does Lyap tail error improve relative to `[0.125, -0.598]`?
- Does `target_quality_bypass_rate` cluster around setpoint changes or disturbed intervals?
- Does `target_quality_mismatch_inf` drop under lexicographic solve?

If too many steps bypass, loosen thresholds or fix the disturbance model. If almost no steps bypass but Lyap still has offset, the quality thresholds are too loose or the target residual metric is not catching the bad anchor.

## Recommended Config For Short RL Smoke Test

After the no-RL run looks better, use a short RL smoke test. Do not start with another full 160000-step run.

Suggested first RL guard config:

```python
performance_guard_config = {
    "enabled": True,
    "reference_policy": "direct_mpc",
    "abs_tol": 0.0,
    "rel_tol": 0.05,
}
```

Suggested residual RL config:

```python
residual_rl_config = {
    "enabled": True,
    "baseline_policy": "offset_free_mpc",
    "authority_scale": 0.20,
    "shrink_error_inf": 0.05,
    "min_authority_scale": 0.10,
}
```

Suggested reward maintenance config:

```python
reward_params = {
    "maintenance_band_scale": 1.0,
    "maintenance_move_weight": 0.05,
    "jitter_weight": 0.01,
    "dwell_bonus": 0.0,
}
```

Only tune the reward after target-quality diagnostics show that the gate is no longer certifying around poor targets.

## What I Would Change First If The Next Run Is Still Bad

1. Prevent poor targets from updating the previous-target smoothing reference.
2. Add plots for `target_quality_bypass`, `target_quality_mismatch_inf`, and output error on the same time axis.
3. Compare `output` versus `mixed` disturbance mode only after verifying the augmented observer model.
4. Reduce or disable `x_ref_weight` and `u_ref_weight` for disturbed tests if lexicographic solve still shows target jumps.
5. Increase `performance_guard.rel_tol` if the RL safe gate becomes too conservative.
6. Reduce residual RL authority if action jitter remains near the setpoint.

## Risks And Caveats

The target-quality gate is intentionally conservative. If thresholds are too tight, the controller may bypass hard Lyapunov too often and behave closer to tracking MPC.

The lexicographic target solve is more principled for unreachable setpoints, but it adds a second optimization stage. It should be monitored for solve time in long notebooks.

The `state_via_B` and `mixed` disturbance modes require a compatible augmented model and observer. The current code exposes the path, but the notebooks still need to supply the correct model.

The performance guard adds a performance criterion to a safety gate. This is desirable for the current failure mode, but it can reduce RL exploration if the tolerance is too strict.

The reward maintenance terms are stateful because the dwell and jitter terms depend on previous reward calls. This is suitable for sequential rollout but should be kept in mind if the same reward object is reused across independent episodes without reset.

## Literature Basis

Muske and Badgwell discuss offset-free MPC disturbance models and why disturbance structure matters when rejecting sustained offsets: https://www.sciencedirect.com/science/article/pii/S0959152401000518

Pannocchia and Bemporad emphasize that disturbance model, observer, target calculation, and dynamic controller should be designed together for offset-free MPC: https://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeetac-distmodel.pdf

Shead, Muske, and Rossiter motivate caution around constrained target calculation, because active constraints can drive the controller toward an undesired feasible target: https://www.sciencedirect.com/science/article/abs/pii/S0959152410001812

Limon et al. motivate artificial/admissible references for tracking MPC when requested references change or are not reachable: https://www.sciencedirect.com/science/article/pii/S0959152409002169

Predictive safety-filter work supports the RL-side change: safety certification should be paired with performance logic when safe actions can still be poor tracking choices: https://www.sciencedirect.com/science/article/pii/S0005109821001175

## Validation Completed

Code syntax validation:

`python -m py_compile` passed on the touched modules using a temporary bytecode directory because the normal OneDrive pycache path denied writes.

Synthetic target validation:

| Strategy | u target | output residual |
|---|---:|---:|
| legacy_ls | 0.009901 | 0.990099 |
| lexicographic | 0.200000 | 0.800000 |

Target-solver smoke validation:

The target-solver smoke test passed. The full direct-controller smoke test could not run in this environment because `cvxpy` is not installed for the default Python interpreter.

## Bottom Line

The next experiment should not start with reward tuning. First, rerun the no-RL disturbed direct comparison with lexicographic target selection and target-quality bypass enabled. If that fixes the Lyap tail offset, then run a short RL smoke test with the performance guard and residual-RL authority. Reward maintenance terms should be the third step, after the target and gate are no longer the bottleneck.
