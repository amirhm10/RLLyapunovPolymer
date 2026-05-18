# Direct Lyapunov Target Quality Gate And RL Guard

Date: 2026-05-18

## Latest Result Diagnosis

The latest three direct notebook exports point to the same root cause: the direct Lyapunov gate is certifying contraction around a target that is sometimes a poor anchor for the disturbed plant. The raw `mpc_only` branch can look better because it keeps optimizing the raw setpoint, while the Lyapunov branch can spend authority satisfying a certificate around a shifted or residual-heavy target.

### No-RL Direct MPC

Bundle: `results/direct_lyapunov_mpc_bounded_four_method_two_setpoint_disturb/20260518_150230/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Lyap | 0.436 | -5.70 |
| mpc_only | 0.357 | -3.88 |

Tail behavior is decisive. The final physical error for `mpc_only` is `[0.004, -0.020]`, while Lyap ends at `[0.125, -0.598]`. That is not a reward-design artifact, because `mpc_only` does not optimize the RL reward. It is a target/model/certificate issue.

### RL Cold-Start

Bundle: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260518_165924/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Safe gate | 0.265 | -3.209 |
| mpc_only | 0.239 | -2.225 |

The safe gate is close but still worse. The target mismatch diagnostics remain large during some disturbed intervals, so an action can be safe around the certificate target while not being the best next action for raw setpoint tracking.

### RL Pretrained

Bundle: `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260518_165928/`

| Case | RMSE mean | Reward mean |
|---|---:|---:|
| Safe gate | 0.255 | -3.036 |
| mpc_only | 0.245 | -2.249 |

Pretraining narrows the gap, but it does not remove the target bottleneck. That is consistent with the no-RL result: a better policy cannot fully repair a poor certificate anchor.

## Implemented Fixes

Target quality gate:

- `target_quality` can now mark a direct target poor when target-setpoint mismatch, residual norm, or target jump exceeds configured tolerances.
- When enabled and poor, the direct solver tracks raw `y_sp` and bypasses hard first-step Lyapunov enforcement around that target.
- Exports now include `target_quality_ok`, `target_quality_reason`, `target_rate_inf`, and `target_quality_bypass`.

Lexicographic bounded target solve:

- `solve_strategy="legacy_ls"` keeps the previous behavior.
- `solve_strategy="lexicographic"` first minimizes reachable setpoint and steady-state residual, then minimizes smoothing anchors within a small tolerance of the stage-1 quality.
- This prevents `u_prev` and previous-target smoothing from redefining the output target.

Disturbance model exposure:

- `disturbance_model_mode` now accepts `output`, `state_via_B`, or `mixed`.
- `output` preserves the frozen output-disturbance path.
- `state_via_B` and `mixed` route through the augmented target selector so state-channel disturbance blocks can participate when the observer model exposes them.

RL maintenance and guards:

- The relative reward supports optional dwell bonus, near-band move penalty, and output-jitter penalty.
- The direct RL gate supports an optional performance guard that rejects Lyapunov-safe actions when their one-step raw tracking cost is worse than a direct fallback or hold input beyond tolerance.
- A residual-RL option is available: `u = baseline + authority * action`, with optional authority shrinkage near setpoint.

## Why This Matches The Literature

Muske and Badgwell show that output-disturbance models are common but can be weak when disturbances enter through inputs or states, motivating the new disturbance-mode exposure: https://www.sciencedirect.com/science/article/pii/S0959152401000518

Pannocchia and Bemporad formulate offset-free MPC as a coupled observer, disturbance-model, target-calculator, and dynamic-controller design, which matches the observed failure mode here: https://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeetac-distmodel.pdf

Shead, Muske, and Rossiter explain that active constraints can make an MPC target converge to a suboptimal feasible point, supporting the lexicographic target fix: https://www.sciencedirect.com/science/article/abs/pii/S0959152410001812

Limon et al. motivate artificial/admissible targets for changing or unreachable references, which is the design direction behind guarding poor targets instead of certifying against them blindly: https://www.sciencedirect.com/science/article/pii/S0959152409002169

Predictive safety-filter work supports the RL-side fix: a safety filter can certify safety, but performance still needs a separate acceptance rule when safe actions are poor: https://www.sciencedirect.com/science/article/pii/S0005109821001175

## Validation

- `python -m py_compile` passed on the touched modules using a temporary bytecode directory.
- A synthetic bounded target test confirmed that `legacy_ls` allowed a strong `u_ref` anchor to pull the target away from the best reachable output, while `lexicographic` selected the input bound that minimized output mismatch first.

Synthetic result:

| Strategy | u target | output residual |
|---|---:|---:|
| legacy_ls | 0.009901 | 0.990099 |
| lexicographic | 0.200000 | 0.800000 |

## Next Run Recommendation

Use `solve_strategy="lexicographic"` and enable `target_quality` for the disturbed two-setpoint runs before tuning the RL reward. The first acceptance criterion should be: if the target is poor, log it and avoid hard contraction around it. Reward tuning should come after this, because the current offset in `mpc_only` points to disturbance-model, target-calculation, horizon, and constraint effects rather than the RL reward.
