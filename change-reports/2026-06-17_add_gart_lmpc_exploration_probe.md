# Add GART-LMPC Exploration Probe

Date: 2026-06-17

## Objective

Create a separate diagnostic runner to test whether small executed-input excitation changes the observer disturbance estimate enough to affect GART-LMPC contraction margins or violation behavior.

## Latest Result Context

The latest GART-LMPC run, `results/GARTLMPC/20260617_180012`, used:

```python
DU_S_MAX_ABS = [0.2, 0.2]
DY_S_MAX_ABS = 0.25
D_RATE_SCALE = 0.25
ALPHA_D = 0.05
```

The run reduced large applied-input jumps and had formal hard contraction success, but the target mismatch remained significant. This suggests that the next diagnostic should isolate the observer/disturbance channel instead of only retuning target-rate bounds.

## Change

Added `GARTLyapunovMPC_ExplorationProbe.py`, a copy-style root runner that keeps the current GART-LMPC target-governor tuning and injects small scaled-deviation input excitation:

```python
INPUT_EXPLORATION_STD = [0.005, 0.005]
INPUT_EXPLORATION_SEED = 20260617
```

The probe writes to:

```text
results/GARTLMPCExplorationProbe/<timestamp>
```

## Implementation Details

The closed-loop helper now accepts optional default-off arguments:

```python
input_exploration_std
input_exploration_seed
```

When enabled, excitation is added after the LMPC solve and before plant simulation and observer update. The logs preserve both actions:

- `u_apply_nominal`: solver action before excitation
- `u_apply_executed`: action actually sent to the plant
- `input_exploration_applied`: clipped excitation actually applied

The helper also logs an executed-action Lyapunov contraction diagnostic:

$$
V(x_{k+1}^{\mathrm{exec}} - x_s) - \left(\rho V(x_k - x_s) + \epsilon\right)
$$

This separates direct contraction failure from one-step-later observer/disturbance effects.

## What To Check

After running the probe, compare against the non-exploration run:

- `dhat_delta_inf_mean`, `dhat_delta_inf_p95`, `dhat_delta_inf_max`
- `executed_contraction_satisfied_rate`
- `executed_contraction_violation_max`
- `input_exploration_inf_max`
- `output_rmse_to_ys`
- `target_reference_error_inf_mean`
- large applied `delta_u` events

## Interpretation Rule

If small excitation increases `dhat` movement and the next-step target/contraction behavior worsens, then the disturbance observer is likely part of the root mechanism. If excitation changes `dhat` but executed contraction remains satisfied and the target mismatch is unchanged, then the issue is more likely in the target selection/reference mismatch rather than local observer sensitivity.
