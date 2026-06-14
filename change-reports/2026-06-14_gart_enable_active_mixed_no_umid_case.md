# Enable GART Active Mixed No-u-mid Case

## Summary

Enabled one mixed-objective GART-LMPC closed-loop case after removing the target-selector regularizers that were making the steady target too sticky. The root runner now compares:

- `gart_target_raw_no_dx_headroom_0p01_dy2_no_umid`
- `gart_target_raw_no_dx_headroom_0p01_dy4_no_umid`
- `gart_target_mixed_no_dx_headroom_0p01_dy2_no_umid`

The old governed-reference baseline remains disabled, and the dy4 mixed case is available as a manual sensitivity toggle.

## Method Change

The active mixed case keeps the same cleaned GART target selector:

- `disable_dx_rate=True`
- `disable_u_mid_tiebreak=True`
- `disable_x_smoothing=True`
- `disable_y_smoothing=True`
- `input_headroom_frac=0.01`
- `dy_rate_scale=2.0`

The mixed MPC override is:

```python
eta_y = 0.1
eta_u = 0.1
target_term_gate_enabled = False
```

This makes the mixed case a true diagnostic mixed-objective run. The earlier gated behavior could make the mixed case numerically identical to raw whenever the target remained far from the raw setpoint. The hard Lyapunov contraction and terminal constraints are still active.

## Runtime Change

The editable root runner guard limits were increased for the three active closed-loop cases:

- `MAX_TARGET_EVALS=15000`
- `MAX_CLOSED_LOOP_STEPS=15000`
- `MAX_SOLVER_CALLS=15000`
- `MAX_WALL_CLOCK_SECONDS=21600.0`

The experiment runner's automatic closed-loop guard now budgets for three default closed-loop cases instead of two.

## Validation

Passed:

```powershell
python -m py_compile GARTLyapunovMPC.py experiments\run_gart_target_selector_study.py Lyapunov\gart_lmpc.py Lyapunov\gart_target.py utils\gart_defaults.py
```

Passed smoke run:

```powershell
python GARTLyapunovMPC.py --closed-loop --no-target-only --mode nominal --n-tests 1 --set-points-len 5 --timestamp codex_smoke_mixed_active_no_umid --max-target-evals 300 --max-closed-loop-steps 300 --max-solver-calls 300 --max-wall-clock-seconds 240
```

Smoke-run result:

| case | reward mean | output RMSE mean | solver success | hard contraction |
|---|---:|---:|---:|---:|
| raw dy2 no-u-mid | -40.155 | 1.268 | 1.000 | 1.000 |
| raw dy4 no-u-mid | -40.155 | 1.268 | 1.000 | 1.000 |
| mixed dy2 no-u-mid | -62.869 | 1.534 | 1.000 | 1.000 |

The mixed case was intentionally different from raw in this smoke test. Its step diagnostics had `target_terms_enabled=True` for all 10 steps and `target_term_gate_reason=disabled`.

## Interpretation

The short smoke run suggests the active mixed objective can still pull the optimizer toward a poor admissible target when that target is far from the raw setpoint. That is useful diagnostically: the full run will now show whether the cleaned target selector makes mixed viable over the standard disturbance scenario, rather than silently falling back to raw behavior through the target-term gate.
