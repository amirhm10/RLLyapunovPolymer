# GART dy2-Only Default

## Summary

Removed the dy4 GART case from the editable root runner and the default closed-loop study matrix. The forward GART-LMPC default is now the single raw dy2 controller:

```text
gart_target_raw_no_dx_headroom_0p01_dy2_no_umid
```

Mixed cases remain disabled, and the old governed-reference case remains disabled for manual comparison only.

## Rationale

The latest full disturbance comparison showed dy2 and dy4 had essentially identical tracking performance:

| Case | Reward Mean | Output RMSE | Target Mismatch | Governor Active |
|---|---:|---:|---:|---:|
| raw dy2 no-u-mid | -4.157946 | 0.370104 | 0.561271 | 0.01125 |
| raw dy4 no-u-mid | -4.157946 | 0.370104 | 0.561353 | 0.00450 |

Since dy4 did not improve tracking or target quality, dy2 is the cleaner forward setting. It keeps the target-output rate bound more conservative:

$$
\left|y_s(k)-y_s(k-1)\right| \le 2 d y_{\max,\mathrm{base}}.
$$

## Implementation

- Removed `GART_RELAXED_DY4_OVERRIDES` from the default experiment runner.
- Removed dy4 from `GARTLyapunovMPC.py` case specs.
- Removed dy4 from the default `run_closed_loop(...)` case list.
- Removed dy4 entries from the target-ablation matrix.
- Updated automatic closed-loop guard sizing from two default cases to one.

## Validation

Passed:

```powershell
python -m py_compile GARTLyapunovMPC.py experiments\run_gart_target_selector_study.py
```

Passed one-case smoke run:

```powershell
python GARTLyapunovMPC.py --closed-loop --no-target-only --mode nominal --n-tests 1 --set-points-len 5 --timestamp codex_smoke_dy2_only --max-target-evals 120 --max-closed-loop-steps 120 --max-solver-calls 120 --max-wall-clock-seconds 240
```

Smoke output confirmed:

```text
enabled_cases = ['gart_target_raw_no_dx_headroom_0p01_dy2_no_umid']
```

The short nominal smoke solved successfully with hard contraction active.
