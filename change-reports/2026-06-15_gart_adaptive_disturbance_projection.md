# GART Adaptive Certified-Disturbance Projection

## Summary

Implemented a proof-friendly adaptive projection for the certified disturbance
used by GART target selection. The change addresses disturbed-run behavior
where raw observer jumps in $\hat d$ caused repeated capped motion in $d^c$,
moving $y_s=Cx_s+d^c$ after the output had already settled near the setpoint.

## Technical Changes

- Added adaptive disturbance-rate fields to `CertifiedDisturbanceConfig`.
- Updated the certified disturbance law to project the low-pass candidate onto:

$$
\mathcal D\cap \mathcal B_{\gamma(k)\Delta d_{\max}}(d^c_{k-1}).
$$

- Added diagnostics for raw disturbance gap, effective rate cap, and adaptive
  scale.
- Added the adaptive raw GART case as the active root-runner case:
  `gart_target_raw_dxabs0p05_adaptive0p25_min0p10_headroom_0p01_dy2_no_umid`.
- Kept the fixed asymmetric `[1.0, 0.5]` disturbance-rate case available but
  disabled.
- Preserved the long-run runner settings requested during this study:
  disabled target/step/solver/time/memory caps and `THREADS = 4`.

## Validation

Planned validation:

```powershell
python -m py_compile Lyapunov/gart_target.py utils/gart_defaults.py experiments/run_gart_target_selector_study.py GARTLyapunovMPC.py utils/gart_runtime.py
pytest tests/test_gart_target.py
python GARTLyapunovMPC.py --mode disturb --n-tests 1 --set-points-len 20 --closed-loop --no-target-only --full --confirm-full --timestamp codex_smoke_adaptive_projection
```

## Notes

This intentionally avoids a heuristic settled-output freeze. The target
selector remains driven by a certified bounded-rate disturbance update, which is
more compatible with the practical-stability proof and future RL exploration.
