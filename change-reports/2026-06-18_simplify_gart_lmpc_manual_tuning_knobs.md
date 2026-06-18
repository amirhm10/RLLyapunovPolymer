# Simplify GART-LMPC Manual Tuning Knobs

Date: 2026-06-18

## Objective

Remove the `ABLATION_CASE` switch from `GARTLyapunovMPC.py` and leave the GART-LMPC target-selector tuning as direct editable constants.

## Change

The runner now exposes the current values directly:

```python
DX_S_MAX_ABS = 0.05
DU_S_MAX_ABS = [0.2, 0.2]
DY_S_MAX_ABS = 0.25
D_RATE_SCALE = 0.25
ALPHA_D = 0.05
INPUT_HEADROOM_FRAC = 0.03
PRIMARY_TOL_REL = 1.0e-4
W_U_SMOOTH_DIAG = [2.0, 2.0]
TARGET_WY_DIAG = [1.0, 1.0]
```

Original active values are documented beside the constants so manual one-at-a-time ablations can be done by direct editing.

## Rationale

Manual edits are simpler for quick experiment iteration and avoid accidentally running a hidden ablation case. The saved run summary still records the resolved `target_overrides`, which is the authoritative configuration for each result.
