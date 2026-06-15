# Select GART dx5 Case And Prepare eps=1e-4 Run

## Summary

Narrowed the editable GART runner to the raw dx5 case and changed GART Lyapunov epsilon defaults to `1e-4` for the next diagnostic run.

## Result Basis

Analyzed `results/GARTLMPC/20260614_204326`, which compared:

- `gart_target_raw_dx5_headroom_0p01_dy2_no_umid`
- `gart_target_raw_dx10_headroom_0p01_dy2_no_umid`
- `gart_target_raw_dx20_headroom_0p01_dy2_no_umid`

The cases were essentially identical in reward, RMSE, hard contraction success, and governor activity. The dx5 case was selected because it is the tightest finite $x_s$ rate bound among the tested settings without a performance penalty.

## Code Changes

- Removed dx10 and dx20 from the default GART closed-loop case list.
- Kept `dx_rate_scale = 5`, `dy_rate_scale = 2`, `input_headroom_frac = 0.01`, no `u_mid` tie-breaker, no $x_s$ smoothing, and no $y_s$ smoothing as the active raw GART case.
- Set `eps = 1e-4` in:
  - `Lyapunov/gart_target.py`
  - `Lyapunov/gart_lmpc.py`
  - `utils/gart_defaults.py`
  - `experiments/run_gart_target_selector_study.py`
- Reduced default closed-loop resource limits back to one active case.
- Added `report/gart_dx_rate_selection_2026-06-14.md`.

## Notes

The analyzed dx-scale run itself used `eps = 1e-3`. The code is now prepared for the user's next `eps = 1e-4` run.
