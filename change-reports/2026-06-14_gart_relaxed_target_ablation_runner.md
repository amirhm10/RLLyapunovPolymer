# GART Relaxed Target Ablation Runner

Date: 2026-06-14

## Summary

Updated the GART-LMPC runner and target selector to test whether the conservative target behavior is caused by over-filtering rather than true physical infeasibility.

## Code Changes

- Disabled the old governed-reference case in the root `GARTLyapunovMPC.py` runner.
- Enabled two GART raw closed-loop cases by default:
  - `gart_target_raw_no_dx_headroom_0p01_dy2`
  - `gart_target_raw_no_dx_headroom_0p01_dy4`
- Applied relaxed target-selector overrides to those cases:
  - `disable_dx_rate = True`
  - `input_headroom_frac = 0.01`
  - `dy_rate_scale = 2.0` or `4.0`
- Kept mixed objective cases disabled.
- Kept contraction probe required for real closed-loop cases.
- Added a disabled diagnostic-only case with `contraction_probe_log_only=True`.
- Increased the root runner guard limits to allow two full GART closed-loop cases.
- Added `dy_rate_scale = 4.0` to the target-ablation matrix.
- Changed the target-ablation replay default from old governed-reference to the relaxed GART raw source.
- Applied the relaxed dy2 target override to standalone target-only diagnostics by default.
- Updated standalone closed-loop guard estimation for two default GART cases.

## Target Tie-Breaker Change

The GART target selector stage-2 tie-breaker previously smoothed input targets against the previous accepted steady target `u_s`.

It now accepts an optional `u_smooth_ref` and, in real closed-loop runs, smooths against the previous applied input in scaled-deviation coordinates. If no applied-input reference is available, it falls back to the previous accepted `u_s`.

The diagnostic field `stage2_u_smooth_source` is now logged so runs can verify whether stage 2 used:

- `previous_applied_input`
- `previous_target_u_s`

## Validation

Passed:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe -m py_compile Lyapunov/gart_target.py experiments/run_gart_target_selector_study.py GARTLyapunovMPC.py
```

Passed short target-only smoke:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe GARTLyapunovMPC.py --target-only --no-closed-loop --mode nominal --n-tests 1 --set-points-len 20 --timestamp codex_smoke_relaxed_target --max-target-evals 100 --max-closed-loop-steps 10 --max-solver-calls 10 --max-wall-clock-seconds 120
```

Passed short closed-loop smoke:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe GARTLyapunovMPC.py --closed-loop --no-target-only --mode nominal --n-tests 1 --set-points-len 5 --timestamp codex_smoke_relaxed_closed_loop_v2 --max-target-evals 40 --max-closed-loop-steps 40 --max-solver-calls 40 --max-wall-clock-seconds 180
```

The closed-loop smoke generated both relaxed GART cases and did not run the old governed-reference case. The saved step CSV confirmed `stage2_u_smooth_source = previous_applied_input` and `input_headroom_frac = 0.01`.

Blocked:

```powershell
& C:\Users\hamed\miniconda3\envs\rlenv\python.exe -m pytest tests/test_gart_target.py
```

because `pytest` is not installed in the `rlenv` environment.
