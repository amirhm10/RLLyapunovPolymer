# Cold No-Gate Fast Path

## Objective

Prevent `OnlineTD3_ColdStart_NoSafetyGate.py` from spending the no-safety run inside GART target-selection CVXPY diagnostics.

The observed traceback ended in a `KeyboardInterrupt` while CVXPY was canonicalizing a GART contraction-probe problem. This was not a TD3 exception. The cold no-gate runner was configured to use the `mpc_only_diagnostic` backend and GART-LMPC teacher phases, so it still invoked expensive GART target solves even though no safety intervention was applied.

## Change

- Added no-safety aliases in `utils/online_disturbance_runner.py`:
  - `legacy`
  - `legacy_augstate`
  - `none`
  - `no_diagnostic`
- These aliases resolve to the existing `legacy_augstate` bypass path and are rejected for active safety-gate presets.
- Updated `OnlineTD3_ColdStart_NoSafetyGate.py`:
  - `PROJECTION_BACKEND = "no_diagnostic"`
  - removed the GART noisy-teacher warmup import
  - set warmup, teacher-BC, and handoff episodes to zero
  - set teacher metadata to `policy`
  - kept full-RL exploration in input-deviation space

## Method Interpretation

The cold no-gate runner now applies the TD3 action directly after bounds clipping:

$$
u_k = \mathrm{clip}(u_{\mathrm{TD3},k}, u_{\min}, u_{\max}).
$$

It no longer computes diagnostic GART targets:

$$
(x_s, u_s, y_s)
$$

for the no-safety case. That means this fast cold runner will not produce would-be fallback or diagnostic Lyapunov rejection counts. It is a pure no-safety online TD3 run.

The safety-gate runner is unchanged and still uses GART-LMPC fallback logic.

## Validation

Passed syntax validation:

```powershell
$env:PYTHONPYCACHEPREFIX = Join-Path $env:TEMP 'codex-pycache-lyapunov-polymer'
& "C:\Users\HAMEDI\miniconda3\envs\rl\python.exe" -m py_compile "OnlineTD3_ColdStart_NoSafetyGate.py" "utils\online_disturbance_runner.py" "Simulation\run_rl_lyapunov.py"
```

Passed an import-level configuration check confirming:

- `PROJECTION_BACKEND = "no_diagnostic"` resolves to `legacy_augstate`
- warmup steps are zero
- teacher-BC steps are zero
- handoff steps are zero
- full RL starts immediately
- full-RL exploration remains in input-deviation space

## Caveat

This makes the cold no-safety runner faster and cleaner, but it also removes no-gate diagnostic Lyapunov counts from this run. Use the safety-gate runner or a dedicated diagnostic run when would-be fallback statistics are needed.
