# Offset-Free MPC Setpoint Search Runner

## Objective

Add a fast held-setpoint screening mode to `OffsetFreeMPC_DisturbanceRunner.py`.
The goal is to search for candidate setpoints that are worth testing later in
the slower GART-LMPC and online TD3 comparisons, without changing the shared
Lyapunov/controller configuration.

## Changes

- Added `RUN_SETPOINT_SEARCH = True` as the default mode for the runner.
- Preserved the existing Phase-2 feasibility probe behind the same
  `run_configured_study()` entrypoint when `RUN_SETPOINT_SEARCH = False`.
- Added editable setpoint-search constants:
  - `SEARCH_SETPOINTS_Y_PHYS`
  - `SEARCH_EPISODES`
  - `SEARCH_TAIL_STEPS`
  - `SEARCH_OUTPUT_ROOT`
- Each candidate is run as a held physical setpoint over the same explicit
  disturbance-ramp profile shape.
- Each candidate now gets its own result folder to avoid Windows/OneDrive
  hard-link conflicts in the mirrored summary files.
- Added summary ranking metrics from the saved diagnostic arrays:
  - diagnostic unsafe and unstable counts
  - contraction-margin maximum
  - tail input motion
  - tail target-input motion
  - tail target-output motion
  - tail input sign changes
  - physical output RMSE
  - input min/max ranges

## Initial Search Result

Command:

```powershell
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe .\OffsetFreeMPC_DisturbanceRunner.py
```

Summary:

`results/OffsetFreeMPC_SetpointSearch/20260623_135923_setpoint_search/setpoint_search_summary.csv`

Top safety-stress candidates:

| Candidate | Unsafe count | Unsafe rate | Tail input motion | Sign changes |
| --- | ---: | ---: | ---: | ---: |
| `(3.35, 323.5)` | 1342 | 0.83875 | 0.000980 | 0 |
| `(3.30, 324.5)` | 1335 | 0.83438 | 0.000923 | 0 |
| `(3.20, 324.5)` | 1321 | 0.82563 | 0.000835 | 0 |
| `(3.10, 323.0)` | 1308 | 0.81750 | 0.000749 | 0 |

Top chattering-style candidates:

| Candidate | Unsafe count | Tail input motion | Sign changes | Note |
| --- | ---: | ---: | ---: | --- |
| `(4.00, 320.5)` | 0 | 0.0000448 | 26 | Safe but saturated/high offset |
| `(4.60, 321.0)` | 0 | 0.0000552 | 24 | Safe but saturated/high offset |
| `(4.40, 321.5)` | 0 | 0.0000534 | 18 | Safe but saturated/high offset |

Interpretation:

- The first pass did **not** find a clean large-amplitude oscillatory case.
- It did find strong fixed-config safety-stress setpoints around low
  viscosity/high temperature.
- It also found high-viscosity/low-temperature candidates with repeated input
  sign changes, but the scaled tail input motion is small and the diagnostic
  Lyapunov check stays safe. These are better described as mild chattering or
  saturation/offset cases, not strong oscillation.

## Validation

- Compiled:

```powershell
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe -m py_compile .\OffsetFreeMPC_DisturbanceRunner.py
```

- Ran a two-candidate smoke search with one 800-step hold.
- Ran the full 10-candidate search with two 800-step holds per candidate.

## Next Step

Use `(3.35, 323.5)` or `(3.30, 324.5)` as the next fixed-config safety-stress
scenario candidate for GART-LMPC and RL-with/without-gate evaluation. If the
priority remains a stronger oscillation example, run a narrower search around
the high-viscosity/low-temperature region with longer holds and rank primarily
by tail sign changes and physical input saturation, not by diagnostic unsafe
count.
