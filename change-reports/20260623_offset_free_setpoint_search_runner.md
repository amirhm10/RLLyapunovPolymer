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
- Changed the default search profile from held setpoints to two-setpoint cycles
  with `SEARCH_PROFILE_MODE = "cycle"`.
- Each cycle holds one raw setpoint for 400 samples, then switches to the next
  raw setpoint for 400 samples, repeated over the configured search episodes.
- Added editable setpoint-search constants:
  - `SEARCH_SETPOINTS_Y_PHYS`
  - `SEARCH_CYCLES_Y_PHYS`
  - `SEARCH_EPISODES`
  - `SEARCH_TAIL_STEPS`
  - `SETTLING_TAIL_STEPS`
  - `SETTLING_BAND_PHYS`
  - `SEARCH_OUTPUT_ROOT`
- Each candidate is run over the same explicit disturbance-ramp profile shape.
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
- Added cycle-specific metrics:
  - number of 400-sample blocks
  - nonsettled block count
  - tail and final normalized tracking-error norms
  - output and input sign-change counts inside cycle tails
  - tail output motion
- Added reusable tracking plots for setpoint-search outputs. The plots overlay
  raw setpoint, OF-MPC output, governed Lyapunov target, 400-sample switch
  markers, settling bands, and diagnostic unsafe-action shading.

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

## Cycle-Search Update

The follow-up search uses two-setpoint cycles instead of held setpoints because
the desired failure mode is not simply steady-state infeasibility. The scenario
is meant to stress whether the controller settles before the next reference
change.

Command:

```powershell
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe .\OffsetFreeMPC_DisturbanceRunner.py
```

Summary:

`results/OffsetFreeMPC_SetpointCycleSearch/20260623_162143_setpoint_search/setpoint_search_summary.csv`

Tracking figures:

![Top cycle candidates](../results/OffsetFreeMPC_SetpointCycleSearch/20260623_162143_setpoint_search/tracking_plots/tracking_top_cycle_candidates.png)

![Cycle 07 tracking](../results/OffsetFreeMPC_SetpointCycleSearch/20260623_162143_setpoint_search/tracking_plots/tracking_cycle_07_eta4p6_T321_to_eta3p2_T324p5.png)

For each 400-sample block, the late-block normalized tail error is computed as

$$
\bar e_{\mathrm{tail}}
=
\frac{1}{N_{\mathrm{tail}}}
\sum_{k \in \mathrm{tail}}
\left\|
\frac{y_k-y_{\mathrm{sp},k}}{b}
\right\|_{\infty},
\qquad
b = [0.05,\;0.30].
$$

A block is counted as nonsettled when this normalized tail error is above 1.
This deliberately uses a tight band, so the metric should be interpreted as
"not settled inside the chosen band" rather than open-loop instability.

Top cycle candidates:

| Candidate | Search score | Nonsettled blocks | Unsafe count | Tail error |
| --- | ---: | ---: | ---: | ---: |
| `cycle_07_eta4p6_T321_to_eta3p2_T324p5` | 63.41 | 3 | 32 | 3.059 |
| `cycle_03_eta4p6_T321_to_eta3p35_T323p5` | 60.04 | 3 | 33 | 3.087 |
| `cycle_04_eta4p4_T321p5_to_eta3p3_T324p5` | 52.28 | 3 | 75 | 1.608 |
| `cycle_08_eta4p4_T321p5_to_eta3p1_T323` | 44.50 | 3 | 0 | 1.620 |
| `cycle_09_eta4_T320p5_to_eta3p3_T324p5` | 44.36 | 3 | 65 | 1.005 |

Interpretation:

- The cycle search found a better discussion scenario than the held-setpoint
  scan. The baseline OF-MPC response repeatedly shows underdamped transitions
  and late-block errors outside the chosen settling band before the next
  400-sample switch.
- The behavior is not a clean sustained oscillation around a steady setpoint.
  It is more defensible to describe it as repeated unsettled/off-target tails
  under a cycling reference.
- `cycle_07` is the strongest combined stress case because it has high
  nonsettled tail error and diagnostic unsafe actions.
- `cycle_04` is also attractive because it has the largest unsafe count among
  the top nonsettled candidates.
- `cycle_08` isolates the tracking difficulty without unsafe diagnostics, so
  it is useful as a control-side stress case but weaker as a safety-gate
  motivation.

Recommended next experiment:

Use `cycle_07_eta4p6_T321_to_eta3p2_T324p5` first for a full fair comparison
across OF-MPC, GART-LMPC, and TD3 with and without the safety gate. Use
`cycle_04_eta4p4_T321p5_to_eta3p3_T324p5` as the second candidate if the main
discussion needs a higher unsafe-action count. Keep the shared Lyapunov and RL
configuration fixed across all methods.

Additional validation:

- Compiled with Python cache redirected away from the locked OneDrive
  `__pycache__` folder:

```powershell
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe -X pycache_prefix="$env:TEMP\codex_pycache" -m py_compile OffsetFreeMPC_DisturbanceRunner.py
```

- Generated tracking plots from the saved cycle-search bundles without
  rerunning the simulations.

## Default Reverted to Manual Held-Setpoint Search

The runner default was switched back to the earlier held-setpoint workflow so
new manual setpoint edits can be made directly in `SEARCH_SETPOINTS_Y_PHYS`.
The cycle-search constants remain in the file as an optional mode, but they are
ignored unless `SEARCH_PROFILE_MODE` is changed from `"held"` to `"cycle"`.

Current manual-search defaults:

- `SEARCH_STUDY_NAME = "OffsetFreeMPC_SetpointSearch"`
- `SEARCH_CASE_PREFIX = "setpoint"`
- `SEARCH_PROFILE_MODE = "held"`
- `SEARCH_EPISODES = 2`

The tracking plot helper now labels outputs as either setpoint candidates or
setpoint-cycle candidates depending on the summary file contents.
