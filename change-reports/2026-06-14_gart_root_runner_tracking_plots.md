# GART Root Runner And Tracking Plots

## Summary

Added a root-level GART-LMPC launcher and per-case tracking plots so the nominal smoke study can be monitored like the direct Lyapunov runner.

## Changes

- Added `GARTLyapunovMPC.py` as a main-repo entrypoint for the GART smoke/comparison runner.
- Updated `experiments/run_gart_target_selector_study.py` to save direct-runner-style CSTR tracking plots under each closed-loop case directory.
- Each case now records the tracking plot directory in the closed-loop `summary.json`.

## Usage

```powershell
python GARTLyapunovMPC.py --mode nominal --n-tests 5 --set-points-len 20 --target-only --closed-loop
```

The performance plots are written under:

```text
results/GARTLMPC/<timestamp>/<case>/tracking_plots/
```
