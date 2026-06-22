# Phase 2 Rolling Reward Plots

Date: 2026-06-22

## Objective

Add a Phase 2 reward plot that matches the fixed-duration robustness interpretation of the two-phase experiment. Phase 2 is a continuous 10000-step continuation segment, so a step-level `reward_no_penalty` trace with rolling means is more informative than treating 400-step report windows as full episodes.

## Changes

- Added Phase 2 window selection for safety-filter/TD3 debug plots.
- Added `plots/ph/phase2_reward_no_penalty_rolling.png` for TD3 safety-filter bundles.
- Added the same `plots/ph/phase2_reward_no_penalty_rolling.png` for direct/GART-LMPC debug bundles.
- The plot shows:
  - raw step-level `reward_no_penalty`
  - 100-step rolling mean
  - 400-step rolling mean
  - full Phase 2 mean

## Notes

- This is a plotting/export change only. It does not change controllers, rewards, replay buffers, seeds, or saved numeric arrays.
- For GART-LMPC, `reward_no_penalty` equals the baseline reward because no RL fallback/event penalty is used.

## Validation

- Compiled touched modules with bytecode cache redirected away from OneDrive:

```powershell
$env:PYTHONPYCACHEPREFIX="$env:TEMP\codex_pycache"
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe -m py_compile Lyapunov\safety_debug.py Lyapunov\direct_lyapunov_mpc.py
```

- Checked that both TD3 and direct plot helpers select the `phase2_full` window from the current two-phase profile windows.
