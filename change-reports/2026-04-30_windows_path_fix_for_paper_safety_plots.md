# Windows Path Fix For Paper Safety Plots

## Summary

Fixed a Windows path-length failure in `Lyapunov/safety_debug.py` when saving paper-style safety-filter plots.

## Problem

For studies with long export roots such as:

- `Data/debug_exports/mpc_selector_term_ablation_direct_setup/...`

the paper plot exporter created nested directories like:

- `paper_plots/safety_selector/state_target_channels/full_xhat_vs_xs`

That pushed the full directory and filename path beyond the Windows limit and caused:

```text
FileNotFoundError: [WinError 206] The filename or extension is too long
```

## Change

- Kept the existing outer export directory structure unchanged.
- In paper-plot mode only, shortened the internal state/target plot subdirectories:
  - `state_target_channels` -> `st`
  - `full_xhat_vs_xs` -> `fx_xs`
  - `full_dhat_vs_ds` -> `fd_ds`
  - `full_ys_vs_ysp` -> `fy_ysp`
  - `full_rs_vs_ysp` -> `fr_ysp`
  - `full_ys_decomposition` -> `fy_dec`
  - `last_episode_xhat_vs_xs` -> `lx_xs`
  - `last_episode_dhat_vs_ds` -> `ld_ds`
  - `last_episode_ys_vs_ysp` -> `ly_ysp`
  - `last_episode_rs_vs_ysp` -> `lr_ysp`
  - `last_episode_ys_decomposition` -> `ly_dec`

## Effect

- The matched safety-filter notebook can save paper-style debug figures on Windows without hitting the path-length limit.
- Non-paper plot exports keep their original descriptive directory names.

## Validation

- Syntax-only compile of `Lyapunov/safety_debug.py` succeeded.
- Checked the previously failing path pattern and reduced it to about 243 characters including the figure filename.
