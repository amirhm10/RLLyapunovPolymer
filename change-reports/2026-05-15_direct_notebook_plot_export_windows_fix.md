## Summary

Fixed two direct-notebook plotting/export regressions:

- the RL safety-gate debug exporter could still choose a Windows path that later became too long inside nested plot folders
- the frozen-output direct-MPC notebook still passed `save_paper_plots=...`, but the current `save_direct_lyapunov_debug_artifacts(...)` signature no longer accepted that keyword

## Changes

Updated `Lyapunov/safety_debug.py` so the Windows path budget now reflects the deepest nested direct-safety plot paths, not just the top-level plot files.

The exporter now:

- projects path length using the real nested `state_target_channels/.../ys_decomposition_0.png` style paths
- automatically switches to shorter nested directory and filename aliases when a long Windows output path would overflow the soft limit
- restores the per-episode output-versus-setpoint views under `episode_samples_by_tens/` and `last_episode_summary/`
- shortens comparison-plot filenames on Windows when the study root is already long

Updated `Lyapunov/direct_lyapunov_mpc.py` so the direct-MPC exporter is notebook-compatible again and safer on long Windows study roots.

The exporter now:

- accepts `save_paper_plots` again for backward compatibility with `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
- chooses a shorter per-case debug directory on Windows when `plots/` and optional `paper_plots/` would otherwise exceed the path budget
- shortens comparison-plot filenames on Windows when needed
- records the actual shortened comparison path in `plot_paths["output_rmse"]`

## Validation

Ran import/path-budget checks in the `rl-env` interpreter:

- RL safety-gate case rooted at `results/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/...`
  - selected output dir: `.../bounded_hard/20260515_135500`
  - projected max path length: `238`
  - short nested plot paths: enabled
  - restored episode-window paths also remain within budget:
    - `plots/ep_samples/ep_001_001_010.png`
    - `plots/last_ep/ep_001_last.png`
- Direct frozen-output case rooted at `Data/debug_exports/direct_lyapunov_mpc_bounded_three_scenario_two_setpoint_nominal/...`
  - selected output dir: `.../dl_e31c7b57_135500`
  - projected max path length with paper plots: `239`
- confirmed `save_direct_lyapunov_debug_artifacts` signature now includes `save_paper_plots`
- confirmed in-memory `compile(...)` succeeds for:
  - `Lyapunov/safety_debug.py`
  - `Lyapunov/direct_lyapunov_mpc.py`

Attempted `python -m py_compile` twice with fresh `pycache_prefix` directories, but both attempts hit the existing Windows `.pyc` rename permission problem in this repository, so syntax validation fell back to in-memory compilation.
