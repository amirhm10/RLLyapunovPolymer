# Two-Phase Online TD3/GART Study Runner

Date: 2026-06-22

## Objective

Add a clean two-phase experiment runner for the polymer CSTR online TD3/GART study. The runner supports paired multi-seed execution, explicit Phase-1/Phase-2 setpoint and disturbance profiles, compact per-method artifacts, phase-aware diagnostics, timing summaries, and a GART-LMPC-only baseline.

## Main Changes

- Added `RunOnlineTD3TwoPhaseStudy.py` as the new root batch runner.
- Added `utils/two_phase_profiles.py` for explicit setpoint, disturbance, phase-index, and profile validation helpers.
- Archived the four single-profile online TD3 root runners under `archive/online_td3_single_profile_20260622/`.
- Extended the online TD3 training path to accept explicit setpoint and disturbance profiles.
- Added exploration decay horizon support through `exploration_decay_end_step`.
- Added per-step phase, setpoint, disturbance, exploration, device, and wall-clock timing diagnostics.
- Added compact `phase_table.csv`, phase-window plots, disturbance-profile plots, timing plots, and seed-level comparison plots.
- Added GART-LMPC two-phase support through the existing GART closed-loop path, including matching compact direct-style exports.
- Avoided mirroring large `step_table.csv` and `arrays.npz` into method roots by default.

## Default Experiment Shape

- Phase 1: 200 episodes.
- Phase 2: 50 episodes.
- Episode length: `2 * set_points_len`, default `800` steps.
- Setpoints:
  - Phase 1: `[[4.5, 324.0], [3.4, 321.0]]`
  - Phase 2: `[[4.4, 321.5], [3.3, 324.5]]`
- Disturbance:
  - Phase 1 ramps from nominal to `Qi=102.6`, `Qs=481.95`, `hA=966000.0`.
  - Phase 2 ramps from the Phase-1 endpoint to `Qi=113.4`, `Qs=436.05`, `hA=924000.0`.
- Exploration:
  - Pretrained: `0.02 -> 0.005` by the end of Phase 1, then fixed.
  - Cold-start: `0.10 -> 0.005` by the end of Phase 1, then fixed.

## Validation

- `python -m py_compile` on:
  - `RunOnlineTD3TwoPhaseStudy.py`
  - `utils/two_phase_profiles.py`
  - `utils/helpers.py`
  - `Simulation/run_rl_lyapunov.py`
  - `utils/online_disturbance_runner.py`
  - `Lyapunov/safety_debug.py`
  - `Lyapunov/direct_lyapunov_mpc.py`
  - `experiments/run_gart_target_selector_study.py`
- Profile checks:
  - Default total steps: `250 * 2 * 400 = 200000`.
  - Tiny smoke total steps: `2 * 2 * 5 = 20`.
  - Setpoint switch occurs at the first Phase-2 step.
  - Disturbance start, Phase-1 endpoint, and Phase-2 endpoint match the design.
  - Exploration reaches `0.005` at the Phase-1 endpoint and remains fixed afterward.
- Smoke runs:
  - Tiny TD3 compact run without plots:
    `--n-seeds 1 --methods cold_start_no_safety_gate --phase1-episodes 1 --phase2-episodes 1 --set-points-len 5 --no-save-plots`
  - Tiny GART compact run without plots:
    `--n-seeds 1 --methods gart_lmpc --phase1-episodes 1 --phase2-episodes 1 --set-points-len 5 --no-save-plots`
  - Tiny TD3 compact run with plots enabled, including phase-window, disturbance, timing, and seed-comparison plots.

## Notes

- The default output root is `Path.home() / "Desktop" / "Lyapunov_polymer_results"`.
- Validation smoke outputs were written under ignored `results/_two_phase_smoke/`.
- The default pretrained checkpoint is pinned to `results/PretrainOFMPC/20260621_203346/of_mpc_pretrained_td3_20260622_030149.pkl`.
