# RL Phase Cleanup And Direct Scope Narrowing

Date: 2026-05-13

## Summary

Implemented the requested RL/direct experiment cleanup without modifying the generic standard-Lyapunov path.

Main changes:

- removed the RL warmup buffer-only phase from the pretrained and cold-start direct-gate notebooks by setting `WARMUP_EPISODES = 0`
- strengthened behavioral cloning by running 4 actor-BC updates per environment step during the BC phase
- replaced the current RL notebook full-RL parameter-noise exploration with Gaussian exploration
- set BC Gaussian exploration to `0.005`
- set full-RL Gaussian exploration to start at `0.02` and decay linearly to `0.0` by the final step
- changed the shared direct disturbance setpoint length from `400` to `800`
- reduced the selector comparison set to only no anchoring and mixed anchoring
- fixed selector case naming so exported case names and labels now reflect the actual anchor weights, for example `0.2 -> 0p2` and `0.25 -> 0p25`

## Runtime changes

Updated [Simulation/run_rl_lyapunov.py](../Simulation/run_rl_lyapunov.py):

- `training_phase_config` now accepts:
  - `bc_actor_updates_per_step`
  - `bc_exploration_std`
  - `full_rl_exploration_std_start`
  - `full_rl_exploration_std_end`
  - `full_rl_exploration_decay_mode`
- `_phase_exploration_sigma(...)` now supports phase-aware Gaussian noise:
  - fixed BC sigma
  - full-RL phase-local decay starting at `bc_end_step`
- `_apply_agent_training_updates(...)` now repeats `train_actor_bc_step()` according to `bc_actor_updates_per_step`

## Shared direct-study changes

Updated [utils/direct_lyapunov_study.py](../utils/direct_lyapunov_study.py):

- `DIRECT_DISTURBANCE_SETPOINT_LEN = 800`
- `direct_four_method_case_specs(...)` now supports an optional `variants` subset selector
- case names and labels are now generated dynamically from the actual `anchor_weight` and `smoothness_weight`

The active notebooks now request only:

- `none`
- `mixed`

## Notebook changes

Updated:

- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](../DirectLyapunovSafetyGateRL_ColdStart.ipynb)
- [DirectLyapunovMPC_FourMethodDisturbance.ipynb](../DirectLyapunovMPC_FourMethodDisturbance.ipynb)

Notebook-side cleanup:

- removed active parameter-noise config from the two RL notebooks
- stored only BC-end phase boundaries in neutral `phase_plot_boundaries` metadata
- switched the three notebooks to the two-case selector scope

## Plot/debug compatibility

Updated:

- [Lyapunov/safety_debug.py](../Lyapunov/safety_debug.py)
- [Plotting_fns/rl_plots.py](../Plotting_fns/rl_plots.py)

These now accept the neutral `phase_plot_boundaries` key while preserving compatibility with existing `warm_start_plot`-based bundles.

## Validation

- `python -m py_compile Simulation/run_rl_lyapunov.py utils/direct_lyapunov_study.py Lyapunov/safety_debug.py Plotting_fns/rl_plots.py`
- parsed the edited notebooks as JSON with `ConvertFrom-Json`
- ran source-level helper checks confirming:
  - `DIRECT_DISTURBANCE_SETPOINT_LEN == 800`
  - `direct_four_method_case_specs(..., variants=("none","mixed"))` yields exactly 2 cases in order
  - BC phase resolves to teacher + Gaussian noise + `sigma = 0.005`
  - first full-RL step resolves to policy + Gaussian noise + `sigma = 0.02`
  - final full-RL step resolves to `sigma = 0.0`
  - BC phase carries `bc_actor_updates_per_step = 4`
