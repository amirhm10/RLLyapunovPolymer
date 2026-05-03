# Direct Safety-Gate RL Phase Schedule

## Summary
- Added a three-phase training schedule to the shared direct safety-gate RL runner.
- Kept the public `run_rl_train(...)` return tuple unchanged.
- Updated both direct safety-gate notebooks to use the shared schedule with notebook-specific warmup behavior.

## Implementation
- `Simulation/run_rl_lyapunov.py`
  - Added optional `training_phase_config` support.
  - Added phase resolution for:
    - warmup replay fill
    - teacher-driven behavioral cloning
    - full RL
  - Added per-step metadata in `lyap_info_storage`:
    - `policy_phase`
    - `behavior_policy_source`
    - `training_update_mode`
  - Reused the direct Lyapunov tracking controller as the teacher policy during the BC phase.
- `TD3Agent/agent.py`
  - Added shared exploration application so policy and teacher actions use the same schedule.
  - Added a dedicated BC demo buffer to avoid mixing pretrained warmup policy actions into BC targets.
  - Added online actor BC updates and reused critic-only TD updates during the BC phase.
- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
  - Configured:
    - episodes `1-10`: policy replay fill
    - episodes `11-30`: Lyapunov MPC teacher + BC
    - episode `31+`: full RL
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`
  - Configured:
    - episodes `1-10`: Lyapunov MPC teacher replay fill
    - episodes `11-30`: Lyapunov MPC teacher + BC
    - episode `31+`: full RL

## Validation
- `python -m py_compile TD3Agent/agent.py Simulation/run_rl_lyapunov.py`
- Notebook JSON round-trip validation after the config edits.
- Reduced smoke runs in the `rl` conda environment with:
  - smaller `set_points_len`
  - smaller `batch_size`
  - one direct-gate case per notebook
- Smoke-run checks confirmed:
  - no learning in episodes `1-10`
  - teacher-driven BC phase in episodes `11-30`
  - BC actor updates during the BC phase
  - full-RL update mode beginning after episode `30`
  - notebook-specific warmup behavior differences between pretrained and cold-start

## Assumptions
- One episode is one full two-setpoint cycle.
- BC targets are the final applied safe teacher actions stored during teacher-driven phases.
- The phase-specific exploration override decays linearly from `0.02` to `0.0` over the full run.
