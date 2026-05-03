## Summary

Changed the direct safety-gate RL phase-noise override so it now follows the TD3 agent's configured exploration schedule instead of using a separate linear interpolation.

## Motivation

The phased runner was correctly applying small exploration during warmup and teacher behavioral-cloning phases, but the override path was using a linear decay from `0.02` to `0.0`. The notebooks already configured the TD3 agent itself with:

- `STD_START = 0.02`
- `STD_END = 0.0`
- `STD_DECAY_MODE = "exp"`
- `STD_DECAY_RATE = 0.99992`

That meant policy exploration and teacher exploration were not aligned.

## Change

Updated [Simulation/run_rl_lyapunov.py](/c:/Users/HAMEDI/Desktop/Lyapunov_polymer/Simulation/run_rl_lyapunov.py) so:

- `training_phase_config` now supports `exploration_decay_mode`
- the default phase override mode is `agent_schedule`
- when `agent_schedule` is active, the runner reuses `agent.expl_sched.value(step_idx)`
- the old linear behavior is still available via `exploration_decay_mode="linear"`
- an explicit runner-side exponential fallback is also available via `exploration_decay_mode="exp"`

This keeps teacher-driven and policy-driven exploration on the same decay curve without requiring notebook edits.

## Validation

- `python -m py_compile Simulation/run_rl_lyapunov.py`
- schedule probe in `conda` env `rl`

Observed values with the notebook settings:

- step `0`: `0.02`
- step `8000`: `0.010545578496199018`
- step `80000`: `3.3222638925520755e-05`
- step `159199`: `5.88394982039328e-08`

So for the full direct-disturbance run, the exploration is effectively zero by the end, and the final forced test cycle still runs with exploration disabled.
