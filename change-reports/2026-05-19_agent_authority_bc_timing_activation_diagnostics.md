# Agent-Authority BC, Timing, And Activation Diagnostics

## Summary

Implemented the corrected behavioral-cloning workflow for the direct Lyapunov safety-gate RL studies. The actor now proposes the candidate action during BC, while direct LMPC is computed separately as the teacher target for the actor demo buffer.

## Changes

- Added `policy_with_lmpc_teacher_demo` as a training-phase BC mode in `Simulation/run_rl_lyapunov.py`.
- Added separate replay and demo actions so critic transitions store the executed safe action, while actor BC stores the LMPC teacher action.
- Added a 5-episode linear post-BC handoff mode that blends from LMPC teacher candidate to actor candidate before the safety gate.
- Updated cold-start and pretrained RL scripts to use the new BC mode, preserve current exploration/noise settings, save trained agents, and record wall-clock timing.
- Converted `DirectLyapunovMPC_FourMethodDisturbance.ipynb` to `DirectLyapunovMPC_FourMethodDisturbance.py` and archived the cleared notebook.
- Added wall-clock timing metadata to direct Lyapunov MPC comparison records.
- Added activation/contraction-count diagnostics with raw per-episode counts and 10-episode moving averages.
- Updated MPC-only fallback-count comparison behavior so MPC-only uses diagnostic would-be gate activation counts instead of actual fallback zero.

## Validation

- `python -m py_compile` on touched Python modules and scripts.
- Validate the archived notebook parses as JSON.
- Run a small synthetic training-phase check for the new BC mode.
- Do not run full RL training or long rollouts.

## Notes

Existing saved result folders do not contain the new timing fields. Runtime claims should use reruns generated after this instrumentation.
