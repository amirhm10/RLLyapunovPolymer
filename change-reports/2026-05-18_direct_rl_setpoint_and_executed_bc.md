# Direct RL setpoint length and executed-action BC update

Date: 2026-05-18

## Objective

Apply two direct Lyapunov study configuration fixes:

- Use 400 samples per setpoint segment instead of 800.
- Use TD3 discount factor gamma 0.99 instead of 0.995 in the direct safety-gate RL notebooks.
- For the pretrained RL notebook, make the behavior-cloning phase train on the final executed safety-gate action instead of forcing LMPC teacher behavior.

## Changes

- `utils/direct_lyapunov_study.py`
  - Set `DIRECT_DISTURBANCE_SETPOINT_LEN = 400`.
- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
  - Keeps `set_points_len = DIRECT_DISTURBANCE_SETPOINT_LEN`, now 400.
  - Sets `GAMMA = 0.99`.
  - Sets `bc_teacher_policy` and `bc_behavior_source` to `executed_action`.
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`
  - Keeps the LMPC teacher behavior for cold-start cloning.
  - Sets `GAMMA = 0.99`.
- `Simulation/run_rl_lyapunov.py`
  - Adds `bc_behavior_source` support for `direct_lyapunov_mpc`, `policy`, and `executed_action`.
  - In the behavior-clone phase, `executed_action` uses policy behavior through the safety gate, then stores the final executed action in the actor-demo buffer.

## Validation

- Notebook JSON validation passed for the direct MPC, frozen-output disturbance, cold-start RL, and pretrained RL notebooks.
- Syntax validation passed for `Simulation/run_rl_lyapunov.py` and `utils/direct_lyapunov_study.py`.
- A phase-config smoke check confirmed that `executed_action` BC disables LMPC teacher behavior, pushes demos, and labels the behavior source as `executed_action_gaussian`.
- Search confirmed no remaining `800` or `0.995` literals in Python or notebook files.

## Notes

No full experiment rerun was performed. The repository already had pre-existing uncommitted notebook and controller changes, so this report records the local fix without claiming a clean baseline comparison.
