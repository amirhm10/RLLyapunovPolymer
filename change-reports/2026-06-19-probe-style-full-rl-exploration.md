# Probe-Style Full-RL Exploration for Online TD3

## Objective

Extend probe-style exploration from the noisy GART-LMPC teacher phase to the
online TD3 policy phase used by the active root runners.

## Changes

- The shared noisy-teacher online templates now set:
  - `handoff_exploration_space = "input_dev"`
  - `full_rl_exploration_space = "input_dev"`
- `Simulation/run_rl_lyapunov.py` now honors `input_dev` exploration for actor
  policy behavior, not only teacher behavior.
- Policy exploration now follows:

$$
u_\mathrm{policy} = \operatorname{map}(a_\theta(s)),
\qquad
u_\mathrm{cand} =
\operatorname{clip}(u_\mathrm{policy} + \epsilon_u, u_{\min}, u_{\max}),
\qquad
a_\mathrm{used} = \operatorname{map}^{-1}(u_\mathrm{cand}).
$$

where $\epsilon_u$ is Gaussian noise in scaled-deviation input units.

## Why This Matters

Previously, full-RL exploration used normalized TD3 action noise:

$$
a_\mathrm{cand} = a_\theta(s) + \epsilon_a.
$$

After mapping to the physical controller input coordinates, this effectively
scaled the perturbation by approximately:

$$
\frac{1}{2}(u_{\max} - u_{\min}).
$$

Because the input ranges are wide, small action-space noise could become a much
larger scaled-input perturbation. The new path makes the exploration scale
directly comparable to the GART exploration probe.

## Diagnostics

The safety debug bundle and step table now include policy-side input exploration
fields:

- `policy_input_exploration_sigma`
- `policy_input_exploration_requested_store`
- `policy_input_exploration_applied_store`
- `policy_u_dev_pre_exploration_store`
- `policy_action_nominal_pre_exploration_store`

## Validation

- `py_compile` passed with `rl-env` for:
  - `utils/online_disturbance_runner.py`
  - `Simulation/run_rl_lyapunov.py`
  - `Lyapunov/safety_debug.py`
- A forced-full-RL smoke run passed on `OnlineTD3_ColdStart_SafetyGate.py` with:
  - `EPISODES = 1`
  - `SET_POINTS_LEN = 2`
  - `SAVE_PLOTS = False`
  - `behavior_clone_teacher_episodes = 0`
  - `handoff_episodes = 0`
  - `TIMESTAMP = "diagnostic_probe_style_full_rl_cold_gate"`
- Step-table readback confirmed:
  - `policy_phase = full_rl`
  - `behavior_policy_source = policy_explore`
  - `training_update_mode = td3_full`
  - `behavior_exploration_space = input_dev`
  - `resolved_behavior_exploration_space = input_dev`
  - `policy_input_exploration_sigma = 0.1`

## Notes

Safety-gate runners still may replace the candidate with the GART-LMPC fallback
if the noisy policy action fails the gate. The exploration is now probe-style at
the candidate-generation layer, not a guarantee that the noisy action is always
executed in safety-gate cases.
