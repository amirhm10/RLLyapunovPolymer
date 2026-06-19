# Standard Cold-Gate State and Input-Space Teacher Noise

## Objective

Align the cold-start safety-gate online TD3 runner with the other active online
runners and make the noisy GART-LMPC teacher phase comparable to the
`GARTLyapunovMPC_ExplorationProbe.py` input-excitation experiment.

## Changes

- Set `OnlineTD3_ColdStart_SafetyGate.py` back to `RL_OBSERVATION_MODE =
  "standard"`.
- Added `warmup_exploration_space` and `bc_exploration_space` support in the
  online training phase config.
- Set the noisy teacher templates to use `input_dev` exploration during
  teacher-driven warmup/critic-only phases.
- Updated `Simulation/run_rl_lyapunov.py` so teacher-controller Gaussian noise
  can be applied directly to scaled-deviation input `u_dev`, then mapped back to
  TD3 action coordinates for replay/logging.
- Added step-table diagnostics for the resolved exploration space and
  teacher-input exploration sigma.

## Interpretation

Before this change, the GART exploration probe and online noisy-teacher phase
were not identical:

- The probe used `u_exec = clip(u_gart + epsilon_u)` with `epsilon_u` in
  scaled-deviation input units.
- The online runner used `a_exec = clip(a_gart + epsilon_a)` in normalized TD3
  action units, then mapped `a_exec` to `u_dev`.

The online noisy teacher phase now matches the probe's excitation location for
teacher actions:

$$
u_\mathrm{cand} =
\operatorname{clip}(u_\mathrm{GART} + \epsilon_u, u_{\min}, u_{\max}),
\qquad \epsilon_u \sim \mathcal{N}(0, \sigma_u^2 I).
$$

Safety-gate runners may still differ from the probe because the active gate can
reject the noisy candidate and apply the GART-LMPC fallback. No-safety runners
and diagnostic-only runs execute the noisy teacher candidate directly, apart
from input-bound clipping.

## Validation

- `py_compile` passed with `rl-env` for:
  - `OnlineTD3_ColdStart_SafetyGate.py`
  - `utils/online_disturbance_runner.py`
  - `Simulation/run_rl_lyapunov.py`
  - `Lyapunov/safety_debug.py`
- A 1-episode cold-start safety-gate smoke run passed with:
  - `EPISODES = 1`
  - `SET_POINTS_LEN = 2`
  - `SAVE_PLOTS = False`
  - `TIMESTAMP = "diagnostic_standard_state_input_noise_cold_gate_v2"`
- Step-table readback confirmed:
  - `policy_phase = behavior_clone_teacher`
  - `training_update_mode = critic_td_only`
  - `behavior_exploration_space = input_dev`
  - `resolved_behavior_exploration_space = input_dev`
  - `behavior_exploration_sigma = 0.005`
  - `teacher_input_exploration_sigma = 0.005`
  - `correction_mode = accepted_candidate`

## Notes

The smoke artifacts were written under ignored `results/` paths and were not
staged.
