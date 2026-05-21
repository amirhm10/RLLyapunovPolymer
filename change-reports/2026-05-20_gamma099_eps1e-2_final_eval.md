# Gamma 0.99, Lyapunov Epsilon 1e-2, And Final Evaluation Episode

## Summary

Prepared the next active non-evaluation run by changing the TD3 discount factor, relaxing the Lyapunov contraction epsilon, and restoring the final RL evaluation episode.

## Changes

- `DirectLyapunovSafetyGateRL_ColdStart.py`
  - Set `GAMMA = 0.99`.
  - Set `lyap_eps = 1e-2`.
  - Set `FORCE_FINAL_TEST = True`.

- `DirectLyapunovSafetyGateRL_Pretrained.py`
  - Set `GAMMA = 0.99`.
  - Set `lyap_eps = 1e-2`.
  - Set `FORCE_FINAL_TEST = True`.

- `DirectLyapunovMPC.py`
  - Set `lyap_eps = 1e-2` so the direct comparison uses the same relaxed contraction tolerance.

## Rationale

- `GAMMA = 0.99` shortens the TD3 effective planning horizon relative to `0.995`, which may make learning less sensitive to long delayed penalties and easier to stabilize.
- `lyap_eps = 1e-2` relaxes the one-step Lyapunov contraction check more than the previous `1e-3`, which should reduce gate interventions when the trajectory is already close to the Lyapunov boundary or near small changes in $V$.
- Restoring the final evaluation episode gives a consistent end-of-run test episode again, while the saved-agent evaluation script remains available for separate post-training testing.

## Parameter-Noise Note

Gaussian action noise is intentionally kept unchanged in this setup. Switching back to parameter noise should be tested as a separate experiment because changing exploration type at the same time as `GAMMA`, `lyap_eps`, and final-test behavior would confound interpretation.

## Validation

- `python -m py_compile DirectLyapunovMPC.py DirectLyapunovSafetyGateRL_ColdStart.py DirectLyapunovSafetyGateRL_Pretrained.py`
- `git diff --check`

## Notes

- Full training was not run during this change.
