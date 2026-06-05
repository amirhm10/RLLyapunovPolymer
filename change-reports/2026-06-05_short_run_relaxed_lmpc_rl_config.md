# Short Run Relaxed LMPC/RL Config

Date: 2026-06-05

## Summary

Updated the active direct Lyapunov and safety-gate RL runners for a shorter analysis-oriented experiment with relaxed Lyapunov epsilon and no governed-reference target anchoring in the RL cases.

## Changes

- Set `lyap_eps = 1e-5` in:
  - `DirectLyapunovMPC.py`
  - `DirectLyapunovSafetyGateRL_ColdStart.py`
  - `DirectLyapunovSafetyGateRL_Pretrained.py`
- Kept `rho_lyap = 0.98` in all three active runners.
- Set the RL governed-reference target weights to zero:
  - `u_prev_penalty_weight = 0.0`
  - `xs_prev_penalty_weight = 0.0`
- Removed the `mpc_only` case from the ColdStart and Pretrained RL runners only.
- Kept Direct Lyapunov MPC's own `mpc_only` diagnostic case active.
- Reduced the Direct Lyapunov MPC run length from 300 episodes to 10 episodes.
- Archived `DirectLyapunovSavedAgentEvaluation.py` under `archive/DirectLyapunovSavedAgentEvaluation.py`.

## Exploration Note

The active RL runners use Gaussian behavior noise in the BC and full-RL phases:

```python
"bc_behavior_noise": "gaussian"
"full_rl_behavior_noise": "gaussian"
```

Warmup behavior noise remains disabled:

```python
"warmup_behavior_noise": "none"
```

## Validation

- To run: `python -m py_compile` on the three active runners and the archived evaluation runner.
- Full rollout validation was not run because these files launch long experiments.

## Notes

The direct Lyapunov MPC target regularization weights were left unchanged at `0.1`; only the Lyapunov epsilon and run length were changed there.
