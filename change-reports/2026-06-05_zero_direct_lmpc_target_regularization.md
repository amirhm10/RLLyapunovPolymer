# Zero Direct LMPC Target Regularization

Date: 2026-06-05

## Summary

Aligned the direct Lyapunov MPC runner with the latest RL runner target-regularization experiment by removing previous-input and previous-target-state anchoring from the governed-reference target configuration.

## Changes

- Updated `DirectLyapunovMPC.py`:
  - `u_prev_penalty_weight = 0.0`
  - `xs_prev_penalty_weight = 0.0`

## Validation

- To run: `python -m py_compile DirectLyapunovMPC.py`.

## Notes

The short-run Lyapunov settings remain unchanged:

```python
rho_lyap = 0.98
lyap_eps = 1e-5
n_episodes = 10
```
