# LMPC/RL 300-Episode Proof-Track Configuration

Date: 2026-06-06

## Summary

Updated the three active direct Lyapunov runners to use the current proof-track Lyapunov setting and restored the same-run `mpc_only` baseline in the two RL safety-gate runners.

## Code Changes

- `DirectLyapunovMPC.py`
  - Set `n_episodes = 300`.
  - Kept the active Lyapunov setting at `rho_lyap = 0.99`, `lyap_eps = 5e-3`.
  - Kept the direct `mpc_only` comparison active.

- `DirectLyapunovSafetyGateRL_ColdStart.py`
  - Set `rho_lyap = 0.99`.
  - Set `lyap_eps = 5e-3`.
  - Added explicit `n_episodes = 300` and set `n_tests = n_episodes`.
  - Restored the `mpc_only` governed-reference diagnostic case.

- `DirectLyapunovSafetyGateRL_Pretrained.py`
  - Set `rho_lyap = 0.99`.
  - Set `lyap_eps = 5e-3`.
  - Added explicit `n_episodes = 300` and set `n_tests = n_episodes`.
  - Restored the `mpc_only` governed-reference diagnostic case.

## Report Added

Added:

```text
report/lyapunov_stability_proof_track_2026-06-06.md
```

The report records the stability-proof discussion:

- fixed positive `lyap_eps` supports practical stability, not asymptotic convergence to zero
- `rho = 0.99`, `epsilon = 5e-3` gives an ultimate Lyapunov-value bound of `0.5`
- asymptotic convergence requires either `epsilon = 0` or a vanishing sequence `epsilon_k -> 0`
- moving setpoints and changing disturbances require additive terms for target motion, disturbance-estimate motion, and model mismatch
- the governed-reference target calculation strengthens the practical proof because it centers the Lyapunov function on a feasible model equilibrium instead of an unreachable raw setpoint
- the target-selector mathematics is now written only as the feasible governed steady projection used for the proof discussion
- no immediate target-selection change is required for the fixed-epsilon practical proof, but the future vanishing-epsilon proof should account for measured target motion
- the next algorithmic step is an adaptive epsilon schedule after the fixed `5e-3` 300-episode benchmark

## Validation

Low-cost validation should be:

```powershell
python -m py_compile DirectLyapunovMPC.py DirectLyapunovSafetyGateRL_ColdStart.py DirectLyapunovSafetyGateRL_Pretrained.py
```

No long training runs were executed as part of this change.
