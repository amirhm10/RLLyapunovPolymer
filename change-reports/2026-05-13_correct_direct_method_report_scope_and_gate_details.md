# Correct Direct Method Report Scope And Gate Details

Date: 2026-05-13

## Summary

Corrected the direct-method step-by-step report after a scope mismatch was identified.

The main fixes are:

1. clarified that the original note was reconstructed from:
   - `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
   - `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
   - `DirectLyapunovSafetyGateRL_ColdStart.ipynb`
   and not from `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`
2. corrected the RL gate description to state that the current direct-gate notebooks do not activate hard `delta_u` bounds because `du_min` and `du_max` are not passed into `run_rl_train(...)`
3. corrected the direct MPC description to state that, for the active notebook settings with
   `objective_steady_input_cost=False` and `objective_terminal_cost=False`, the hard-mode objective reduces to output tracking plus `Rdu` move penalty, while the Lyapunov term is enforced as a first-step constraint rather than an objective penalty

## Files updated

- [report/direct_lyapunov_method_step_by_step_2026-05-13.md](../report/direct_lyapunov_method_step_by_step_2026-05-13.md)

## Validation

- Checked `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` for:
  - `rho_lyap = 0.98`
  - `objective_steady_input_cost = False`
  - `objective_terminal_cost = False`
  - `first_step_contraction_on = True`
- Checked the RL notebooks to confirm that `run_rl_train(...)` is called without `du_min` and `du_max`.
- Checked `Lyapunov/direct_lyapunov_mpc.py` to confirm that hard mode zeros the steady-input and terminal-cost objective terms when those flags are false.
