# Refine Direct Method Report Against RL Notebooks

Date: 2026-05-13

## Scope

This update re-audits the RL direct-safety-gate notebooks against the intended direct Lyapunov formulation described by the user.

Files checked:

- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](../DirectLyapunovSafetyGateRL_ColdStart.ipynb)
- [DirectLyapunovMPC_FrozenOutputDisturbance.ipynb](../DirectLyapunovMPC_FrozenOutputDisturbance.ipynb)
- [Simulation/run_rl_lyapunov.py](../Simulation/run_rl_lyapunov.py)
- [Lyapunov/direct_lyapunov_mpc.py](../Lyapunov/direct_lyapunov_mpc.py)
- [Lyapunov/lyapunov_core.py](../Lyapunov/lyapunov_core.py)

## Main findings

The RL notebooks are aligned with the intended method on these points:

- candidate acceptance uses a one-step Lyapunov contraction check
- no hard `\Delta u` safety bound is active, because the notebooks do not pass `du_min` or `du_max`
- the fallback objective uses tracking plus move suppression
- the fallback objective does not include the steady-input objective term
- the fallback objective does not include the terminal Lyapunov objective term

The RL notebooks are not fully aligned with a strict "first-step contraction and nothing else" statement:

- the fallback solver is still instantiated with `terminal_set_on=True`
- the online path may skip that constraint when `\alpha` is very small, but it is still part of the constructed fallback solver

## Report update

Updated [report/direct_lyapunov_method_step_by_step_2026-05-13.md](../report/direct_lyapunov_method_step_by_step_2026-05-13.md) to:

- add an explicit alignment verdict near the top
- state that the active RL gate does not use hard move bounds
- state that the active hard-mode objective is tracking plus move penalty
- state that Lyapunov enters as a first-step constraint, not as an objective penalty
- state explicitly that `terminal_set_on=True` remains a mismatch with the stricter intended formulation

## Validation

Validation was done by direct source inspection:

- notebook parameter blocks were checked with `rg`
- gate behavior was checked in `evaluate_candidate_action(...)`
- fallback construction and constraint behavior were checked in `design_direct_lyapunov_mpc_solver(...)` and `solve_direct_tracking_from_target(...)`
