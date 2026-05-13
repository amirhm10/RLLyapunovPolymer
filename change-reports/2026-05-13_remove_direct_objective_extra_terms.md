# Remove Extra Objective Terms From Direct Lyapunov Path

Date: 2026-05-13

## Scope

This update removes the two extra direct-objective terms that were still appearing in the direct Lyapunov documentation and active direct notebook configurations:

- steady-input anchoring term `\sum \|u_i-u_s\|_{S_u}^2`
- terminal objective term `\|x_{N_P}-x_s\|_{P_x}^2`

The intended direct formulation is now:

$$
\text{tracking cost} + \text{move penalty}
$$

with Lyapunov used only through the first-step contraction check and any separate terminal-set constraint that remains active.

## Code changes

Updated [Lyapunov/direct_lyapunov_mpc.py](../Lyapunov/direct_lyapunov_mpc.py) so the direct solver path no longer adds:

- a steady-input objective term
- a terminal objective term

The hard direct solver no longer routes through the older generic objective path. It now uses the direct-path CVXPY formulation with:

- output tracking term
- `Rdu` move penalty
- first-step contraction constraint
- optional terminal-set constraint

Legacy objective flags were also removed from the active direct notebook calls.

## Notebook cleanup

Removed the old objective flags from:

- [DirectLyapunovMPC_FourMethodDisturbance.ipynb](../DirectLyapunovMPC_FourMethodDisturbance.ipynb)
- [DirectLyapunovMPC_FrozenOutputDisturbance.ipynb](../DirectLyapunovMPC_FrozenOutputDisturbance.ipynb)
- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](../DirectLyapunovSafetyGateRL_Pretrained.ipynb)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](../DirectLyapunovSafetyGateRL_ColdStart.ipynb)

## Report cleanup

Updated:

- [report/direct_lyapunov_method_step_by_step_2026-05-13.md](../report/direct_lyapunov_method_step_by_step_2026-05-13.md)
- [report/direct_lyapunov_mpc_frozen_output_disturbance_run_report.md](../report/direct_lyapunov_mpc_frozen_output_disturbance_run_report.md)
- [report/latest_lyapunov_results_synthesis_2026-04-30.md](../report/latest_lyapunov_results_synthesis_2026-04-30.md)

The writeups now describe the direct objective only as tracking plus move penalty.

## Validation

- Parsed all four edited notebooks with `ConvertFrom-Json`
- Ran `python -m py_compile Lyapunov/direct_lyapunov_mpc.py`
- Checked that the removed objective-flag names no longer appear in the active direct notebooks or direct-method report
