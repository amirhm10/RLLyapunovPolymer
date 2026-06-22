# Revert GART Contraction Probe To CVXPY

## Context

The previous GART contraction-probe speedup replaced the small CVXPY box-QP with a NumPy active-set solve. Although the active-set solve matched CVXPY in spot checks, the experiment should remain on the original CVXPY implementation for now to avoid changing numerical behavior during the online TD3 comparison.

## Changes

- Removed the NumPy active-set helper used by `contraction_probe`.
- Restored the original CVXPY formulation:

  $$\min_{u_{\min} \le u \le u_{\max}} (A\hat{x} + Bu - x_s)^T P_x (A\hat{x} + Bu - x_s).$$

- Left the OF-MPC pretrained critic reset and `80000` online replay-buffer capacity changes unchanged.

## Validation

- `python -m py_compile Lyapunov/gart_target.py`
- Checked that `contraction_probe` no longer references the active-set helper and once again calls `_solve_problem(...)` on a CVXPY problem.
