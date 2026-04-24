# Direct Target U-Prev Regularization

## Summary

Added an optional previous-input target regularization term for the direct
frozen-output-disturbance Lyapunov MPC bounded target projection. The term is
off by default through `u_ref_weight = 0.0`, and the direct notebook now runs a
six-scenario study with two additional bounded cases that use
`lambda_prev = 0.1`.

For bounded cases whose exact target is outside the input box, the target
projection can now solve

```math
\min_{u_{\min}\le u_s\le u_{\max}}
\|G u_s - (y_{\mathrm{sp}}-\hat d)\|_2^2
+
\lambda_{\mathrm{prev}}\|u_s-u_{k-1}\|_2^2 .
```

## Changes

- Extended `Lyapunov/frozen_output_disturbance_target.py` with an optional
  `u_ref` argument and `u_ref_weight` target configuration.
- Passed the per-step `u_prev_dev` from
  `run_direct_output_disturbance_lyapunov_mpc(...)` into the target selector.
- Preserved baseline behavior: existing scenarios keep `u_ref_weight = 0.0`,
  exact bounded targets stay exact, and the new term only changes bounded
  least-squares projection after the exact target violates bounds.
- Added debug/export fields for `target_u_ref`, `target_u_ref_weight`,
  `target_u_ref_penalty`, `target_us_u_ref_inf`, and regularization active
  counts.
- Expanded `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` from four to six
  scenarios:
  - `unbounded_hard`
  - `bounded_hard`
  - `unbounded_soft`
  - `bounded_soft`
  - `bounded_hard_u_prev`
  - `bounded_soft_u_prev`
- Added a smoke-test check showing that enabling `u_ref_weight = 0.1` moves an
  out-of-bounds bounded target closer to the supplied `u_ref`.

## Validation

- Ran `python -m py_compile` on:
  - `Lyapunov/frozen_output_disturbance_target.py`
  - `Lyapunov/direct_lyapunov_mpc.py`
  - `Lyapunov/direct_lyapunov_smoke_tests.py`
- Ran a notebook JSON/AST check confirming exactly six cases, exactly two
  nonzero `u_ref_weight` target configs, and default/off configs for the
  original four cases.
- Ran `git diff --check`.
- Attempted `python Lyapunov/direct_lyapunov_smoke_tests.py`, but this sandbox
  Python environment does not have `numpy` installed.
