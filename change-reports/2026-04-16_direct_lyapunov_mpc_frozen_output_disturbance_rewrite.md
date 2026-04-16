# Direct Frozen-Output-Disturbance Lyapunov MPC Rewrite

## Summary

This change replaces the transitional frozen-`d_hat` first-step-replacement path with a new direct Lyapunov MPC workflow built around:

- a clean frozen output-disturbance target solver
- a direct Lyapunov MPC solve at every online step
- hard and soft Lyapunov modes
- a dedicated notebook, bundle/export path, smoke tests, and technical note

The old frozen-`d_hat` target module and its notebook entrypoint were deleted, and the old first-step runner no longer exposes `target_generation_mode` or `frozen_target_config`.

## Main Changes

### New target solver

Added `Lyapunov/frozen_output_disturbance_target.py` with:

- `solve_target_unbounded_output_disturbance(...)`
- `solve_target_bounded_output_disturbance(...)`
- `solve_output_disturbance_target(...)`

Key behavior:

- validates output-disturbance-only augmentation
- freezes `d_s = d_hat_k`
- solves only for `x_s` and `u_s`
- reports rank, conditioning, residuals, exact-vs-least-squares, and bounded fallback diagnostics
- rejects augmentations with nonzero `A_xd` so the direct path stays aligned with the clean model in the rewrite plan

### New direct controller path

Added `Lyapunov/direct_lyapunov_mpc.py`.

The module reuses the existing `FirstStepContractionTrackingLyapunovMpcSolver` core and extends it only where needed:

- hard mode delegates to the existing first-step-contraction MPC logic
- soft mode adds one nonnegative Lyapunov slack with configurable penalty
- rollout uses one target solve and one direct Lyapunov MPC solve per step
- step logs include target diagnostics, solver status, Lyapunov values, contraction margins, and slack statistics

The same module now provides:

- solver construction helper
- closed-loop rollout entrypoint
- direct-run bundle builder
- summary-table generation
- stable plot export
- timestamped artifact saving/loading

### Legacy cleanup

Deleted:

- `Lyapunov/frozen_dhat_target.py`
- `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`

Updated `Simulation/run_mpc_first_step_contraction.py` to remove:

- frozen target imports
- `target_generation_mode`
- `frozen_target_config`
- bounded/unbounded frozen-target branches

The old runner now stays on the refined-selector path only.

### Notebook and documentation

Added `DirectLyapunovMPC_FrozenOutputDisturbance.ipynb` as the new experiment entrypoint.

Notebook behavior:

- top-level visible config for `target_mode`, `lyapunov_mode`, horizons, `rho_lyap`, `lyap_eps`, and `slack_penalty`
- uses the clean `output_disturbance` augmentation instead of the old `mixed_B_I` augmentation
- calls the new Python helpers instead of embedding controller math in cells
- includes a reusable `run_case(...)` helper, single-run cell, optional comparison sweep, and artifact display cell

Added:

- `report/direct_lyapunov_mpc_frozen_output_disturbance.tex`
- `report/direct_lyapunov_mpc_frozen_output_disturbance.pdf`

Updated:

- `report/README.md`

### Synthetic validation helpers

Added `Lyapunov/direct_lyapunov_smoke_tests.py` with small checks for:

- exact unbounded target solve
- least-squares target fallback
- bounded target with active bounds
- reduced-form target diagnostics
- feasible hard direct controller solve
- soft direct controller solve with positive slack activation

## Validation

Executed with `C:\Users\HAMEDI\miniconda3\envs\rl-env\python.exe` unless noted otherwise.

- `python -m py_compile Lyapunov\frozen_output_disturbance_target.py Lyapunov\direct_lyapunov_mpc.py Lyapunov\direct_lyapunov_smoke_tests.py Simulation\run_mpc_first_step_contraction.py`
- `python -m Lyapunov.direct_lyapunov_smoke_tests`
- import check for:
  - `Lyapunov.direct_lyapunov_mpc`
  - `Lyapunov.frozen_output_disturbance_target`
- notebook JSON validation:
  - `nbformat.validate(DirectLyapunovMPC_FrozenOutputDisturbance.ipynb)`
- lightweight notebook execution in `rl-env` using `nbclient` on a temporary in-memory copy with:
  - `set_points_len = 10`
  - `save_debug_bundle = False`
  - `save_plots = False`
- direct export-path smoke run confirming:
  - summary fields are present
  - stable plot filenames are emitted
  - plots `01` through `06` are written
- LaTeX build:
  - `pdflatex -interaction=nonstopmode direct_lyapunov_mpc_frozen_output_disturbance.tex`
  - run twice from `report/`

## Notes

- The default system `python` still lacks the scientific stack, so import/runtime validation used the existing `rl-env` conda environment.
- Notebook execution required elevated permissions outside the sandbox because Jupyter needed to create a kernel connection file on Windows.
- The notebook setup now derives `TEST_CYCLE` from `n_tests` and uses `augmentation_mode="output_disturbance"` so the new direct path matches the clean frozen-output-disturbance assumptions.
