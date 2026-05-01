# 2026-05-01 Direct Three-Method Rho Sensitivity Report Rewrite

## Why This Revision Was Needed

The first rewrite on `2026-05-01` corrected the high-level scientific framing,
but the report structure was still wrong for the intended use.

The user needed the report to be organized around each method, not around the
sweep summary alone. In particular, the report needed to show:

- what each method is
- the method-specific mathematics
- output tracking performance
- applied inputs and steady inputs `u_s`
- steady target states `x_s`
- steady disturbance targets `d_s`
- Lyapunov evidence, including both:
  - the contraction plot
  - `Delta V = V_next_first - V_k`, which should be negative first and then go
    to zero on a good run

Without that rewrite, the report still looked like a metric digest rather than
a controller-analysis document.

## What Changed

Rewrote:

- `report/direct_lyapunov_bounded_single_setpoint_settling_report_2026-04-30.md`

Extended:

- `analysis/direct_lyapunov_rho_sensitivity_report.py`

The generator now reads the saved `arrays.npz` bundles directly using a
stdlib-only NPY/NPZ reader so it can recover the physical output and input
trajectories even though the active shell Python does not have `numpy`
installed.

## New Report Assets

The updated figure directory is:

- `report/figures/2026-05-01_direct_three_method_rho_sensitivity/`

New generated SVG figures:

- `bounded_hard_outputs_by_rho.svg`
- `bounded_hard_inputs_by_rho.svg`
- `bounded_hard_disturbance_targets_by_rho.svg`
- `bounded_hard_delta_v_by_rho.svg`
- `bounded_hard_u_prev_0p1_outputs_by_rho.svg`
- `bounded_hard_u_prev_0p1_inputs_by_rho.svg`
- `bounded_hard_u_prev_0p1_disturbance_targets_by_rho.svg`
- `bounded_hard_u_prev_0p1_delta_v_by_rho.svg`
- `bounded_hard_xs_prev_0p1_outputs_by_rho.svg`
- `bounded_hard_xs_prev_0p1_inputs_by_rho.svg`
- `bounded_hard_xs_prev_0p1_disturbance_targets_by_rho.svg`
- `bounded_hard_xs_prev_0p1_delta_v_by_rho.svg`

Retained and reused:

- `method_metrics_by_rho.svg`
- `bounded_hard_contraction_ratio_by_rho.svg`
- `bounded_hard_u_prev_0p1_contraction_ratio_by_rho.svg`
- `bounded_hard_xs_prev_0p1_contraction_ratio_by_rho.svg`

New generated CSV summaries:

- `rho_sweep_detailed_summary.csv`
- `rho_sweep_tail_targets.csv`

Existing generated CSVs retained:

- `rho_sweep_summary.csv`
- `rho_run_mapping.csv`

## Scientific Content Added

The report now makes five specific improvements:

1. It presents a common direct-Lyapunov formulation first, then gives the
   method-specific target objective for each of:
   - `bounded_hard`
   - `bounded_hard_u_prev_0p1`
   - `bounded_hard_xs_prev_0p1`
2. It reports per-method RMSE, solver success, hard-contraction rate,
   bounded-LS usage, and violation counts across
   `rho_lyap = 0.95, 0.98, 0.985, 0.99`.
3. It adds explicit Lyapunov decrement tables using
   `Delta V_pred = V_next_first - V_k`, including the tail-mean magnitude.
4. It reports tail steady targets for every method and rho:
   - `u_s` in physical units
   - `x_s` in saved augmented-model coordinates
   - `d_s` in saved augmented-model coordinates
5. It states the key interpretive distinction that was missing before:
   a run can satisfy the Lyapunov contraction pattern and still be a poor
   controller if it converges to the wrong steady target.

The clearest example of that last point is
`bounded_hard_u_prev_0p1` at `rho_lyap = 0.985`, where
`Delta V_pred` goes to zero but the selected steady target is shifted and the
tracking RMSE is poor.

## Validation

Ran successfully:

- `python analysis/direct_lyapunov_rho_sensitivity_report.py`

Low-cost syntax validation:

- `python -c "compile(open(...).read(), ..., 'exec')"`

`python -m py_compile analysis/direct_lyapunov_rho_sensitivity_report.py`
still hits the local Windows bytecode-write permission problem in this repo
setup, so the fallback syntax validation was used after regeneration completed
successfully.

