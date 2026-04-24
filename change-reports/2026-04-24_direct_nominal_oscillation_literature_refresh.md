# Direct Nominal Oscillation Literature Refresh

## Summary

Rewrote `report/direct_lyapunov_nominal_oscillation_analysis.md` around the
latest nominal ten-scenario export:

`Data/debug_exports/direct_lyapunov_mpc_ten_scenario/20260423_234338`

The updated report keeps the earlier four-scenario oscillation windows as the
failure-mode evidence, then uses the ten-scenario results to show that
previous-input target regularization removes the target-corner toggling
signature.

## Changes

- Reframed the oscillation diagnosis as an internal target-management problem:
  moving `d_hat` plus nonsmooth bounded target projection plus Lyapunov
  recentering.
- Added ten-scenario performance evidence for all bounded regularized cases.
- Added a window retest showing that the old flat-setpoint target-corner
  toggles disappear in the regularized soft cases.
- Extended the next-step recommendation beyond weight tuning:
  - keep `bounded_soft_u_prev_1p0` as the current nominal baseline;
  - inspect the remaining hard `lambda_prev = 10.0` held steps;
  - run a focused `lambda_prev` sweep around 1.0;
  - implement a Limon-style artificial steady-state tracking MPC variant;
  - revisit disturbance-model and observer design after target management is
    stable;
  - consider a reference governor if infeasible setpoints remain problematic.
- Added literature context and references for offset-free MPC, constrained
  tracking MPC with artificial steady states, Lyapunov MPC, and reference
  governors.

## Validation

- Checked every figure link in the rewritten report resolves to an existing
  file.
- Re-read the ten-scenario `comparison_table.csv` for the reported metrics.
- Parsed the latest ten-scenario step tables to compare target-corner toggles
  in the old oscillation windows.
- Ran `git diff --check` on the rewritten report.
