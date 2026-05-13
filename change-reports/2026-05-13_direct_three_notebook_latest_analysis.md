# Direct Three-Notebook Latest Analysis

Date: 2026-05-13

## Summary

Added a new Markdown report that synthesizes the latest saved results from the three direct Lyapunov notebooks:

- `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

The report separates three issues that had been getting conflated:

1. direct-MPC target mismatch and target drift
2. RL final-episode settling versus shifted-target settling
3. cold-start RL instability at the BC-to-full-RL transition

## Main findings captured in the report

- The direct Lyapunov contraction test certifies contraction around the selected admissible target `(x_s, u_s)`, not around the raw setpoint.
- All three current direct notebooks still use raw-setpoint tracking in the direct tracking objective, so contraction and raw tracking remain structurally misaligned.
- The latest complete pretrained RL sweep is strongest in the combined regularized case.
- The latest complete cold-start RL sweep is also strongest in the combined regularized case.
- The cold-start `u_prev`-only case is not mainly a final-episode settling failure. Its whole-run metrics are dominated by catastrophic episodes 31 to 33 right after the BC-to-RL handoff.
- The latest direct-MPC rerun on 2026-05-12 is incomplete, so the direct four-method conclusions still rely on the latest complete sweep from 2026-05-11.
- The latest case labels still say `0p1` even though the saved runs use `0.25` weights, so the names are stale and should not be interpreted literally.

## Files added

- [report/direct_lyapunov_three_notebooks_latest_analysis_2026-05-13.md](../report/direct_lyapunov_three_notebooks_latest_analysis_2026-05-13.md)

## Validation

- Checked the latest comparison summaries in `Data/debug_exports/...` for all three studies.
- Cross-checked the current notebook configuration cells for:
  - `rho_lyap = 0.99`
  - `anchor_weight=0.25`
  - `smoothness_weight=0.25`
  - raw-setpoint tracking in the direct path
- Verified the cold-start `u_prev` failure cluster from the saved `episode_table.csv`.
