# Direct Lyapunov Four-Scenario Supervisor Report

## Summary

Added a new supervisor-facing Markdown report for the latest direct Lyapunov MPC four-scenario run:

- `report/direct_lyapunov_mpc_four_scenario_supervisor_report.md`

The report explains the frozen output-disturbance direct Lyapunov MPC method step by step, including:

- output-disturbance augmentation,
- unbounded exact steady target selection,
- bounded input-constrained target projection,
- the corrected normal MPC objective using output setpoint tracking and input increments,
- terminal set interpretation,
- hard Lyapunov contraction,
- soft Lyapunov slack,
- four-scenario interpretation.

## Latest Run Analyzed

The report analyzes:

`Data/debug_exports/direct_lyapunov_mpc_four_scenario/20260423_195957`

Main findings:

- `unbounded_hard` is infeasible for every MPC step because exact targets are outside the admissible input box.
- `bounded_hard` has the best mean reward and satisfies hard contraction whenever the accepted solve succeeds.
- `unbounded_soft` has the lowest output RMSE but relies on large, frequent Lyapunov slack and inadmissible targets.
- `bounded_soft` has the highest solver success rate and only sparse, small Lyapunov slack.

## Figures

Copied report-ready figures into:

`report/figures/direct_lyapunov_mpc_frozen_output_disturbance/`

Newly referenced figures include:

- comparison reward, RMSE, solver/contraction, slack, target residual, output overlay, and input overlay plots,
- per-case output/target plots,
- per-case input/steady-target plots,
- per-case state-target error plots,
- per-case Lyapunov diagnostics,
- per-case target diagnostics.

Follow-up update:

- added per-case Lyapunov delta SVG diagnostics for `V_1|k - V_k` and `V_k - V_{k-1}`,
- added a comparison Lyapunov delta SVG plot,
- added `lyapunov_delta_summary.csv` for the report-ready delta statistics,
- updated the report interpretation to distinguish first-step predicted decrease from logged step-to-step decrease under moving targets.

## Validation

Performed lightweight validation:

- confirmed every local image link in the new report resolves,
- confirmed the new Markdown report contains only ASCII characters.
