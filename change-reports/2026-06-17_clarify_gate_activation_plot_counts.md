# Clarify Safety-Gate Activation Plot Counts

## Objective

Resolve the mismatch between low console fallback counts and larger per-episode activation counts in the latest cold-start safety-gate plots.

## Diagnosis

The latest saved cold-start safety-gate run,
`results/OnlineTD3_ColdStart_SafetyGate/20260617_104553`, contains Section-16 projection activity:

- accepted TD3 candidate: 231,311 steps
- Section-16 QCQP projection: 6,987 steps
- verified GART-LMPC fallback: 19 steps
- GART target-not-usable hold previous: 1,683 steps

The activation plot used total actual safety activity, while the console print showed only fallback/hold-previous activity and excluded the Section-16 projections. This made the print and plot look inconsistent even though they were aggregating different quantities.

## Changes

- Updated `Simulation/run_rl_lyapunov.py` so future GART safety-gate block prints report:
  - accepted candidate count
  - Section-16 projection count
  - fallback/hold-previous count
  - total intervention count
- Updated `Lyapunov/safety_debug.py` episode records with explicit projection and GART hold-previous columns.
- Updated safety comparison mode plots to include Section-16 projection and total hold-previous counts.
- Updated `Simulation/saved_agent_evaluation.py` so intervention-count plots use `actual_intervention_flags` when available.
- Added a plot-count audit section to `report/online_gart_gate_fallback_analysis_2026-06-17.md`.

## Validation

Low-cost Python compilation was run on the touched Python modules.

