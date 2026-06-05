# Latest Run Phase Math Report

Date: 2026-06-05

## Summary

Added a research report that summarizes the latest direct Lyapunov MPC and safety-gate RL run state, reconstructs the governing mathematics, and identifies the current workflow phase as saved-agent evaluation alignment.

## Changes

- Added `report/latest_run_phase_math_algorithm_2026-06-05.md`.
- Summarized the latest direct, cold-start, and pretrained result folders on disk.
- Documented the governed-reference target equations, direct Lyapunov MPC contraction test, safety-gate RL action selection, and reward/no-penalty reward channels.
- Flagged the current saved-agent evaluation mismatch: the root saved-agent script is set to nominal plant mode, while `Simulation/saved_agent_evaluation.py` still hard-codes bounded target mode.
- Recommended a two-stage saved-agent evaluation plan: governed-reference aligned nominal evaluation first, then disturbed evaluation.

## Validation

- No Python files were changed for this report.
- Existing result summaries and comparison tables were read from `results/`.
- The report embeds existing figures and does not modify raw result artifacts.

## Notes

The existing local modification in `DirectLyapunovSavedAgentEvaluation.py` was inspected but not committed as part of this report. It remains a user/worktree change.
