# Pretrained `rho=0.99` Last-Episode Target Offset Analysis

Date: 2026-05-13

## Summary

Added a new focused report for the latest pretrained direct safety-gate run:

- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- export bundle `20260512_071313`

The analysis answers three specific questions:

1. why oscillation can remain after the loop reaches the setpoint region
2. why offset can remain even though `y_sp` is used in target selection
3. whether `y_s = y_sp` in the last episode for each of the four scenarios

## Files added

- [report/pretrained_direct_safety_gate_rho99_last_episode_target_offset_analysis_2026-05-13.md](../report/pretrained_direct_safety_gate_rho99_last_episode_target_offset_analysis_2026-05-13.md)

## Figures added

- [report/figures/2026-05-13_pretrained_rho99_last_episode/episode200_seg2_outputs_vs_targets.png](../report/figures/2026-05-13_pretrained_rho99_last_episode/episode200_seg2_outputs_vs_targets.png)
- [report/figures/2026-05-13_pretrained_rho99_last_episode/episode200_seg2_tail100_outputs_vs_targets.png](../report/figures/2026-05-13_pretrained_rho99_last_episode/episode200_seg2_tail100_outputs_vs_targets.png)

## Main findings captured

- The combined anchored case is the cleanest late-episode case, but its remaining offset is almost entirely `y_s - y_sp`.
- The plain `bounded_hard` case still has both target mismatch and residual oscillatory tracking around the selected target.
- `y_s = y_sp` only on steps where the selector stays in `frozen_output_disturbance_exact_bounded`.
- In bounded least-squares mode, `y_sp` is a desired target in the objective, not a hard equality guarantee.

## Validation

- Cross-checked notebook configuration for `rho_lyap = 0.99` and raw-setpoint fallback tracking.
- Read the saved `summary.json`, `episode_table.csv`, `step_table.csv`, and `arrays.npz` for all four pretrained cases.
- Computed episode-200 and tail-100 metrics directly from the saved arrays using the repo's `rl-env` Python environment.
