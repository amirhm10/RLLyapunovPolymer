# Fixed Event 10 RL Results Report Refresh

## Summary

Refreshed the current agent-authority BC report after rerunning the cold-start and pretrained RL scripts with `fallback_event_penalty = 10.0`. The direct Lyapunov MPC comparison remains the latest available direct run because only the RL scripts were rerun.

## Analyzed Runs

- Cold-start RL: `results/ColdStart/20260520_134418`
- Pretrained RL: `results/Pretrain/20260520_134418`
- Direct LMPC reference: `results/directLyap/20260520_005354`

## Main Result Update

- Increasing the fixed event penalty from `2.0` to `10.0` slightly reduced fallback rates.
- The reduction did not improve full-horizon RMSE.
- Augmented reward became worse because the higher fixed event cost is now visible in the reward.
- Pretrained RL remains the better learned controller, but it still does not beat its corresponding MPC-only diagnostic.
- Cold-start RL remains more gate-compatible, but its BC phase is still very costly under high exploration.
- The target-selector tension remains central because MPC-only raw tracking is strong while would-be Lyapunov gate activation remains high.

## Report And Figure Changes

- Updated `report/rl_agent_authority_bc_latest_analysis_2026-05-19.md` with the latest RL folders, tables, interpretation, and already-tried notes.
- Added a new figure folder:
  - `report/figures/2026-05-20_fixed_event_10_latest_analysis/`
- Added a fixed-event penalty comparison figure to show the `2.0` versus `10.0` effect.
- Regenerated the self-contained HTML report with embedded figures.

## Validation

- Confirmed the latest result folders are referenced in the report.
- Confirmed the HTML embeds figures with `data:image/...;base64`.
- Confirmed the HTML has no local `src="figures/...` references.
- Ran `git diff --check` before commit.
