# Latest Strict Geometric Results Report Refresh

## Summary

Refreshed the current agent-authority BC report after the latest rerun of all three active entrypoints. The result section now reflects the new strict geometric reward setup, updated run folders, latest figures, revised findings, and the current interpretation of why pretrained RL now outperforms cold-start RL overall.

## Changes

- Updated `report/rl_agent_authority_bc_latest_analysis_2026-05-19.md` to analyze:
  - `results/ColdStart/20260520_005358`
  - `results/Pretrain/20260520_005403`
  - `results/directLyap/20260520_005354`
- Rewrote the results and findings around the latest evidence:
  - Pretrained RL is now the better learned controller by full-horizon reward and RMSE.
  - Cold-start RL is better mainly in safety-gate authority and final-tail temperature offset.
  - Cold-start BC remains fragile because high exploration with an untrained actor creates poor early episodes.
  - MPC-only still beats the corresponding safety-gated RL cases on full-horizon reward and RMSE.
  - The stronger fallback penalty is visible, but it does not eliminate gate dependence.
  - The target-selector tension remains central.
- Added the latest dated figure set to the report references:
  - `report/figures/2026-05-20_strict_geom_latest_analysis/...`
- Fixed a Markdown table header that used literal norm bars inside a pipe table.
- Updated the project-level handoff section so the stricter reward is described as the latest analyzed run, not a future run.
- Regenerated the self-contained HTML export with embedded base64 figures.

## Validation

- Confirmed the Markdown references the latest result folders.
- Confirmed the HTML embeds the latest figures as `data:image/...;base64`.
- Confirmed the HTML has no local `src="figures/...` image references.
- Ran `git diff --check` before commit.
