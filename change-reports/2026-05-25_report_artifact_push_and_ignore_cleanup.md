# Report Artifact Push And Ignore Cleanup

## Summary

Committed report-like artifacts that were sitting untracked in the working tree and added ignore rules for generated CSChE slide build products.

## Included Report/Source Artifacts

- Root deep-research Markdown reports:
  - `deep-research-report.md`
  - `deep-research-report2.md`
- Governed-reference analysis report bundle:
  - `report/governed_reference_latest_three_run_analysis_2026-05-23.md`
  - `report/governed_reference_latest_three_run_metrics_2026-05-23.json`
- CSChE presentation source bundle:
  - Beamer source, README, speaker notes, and draft change report
  - figure-generation script
  - selected slide figure assets required by the deck

## Ignored Generated Outputs

Added `.gitignore` entries for CSChE LaTeX build products, rendered slide preview pages, and script bytecode caches.

## Left Out Intentionally

The existing one-line change in `DirectLyapunovSavedAgentEvaluation.py` changes the evaluation mode from `disturb` to `nominal`; it is not a report or generated-result cleanup, so it was not included in this commit.
