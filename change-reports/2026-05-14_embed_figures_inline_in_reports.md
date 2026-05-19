# Change Report: Embed Figures Inline In Reports

## What Changed

Updated the latest RL jitter report so the figures render inline inside the Markdown document:

- [report/latest_rl_jitter_root_cause_analysis_2026-05-13.md](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/latest_rl_jitter_root_cause_analysis_2026-05-13.md>)

Updated the repository guidance in:

- [AGENTS.md](/abs/c:/Users/HAMEDI/OneDrive%20-%20McMaster%20University/PythonProjects/Lyapunov_polymer/AGENTS.md)

## Policy Added

Reports that use figures should:

- embed the figures inline with relative Markdown image paths
- place the figures near the relevant discussion
- not leave figures only as bare links at the end of the report

## Why

The report preview in the IDE is more useful when the figures appear in the body of the document, especially for result-analysis notes where the figures directly support the argument.

## Validation

Validation was a source-level check:

- confirmed the report now contains inline image Markdown
- confirmed `AGENTS.md` now contains the new Markdown-report rule
