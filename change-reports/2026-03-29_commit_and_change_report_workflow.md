# Commit And Change-Report Workflow

## Why
- The repository is now under Git and connected to GitHub.
- Major controller and notebook changes need stable checkpoints so they can be revisited, compared, or reverted later.
- The project already uses `change-reports/`, so the Git history should be paired with those reports instead of leaving them disconnected.

## What Changed
- Updated `AGENTS.md` to reflect that the folder is now a Git repository with a GitHub remote.
- Added an explicit workflow rule:
  - major updates should end with a Git commit
  - the commit message should describe the technical change clearly
  - each major commit should include a matching Markdown report under `change-reports/`
  - distinct major updates should prefer distinct commits and distinct change reports
  - low-cost validation should be run before committing when practical

## Expected Workflow Going Forward
1. Make the code or notebook change.
2. Run the lightweight validation appropriate for the touched files.
3. Create or update a matching file in `change-reports/`.
4. Commit the code change and the change report together.
5. In the close-out message, report the commit hash and the change-report path.

## Notes
- The change report is not "attached" to the commit in a special Git feature; instead, it is included in the same commit so the commit itself contains both the implementation and its report.
- This keeps the history simple and works well with GitHub browsing, diffs, and rollback.
