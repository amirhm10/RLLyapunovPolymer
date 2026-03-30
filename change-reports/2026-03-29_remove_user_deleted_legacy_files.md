# Remove User-Deleted Legacy Files

## Why
- These files were already removed locally and the intent is to keep them deleted in the repository.
- They should be tracked as an explicit cleanup commit rather than remaining as dangling local deletions.

## Files Removed
- `ComparePlots.ipynb`
- `LyapDetails.md`
- `dmc.ipynb`
- `lyapunov_safety_filter_report_outline.tex`
- `safe_mpc_filter_research_bundle.py`
- `target_selector_mode_smoke_test.py`

## What Was Not Included
- No controller code changes.
- No notebook parameter resets.
- No unrelated untracked review documents were included in this commit.

## Validation
- Cleanup-only Git commit; no runtime behavior changed.
