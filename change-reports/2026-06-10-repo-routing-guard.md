# Repo Routing Guard

## Summary
- Added an explicit sibling-repository boundary to `AGENTS.md` in `Lyapunov_polymer`.
- Clarified that code, controller, simulation, result-export, and plotting work belongs in `Lyapunov_polymer`.
- Clarified that manuscript, paper-memory, literature, citation, figure/table registry, and workflow-note work belongs in `Lyapunov Paper`.
- Added the reciprocal boundary note to the paper workspace `AGENTS.md` outside this git repository.

## Safety Impact
- Future sessions should announce the intended write target when a request could touch both folders.
- The sibling folder is treated as read-only context unless Amir explicitly requests cross-repo edits.
- Paper tasks keep the code repo read-only by default; code tasks keep the paper workspace read-only by default.

## Validation
- Reviewed the code-repo diff for `AGENTS.md`.
- Read back the updated paper-workspace `AGENTS.md` section.
- No Python validation was needed because this was documentation-only.
