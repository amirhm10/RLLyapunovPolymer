# Safety Debug Windows Paper Plot Paths

## Why
- The direct RL safety-gate notebooks failed while saving paper-style safety debug figures on Windows.
- The failing export path exceeded the practical Windows path limit for OneDrive-hosted repo paths.
- The path-length estimator in `Lyapunov/safety_debug.py` did not include the longest paper-style figure name, so it could still choose an output directory that was too deep.

## What Changed
- Added paper-style filename shortening in [`Lyapunov/safety_debug.py`](../Lyapunov/safety_debug.py) through a dedicated filename map and helper path builder.
- Updated paper-style safety-selector exports to use the shortened filenames automatically while leaving non-paper debug exports unchanged.
- Extended the Windows path-length projection logic to include the long last-episode decomposition figure that triggered the failure.

## Effect
- The paper-style `safety_selector` figure paths are now shorter on Windows.
- The specific failing file `ys_decomposition_summary_last_episode.png` is saved as the shorter paper-style filename `ys_dec_sum_last.png`.
- Other long paper-style figure names such as last-episode overlays and first-step diagnostics are shortened the same way for margin.

## Validation
- Imported the updated helper module successfully in Python.
- Ran in-memory compile checks on the updated module logic.
- Verified that the previously failing paper-style path length drops from 261 characters to 239 characters for the same study root pattern.
- `python -m py_compile Lyapunov/safety_debug.py` still hits the existing OneDrive `__pycache__` rename permission issue in this repo, so validation used import and in-memory compilation instead.

## Notes
- Non-paper debug figures keep their original filenames.
- The figure manifest continues to reference the paper figure directory rather than specific filenames, so downstream directory-level discovery remains intact.
