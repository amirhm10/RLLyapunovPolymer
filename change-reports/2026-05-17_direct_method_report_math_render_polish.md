# Direct Method Report Math Render Polish

## What changed
- Updated `report/direct_lyapunov_direct_and_rl_method_2026-05-16.md` to improve Markdown math rendering reliability.
- Replaced several code-formatted variable names used as mathematical symbols with proper math notation, including:
  - `$Q_i$`
  - `$Q_s$`
  - `$hA$`
  - `$P_x$`
  - `$u_k$`
  - `$\alpha$`
- Rewrote the remaining multi-line display equations into compact single-line `$$ ... $$` display math blocks.

## Why
- Some IDE Markdown previews were still rendering parts of the report poorly.
- Two common sources remained:
  - math-like symbols written as code spans instead of math;
  - multi-line display equations that were valid LaTeX but less robust across previewers.

## Validation
- Scanned the report for leftover multi-line `$$ ... $$` blocks.
- Scanned for the specific code-style variable patterns that were causing poor rendering.
- Confirmed the report now uses compact display math consistently in the affected sections.
