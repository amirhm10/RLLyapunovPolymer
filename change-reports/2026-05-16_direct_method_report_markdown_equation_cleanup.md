# Direct Method Report Markdown Equation Cleanup

## What changed
- Reworked `report/direct_lyapunov_direct_and_rl_method_2026-05-16.md` so equation-heavy sections no longer use multi-line pseudo-equation code fences.
- Replaced the awkward plain-text placeholder notation such as `x_(k+1)` with proper Markdown math using `$...$` and `$$...$$`.
- Converted the main model, observer, target-selector, Lyapunov, MPC, reward, and behavioral-cloning equations into standard rendered math notation.
- Cleaned up several index notations such as previous-input and next-step references so they now appear as standard forms like `$u_{k-1}$`, `$x_{k+1}$`, and `$V_{k+1}^{\mathrm{cand}}$`.

## Why
- The report was still mathematically correct, but several expressions rendered like raw pseudo-LaTeX or code dumps instead of readable equations.
- The goal of this cleanup was to preserve the math and implementation detail while making the document render as real equations in a Markdown preview that supports math.

## Validation
- Scanned the report to confirm there are no remaining ````text` fenced pseudo-equation blocks.
- Checked the report for leftover placeholder notation such as `x_(k+1)` and `u_(k-1)`.
- Confirmed the main mathematical statements now use Markdown math delimiters instead of pseudo-equation code formatting.
