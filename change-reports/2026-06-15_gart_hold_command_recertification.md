# GART Hold-Command Recertification Fix

## Summary

Fixed the GART target selector behavior for the zero-move governor case. Holding the governor command now means holding the command reference, not reusing a stale target triple.

## Problem

The governor loop could solve and accept an `alpha = 0` candidate, corresponding to

$$
r_k=r_{k-1}.
$$

However, the selector only returned governed candidates when `governor_alpha > 0`. If the only accepted candidate was `alpha = 0`, the code fell back to `_hold_previous_result(...)`, which reused the previous stored `x_s`, `u_s`, and `y_s` while attaching the current certified disturbance `d^c_k`.

That could make the target package inconsistent with the final target equation:

$$
y_{s,k}=Cx_{s,k}+d^c_k.
$$

## Changes

- Return any accepted governor candidate, including `alpha = 0`.
- Mark the accepted zero-move candidate as `accepted_held_command_reference`.
- Treat `alpha = 0` as a held-command solve with `hold_previous=True`.
- Make `_hold_previous_result(...)` explicitly non-usable for LMPC.
- Preserve stale-target diagnostics, including the target-equation residual, but prevent LMPC from optimizing around that stale package.
- Updated the stability workflow report wording to distinguish command-reference holding from stale target reuse.
- Added regression tests for:
  - accepted `alpha = 0` re-certification with current `d^c_k`;
  - non-usable stale fallback when the zero-move target cannot be re-certified.

## Validation

- `python -m py_compile Lyapunov/gart_target.py tests/test_gart_target.py`
- Direct Python regression check through `rlenv` for the two new hold-command scenarios.
- `pdflatex -interaction=nonstopmode -halt-on-error gart_stability_workflow.tex`

`pytest` is not installed in the available Python environments, so the full `pytest tests/test_gart_target.py` command could not be run locally.
