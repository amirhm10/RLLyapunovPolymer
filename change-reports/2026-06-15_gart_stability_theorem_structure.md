# GART Stability Workflow Theorem Structure

## Summary

Reframed the GART stability workflow report in a more paper-style theorem format and removed the generated HTML report artifact.

## Changes

- Added Section 12.1 with explicit assumptions for the stability statement:
  - valid Lyapunov matrix;
  - accepted target sequence;
  - certified executed input;
  - successful certified step;
  - bounded model/estimator residual for raw tracking.
- Added Theorem 1 for practical target-centered stability.
- Added a proof-conservatism note explaining the small allowable proof parameter range when `rho = 0.98`.
- Added Corollary 1 for raw-setpoint tracking interpretation.
- Added Proposition 1 for RL exploration under the GART hard safety projection.
- Clarified that the terminal constraint supports the finite-horizon MPC design but is not the minimal ingredient for the one-step moving-target Lyapunov proof.
- Added `u_k^{exec}` to the notation table.
- Removed `report/gart_stability_workflow.html`.

## Validation

- Rebuilt `report/gart_stability_workflow.pdf` from `report/gart_stability_workflow.tex` with `pdflatex -interaction=nonstopmode -halt-on-error`.
- LaTeX compilation succeeded. Remaining warnings are nonfatal overfull boxes in long numeric notation-table entries.
