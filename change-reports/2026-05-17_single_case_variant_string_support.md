# Single-Case Variant String Support

## What changed

Updated `utils/direct_lyapunov_study.py` so `direct_four_method_case_specs(...)` accepts single-scenario variant selections as plain strings.

Supported forms now include:

- `"mixed"`
- `("mixed",)`
- `["mixed"]`
- `"none,mixed"`

The helper normalizes case names, trims whitespace, and keeps the existing validation for unknown entries.

## Notebook usability

Added a short hint in these notebooks:

- `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

The visible config line now indicates that a single scenario can be written as:

- `case_variants = "mixed"`

instead of requiring Python one-item tuple syntax.

## Result

You can now switch from multiple cases to one case directly in the notebook without hitting the character-splitting error from iterating over a bare string.

## Validation

- Verified `direct_four_method_case_specs(variants="mixed")`
- Verified `direct_four_method_case_specs(variants=("mixed",))`
- Verified `direct_four_method_case_specs(variants="none,mixed")`
- Validated all three edited notebooks with `nbformat.validate(...)`
