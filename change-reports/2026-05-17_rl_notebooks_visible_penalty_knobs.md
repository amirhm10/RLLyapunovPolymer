# RL Notebook Visible Penalty Knobs

## What changed

Updated these notebooks:

- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

Both notebooks now expose the direct target regularization weights near the top of the main configuration cell:

- `u_prev_penalty_weight = 0.25`
- `xs_prev_penalty_weight = 0.25`
- `case_variants = ("none", "mixed")`

The case builder now uses those visible variables instead of hard-coded numeric values when creating `case_specs`.

## Result

You can change the previous-input anchor weight and the state smoothness weight directly in the notebook before running RL experiments.

The saved per-case metadata now also records:

- `u_prev_penalty_weight`
- `xs_prev_penalty_weight`
- `case_variants`

## Validation

- Notebook structure validated with `nbformat.validate(...)` for both edited notebooks.
- No training run was executed as part of this change.
