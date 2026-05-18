# No-RL Notebook Visible Case Variants

## What changed

Updated `DirectLyapunovMPC_FourMethodDisturbance.ipynb` so the selected direct-study case variants are visible in the main configuration cell.

Added:

- `case_variants = ("none", "mixed")`

and wired that variable into:

- `active_config["case_variants"]`
- `direct_four_method_case_specs(..., variants=case_variants)`

## Result

The no-RL direct disturbance notebook now matches the RL notebooks more closely:

- the penalty weights are visible
- the selected case subset is also visible and editable
- the saved configuration records which case variants were used

## Validation

- Notebook structure validated with `nbformat.validate(...)`
- Confirmed the config cell now contains the visible `case_variants` variable and uses it in `case_specs`
