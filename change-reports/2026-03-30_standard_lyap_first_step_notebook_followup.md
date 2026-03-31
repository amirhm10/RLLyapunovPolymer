# Standard Lyapunov First-Step Contraction Notebook Follow-Up

## Why
- The initial notebook version exposed the contraction settings, but it did not mirror the other controller notebooks closely enough.
- In particular:
  - selector-related values were not grouped under a `selector_config` block in `run_config`,
  - the notebook ended with only a small summary output,
  - there was no built-in plotting/export section.

## What Changed
- Updated `StandardLyapMPCFirstStepContraction.ipynb` to:
  - add `selector_config` inside `run_config`,
  - pass selector settings from `run_config["selector_config"]`,
  - unpack the full result tuple explicitly,
  - add standard MPC-style output/input plotting using `plot_mpc_results_cstr(...)`,
  - add a contraction-diagnostics plot showing:
    - `V_k`
    - `V_next_first`
    - `V_bound`
    - contraction margin
    - first-step contraction satisfaction
    - target-selector success

## What Was Not Changed
- No Python modules were changed.
- No controller behavior was changed.
- No existing notebook parameters outside this new notebook were touched.

## Validation
- Notebook JSON validation for `StandardLyapMPCFirstStepContraction.ipynb`
