# Archive RL Notebooks As Scripts

## Summary

Converted the two active direct Lyapunov safety-gate RL notebooks into Python script entrypoints and moved the original notebooks into `archive/`.

## Changes

- Added `DirectLyapunovSafetyGateRL_ColdStart.py` and `DirectLyapunovSafetyGateRL_Pretrained.py`.
- Preserved notebook cell boundaries in the scripts using `# %%` markers so IDEs can still run cells interactively.
- Moved the original notebooks to `archive/`.
- Cleared archived notebook outputs and execution counts so result blobs and figures are not carried forward in source control.

## Validation

- Passed `py_compile` for both generated scripts.
- Passed `nbformat` validation for both archived notebooks.

## Notes

No training runs, result regeneration, or figure generation were performed.
