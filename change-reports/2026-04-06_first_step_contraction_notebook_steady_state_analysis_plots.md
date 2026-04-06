# 2026-04-06 First-Step-Contraction Notebook Steady-State Analysis Plots

## Summary

Added the same frozen-`dhat` steady-state analysis sidecar used in the offset-free debug notebook to `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`.

## What Changed

- Extended the notebook imports to include:
  - `analyze_offsetfree_rollout`
  - `save_offsetfree_ss_debug_artifacts`
  - `IPython.display.Image`
- Added a new notebook cell after the first-step-contraction debug export that:
  - builds a rollout dictionary from the first-step-contraction run bundle
  - runs the same steady-state analysis sidecar
  - saves the analysis artifacts under `Data/standard_lyap_first_step_contraction/steady_state_debug/`
  - prints the summary and box-analysis summary
  - renders key analysis plots inline in the notebook

## Inline Plots Added

- `outputs_vs_target.png`
- `inputs_vs_targets_dev.png`
- `reduced_rhs_vs_Gu.png`
- `tail_last_20_samples_overview.png`

## Purpose

This makes it easier to compare the bounded frozen-`dhat` first-step-contraction case against the earlier offset-free steady-state analysis directly in the notebook, especially when inspecting why `x_s` is near zero here while it was not in the other workflow.

## Validation

Ran notebook JSON parse for:

- `LyapunovFirstStepContractionMPC_BoundedFrozenDhat.ipynb`

## Notes

- No controller code changed in this update.
- Existing unrelated worktree changes were left untouched.
