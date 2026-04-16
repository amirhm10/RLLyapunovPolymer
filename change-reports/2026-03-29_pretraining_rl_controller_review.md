# Pretraining RL Controller Review Note

## Why
- The notebook `pretraining_rl_controller.ipynb` is a key entrypoint for the offline MPC-to-TD3 pretraining path.
- Before changing that pipeline, we needed a precise written account of what the notebook actually does and which Python modules define its behavior.

## What Was Added
- Added `report/pretraining_rl_controller_review.md`.

## What The New Note Covers
- cell-by-cell walkthrough of the notebook
- actual helper modules used by the notebook
- scaling and augmented-state conventions
- replay-buffer generation from MPC labels
- actor behavioral cloning stage
- critic TD-pretraining stage
- save/load and evaluation behavior
- main weaknesses and concrete improvement directions

## What Was Not Changed
- No notebook parameters were changed.
- No `.py` files were changed.
- No old notebooks were removed.
