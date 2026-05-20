# Reward Diagnosis Report Extension

## Summary

Extended the latest agent-authority BC analysis report with a reward-focused diagnosis and a proposed next reward setup.

## Changes

- Added the full current reward formula with term-by-term explanation.
- Added measured reward-component evidence for cold-start and pretrained RL.
- Added figures for reward penalty scale and episode-level reward-base versus fallback-penalty trends.
- Added a proposed stricter reward setup focused on fallback events and near-zero temperature offset.

## Main Finding

The fixed fallback event penalty is too small on an average-per-step basis. With `fallback_event_penalty = 0.5`, the fixed event contribution is only about `0.006` per step for cold RL and `0.014` per step for pretrained RL in the latest completed runs. The fallback signal mostly comes from the correction-gap term, so frequent small fallbacks are not punished strongly enough.

## Validation

- Recomputed reward component summaries from the saved latest RL arrays and episode tables.
- Checked the Markdown report for control-character issues after adding LaTeX formulas.
- Did not change reward code or rerun training.
