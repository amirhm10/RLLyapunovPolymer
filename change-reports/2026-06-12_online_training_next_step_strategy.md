# Online TD3 Next-Step Strategy Report

Date: 2026-06-12

## Summary

Added a research strategy report ranking the next online TD3 training ideas:

- pretrained critic recalibration
- critic last-layer reset
- full critic reset
- third fresh critic
- DAgger-style visited-state relabeling
- decayed BC strength and teacher-gap-triggered relabeling

## Report

- `report/online_training_next_step_strategy_2026-06-12.md`

## Main Recommendation

Run the new low-noise BC/handoff schedule first, then implement two critic
recalibration variants:

- keep pretrained critics and run actor-frozen critic-only recalibration
- reset critic output layers and run the same recalibration

After that, add visited-state teacher relabeling with decayed BC if the actor
still drifts away from the teacher after handoff.

## Validation

Markdown-only report update. No Python validation was required.
