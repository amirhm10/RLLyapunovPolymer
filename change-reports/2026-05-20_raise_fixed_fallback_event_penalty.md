# Raise Fixed Fallback Event Penalty

## Summary

Raised the fixed safety-gate fallback event penalty for the next cold-start and pretrained RL experiment. The previous latest analyzed run used `fallback_event_penalty = 2.0`; the next-run active scripts now use `fallback_event_penalty = 10.0`.

## Rationale

The latest report showed that the stronger fallback penalty was visible, but the fixed event component was still much smaller than the correction-gap component:

- Cold RL fixed event mean was about `0.027` per step with `fallback_event_penalty = 2.0`.
- Pretrained RL fixed event mean was about `0.068` per step with `fallback_event_penalty = 2.0`.

Increasing the fixed event penalty to `10.0` should make frequent small fallback events more expensive while keeping the correction-gap term unchanged.

## Changes

- Updated `DirectLyapunovSafetyGateRL_ColdStart.py`:
  - `fallback_event_penalty = 2.0` to `fallback_event_penalty = 10.0`.
- Updated `DirectLyapunovSafetyGateRL_Pretrained.py`:
  - `fallback_event_penalty = 2.0` to `fallback_event_penalty = 10.0`.
- Updated the current RL report to distinguish:
  - analyzed latest run: `fallback_event_penalty = 2.0`
  - next experiment active scripts: `fallback_event_penalty = 10.0`

## Validation

- Ran `python -m py_compile` on both active RL scripts.
- Regenerated the self-contained HTML report.
- Confirmed the HTML still embeds figures with `data:image/...;base64`.
- Ran `git diff --check` before commit.
