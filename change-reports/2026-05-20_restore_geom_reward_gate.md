# Restore Geometric Reward Gate

## Summary

Changed the active cold-start and pretrained direct safety-gate RL reward setup from the stricter product near-setpoint gate back to the earlier geometric gate.

## Changes

- Updated `DirectLyapunovSafetyGateRL_ColdStart.py`:
  - `gate="prod"` to `gate="geom"` in `make_reward_fn_relative_QR(...)`.
- Updated `DirectLyapunovSafetyGateRL_Pretrained.py`:
  - `gate="prod"` to `gate="geom"` in `make_reward_fn_relative_QR(...)`.
- Updated the current RL report and HTML export so the active reward math and next-run defaults show the geometric gate:

$$
w_{\rm in}=\left(\prod_{i=1}^{n_y}s_i\right)^{1/n_y}.
$$

## Validation

- Ran `python -m py_compile` on both active RL scripts.
- Regenerated the self-contained HTML report.
- Confirmed the HTML still embeds 8 figures and has no broken local figure references.
- Ran `git diff --check` before commit.
