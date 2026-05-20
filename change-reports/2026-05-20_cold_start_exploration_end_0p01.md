# Cold-Start Exploration End 0.01

## Summary

Updated the cold-start direct safety-gate RL script so exploration does not decay to zero or remain at the previous `0.1` floor. The cold-start full-RL Gaussian exploration now decays from `0.2` to `0.01`.

## Changes

- Set `DirectLyapunovSafetyGateRL_ColdStart.py` phase config:
  - `full_rl_exploration_std_start = 0.2`
  - `full_rl_exploration_std_end = 0.01`
- Set the cold-start TD3 agent legacy exploration endpoint:
  - `STD_END = 0.01`
- Updated the current RL report and HTML export so the documented active cold-start exploration schedule matches the script.

## Validation

- Ran `python -m py_compile DirectLyapunovSafetyGateRL_ColdStart.py`.
- Regenerated the self-contained HTML report.
- Confirmed the HTML still embeds 8 figures and has no broken local figure references.
- Ran `git diff --check` before commit.
