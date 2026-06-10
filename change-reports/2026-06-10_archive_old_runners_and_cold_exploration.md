# Archive Old Runners And Tune Cold Exploration

## Summary

- Moved the superseded compatibility safety-gate runner entrypoints into `archive/`.
- Changed cold-start online TD3 exploration start from `0.20` to `0.10`.
- Updated the online disturbance runner algorithm audit so the documented BC and full-RL Gaussian exploration schedule matches the shared runner config.

## Files Changed

- `archive/DirectLyapunovSafetyGateRL_Pretrained.py`
- `archive/DirectLyapunovSafetyGateRL_ColdStart.py`
- `utils/online_disturbance_runner.py`
- `report/online_disturbance_runner_algorithm_audit_2026-06-10.md`
- `AGENTS.md`

## Validation

- Passed: `python -m py_compile utils/online_disturbance_runner.py archive/DirectLyapunovSafetyGateRL_Pretrained.py archive/DirectLyapunovSafetyGateRL_ColdStart.py OnlineTD3_ColdStart_NoSafetyGate.py OnlineTD3_ColdStart_SafetyGate.py`
- Passed: direct config check for pretrained and cold-start exploration schedule values:
  - pretrained: BC std `0.02`, full-RL start `0.02`, full-RL end `0.005`
  - cold-start: BC std `0.10`, full-RL start `0.10`, full-RL end `0.005`
