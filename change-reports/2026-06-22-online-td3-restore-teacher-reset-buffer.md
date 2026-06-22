# Online TD3 Cold-Start Revert, Critic Reset, And Buffer Increase

## Context

The cold-start no-safety-gate diagnostic run showed a large increase in would-have-activated GART gate counts after the fast cold no-gate path removed the GART teacher/critic warmup phases. That made the cold-start comparison harder to interpret.

## Changes

- Restored `OnlineTD3_ColdStart_NoSafetyGate.py` to the GART teacher critic-warmup configuration through `default_noisy_teacher_critic_warmup_overrides(...)`.
- Kept `PROJECTION_BACKEND = "mpc_only_diagnostic"` for cold no-gate runs so the diagnostic would-have-activated safety count remains available.
- Set `RESET_PRETRAINED_CRITIC = True` in both OF-MPC-pretrained online runners:
  - `OnlineTD3_OFMPCPretrained_SafetyGate.py`
  - `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`
- Set the shared online TD3 default `DEFAULT_RESET_PRETRAINED_CRITIC = True`.
- Increased online replay capacity from `40000` to `80000`.

## Validation

- `python -m py_compile OnlineTD3_ColdStart_NoSafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py utils/online_disturbance_runner.py`
- Import/config assertion check confirmed:
  - online buffer capacity is `80000`
  - default pretrained critic reset is enabled
  - both OF-MPC pretrained runners reset critic
  - cold no-gate still uses `mpc_only_diagnostic`
  - cold no-gate uses GART-LMPC teacher critic warmup and handoff phases
