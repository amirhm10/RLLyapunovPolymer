# Online BC And Handoff Exploration Schedule

Date: 2026-06-12

## Summary

Updated all shared online TD3 disturbance runners to use cleaner behavior
cloning and gentler handoff exploration:

- pretrained BC now executes the clean teacher action with no Gaussian noise
- cold-start BC executes teacher actions with tiny Gaussian noise, `0.005`
- handoff uses a separate tiny policy-side Gaussian schedule
- full-RL Gaussian exploration starts after handoff, not at the BC boundary

The six online TD3 root runners inherit this through
`utils/online_disturbance_runner.py`. Direct LMPC and OF-MPC baseline runners are
unchanged.

## New Defaults

| Runner family | BC std | Handoff std start | Handoff std end | Full RL std start | Full RL std end |
|---|---:|---:|---:|---:|---:|
| pretrained | 0.000 | 0.000 | 0.005 | 0.020 | 0.005 |
| cold-start | 0.005 | 0.000 | 0.010 | 0.100 | 0.005 |

During handoff, the clean teacher action is blended with the policy candidate.
Only the policy candidate receives the small handoff noise; no additional noise
is added after blending.

## Files Changed

- `utils/online_disturbance_runner.py`
- `Simulation/run_rl_lyapunov.py`
- `report/online_disturbance_runner_algorithm_audit_2026-06-10.md`

## Validation

```powershell
C:\Users\hamediaa\.conda\envs\rl-env\python.exe -m py_compile utils\online_disturbance_runner.py Simulation\run_rl_lyapunov.py OnlineTD3_LMPCPretrained_SafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_LMPCPretrained_NoSafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_ColdStart_NoSafetyGate.py
```

Also ran a focused phase-config check confirming:

- pretrained BC: `bc_behavior_noise="none"`, sigma disabled
- cold-start BC: `bc_behavior_noise="gaussian"`, sigma `0.005`
- handoff first step: teacher weight `1.0`, policy weight `0.0`, sigma `0.0`
- handoff last step: teacher weight `0.0`, policy weight `1.0`, tiny handoff sigma
- first post-handoff full-RL step: full exploration sigma restored
