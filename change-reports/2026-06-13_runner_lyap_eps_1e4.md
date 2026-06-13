# Runner Lyapunov Epsilon 1e-4

Date: 2026-06-13

## Summary

Changed the Direct LMPC contraction epsilon used by the shared disturbance
runner from `1e-3` to `1e-4`.

This affects the six online TD3 disturbance runners and the two disturbance
baselines because they all route through `utils/online_disturbance_runner.py`:

- `OnlineTD3_LMPCPretrained_SafetyGate.py`
- `OnlineTD3_OFMPCPretrained_SafetyGate.py`
- `OnlineTD3_LMPCPretrained_NoSafetyGate.py`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`
- `OnlineTD3_ColdStart_SafetyGate.py`
- `OnlineTD3_ColdStart_NoSafetyGate.py`
- `DirectLyapunovMPC_DisturbanceRunner.py`
- `OffsetFreeMPC_DisturbanceRunner.py`

## Scope

This is a runner-only experiment. The shared Direct LMPC selector defaults used
by LMPC pretraining remain at `lyap_eps = 1e-3`, so existing and future
pretraining workflows are not changed by this update.

New runner exports will record:

```json
"lyap_eps": 0.0001,
"lyap_eps_default": 0.0001,
"lyap_eps_override_reason": "runner-only stricter bounded-mixed Direct LMPC epsilon"
```

## Rationale

The previous online batch with `lyap_eps = 1e-3` fixed the handoff instability
after critic reset and actor-frozen handoff. This new batch tests whether a
stricter first-step practical contraction tolerance changes:

- safety-gate acceptance and fallback behavior,
- no-gate Direct LMPC monitor activation,
- Direct LMPC baseline tracking,
- OF-MPC baseline diagnostic safety-rate,
- online TD3 reward without fallback penalty.

## Validation

Ran:

```powershell
python -m py_compile utils\online_disturbance_runner.py OnlineTD3_LMPCPretrained_SafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_LMPCPretrained_NoSafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_ColdStart_NoSafetyGate.py DirectLyapunovMPC_DisturbanceRunner.py OffsetFreeMPC_DisturbanceRunner.py
```

Also checked that:

- `utils/online_disturbance_runner.py` now has `LYAP_EPS = 1e-4`.
- `utils/direct_lmpc_selector_defaults.py` and `utils/lmpc_td3_workflow.py`
  still keep the LMPC pretraining default at `1e-3`.
