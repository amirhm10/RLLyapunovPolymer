# Use GART For OF-MPC-Pretrained Online TD3

## Summary

The OF-MPC-pretrained online TD3 runners now keep the OF-MPC pretrained actor
checkpoint as initialization while switching online teacher, target selection,
and safety fallback behavior to the GART path.

## Changes

- `OnlineTD3_OFMPCPretrained_SafetyGate` now uses:
  - `pretrain_source="of_mpc"`
  - `teacher_source="gart_lmpc"`
  - `direct_target_mode="gart"`
  - `fallback_controller="gart_lmpc"`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate` now uses:
  - `pretrain_source="of_mpc"`
  - `teacher_source="gart_lmpc"`
  - `direct_target_mode="gart"`
  - `fallback_controller="none"`
- LMPC-pretrained presets are intentionally unchanged.
- Stale GART-only text was generalized so OF-MPC-pretrained GART runs are not
  described as cold-start or Direct-LMPC-controlled.

## Validation

Completed checks:

```powershell
python -m py_compile utils/online_disturbance_runner.py Simulation/run_rl_lyapunov.py Lyapunov/safety_debug.py
& "C:\Users\hamed\miniconda3\envs\rlenv\python.exe" -c "<assert OF-MPC-pretrained presets use GART and LMPC-pretrained presets are unchanged>"
& "C:\Users\hamed\miniconda3\envs\rlenv\python.exe" OnlineTD3_OFMPCPretrained_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
& "C:\Users\hamed\miniconda3\envs\rlenv\python.exe" OnlineTD3_OFMPCPretrained_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
```

Smoke-run summaries and first `step_table.csv` rows confirmed:

- `pretrain_source="of_mpc"`
- `actor_loaded_from_checkpoint=True`
- `target_mode="gart"`
- `teacher_source="gart_lmpc"`
- safety-gate run: `fallback_controller="gart_lmpc"` and
  `safety_gate_active=True`
- no-safety run: `fallback_controller="none"` and
  `safety_gate_active=False`
- GART diagnostics columns were present:
  `target_rejection_reason`, `target_usable_for_lmpc`,
  `contraction_probe_margin`, and `governor_alpha`

`pytest tests/test_gart_target.py` was not run because `pytest` is not installed
in the default shell or the `rlenv` environment.
