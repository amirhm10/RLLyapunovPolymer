# BC Teacher Execution And No-Gate Supervisor Update

Date: 2026-06-10

## Summary

Updated the online TD3 disturbance runners so behavior cloning executes the selected teacher action with Gaussian exploration while the actor imitates the clean teacher action. No-safety-gate runners now use OF-MPC as their online BC/handoff supervisor so Direct LMPC remains diagnostic only.

## Changes

- Changed the shared BC phase config from `policy_with_lmpc_teacher_demo` to `bc_behavior_source = teacher_source`.
- Set no-safety-gate online teacher sources to OF-MPC:
  - `OnlineTD3_LMPCPretrained_NoSafetyGate`
  - `OnlineTD3_OFMPCPretrained_NoSafetyGate`
  - `OnlineTD3_ColdStart_NoSafetyGate`
- Kept safety-gate teacher sources unchanged:
  - Direct LMPC for LMPC-pretrained safety and cold-start safety
  - OF-MPC for OF-MPC-pretrained safety, with Direct LMPC still used as safety gate
- Updated `report/online_disturbance_runner_algorithm_audit_2026-06-10.md` to describe teacher-executed BC, critic/actor update roles, OF-MPC no-gate supervision, and Direct LMPC diagnostic-only behavior.

## Validation

- Static validation:
  - `python -m py_compile utils/online_disturbance_runner.py Simulation/run_rl_lyapunov.py`
- Config checks:
  - direct teacher BC config uses `bc_behavior_source="direct_lyapunov_mpc"`
  - OF-MPC teacher BC config uses `bc_behavior_source="offset_free_mpc"`
  - no-gate presets report `teacher_source="offset_free_mpc"`
- Smoke checks:
  - `python OnlineTD3_LMPCPretrained_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots`
  - `python OnlineTD3_ColdStart_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots`
  - `python OnlineTD3_LMPCPretrained_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots`
  - `python OnlineTD3_OFMPCPretrained_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots`
- Artifact checks confirmed:
  - no-gate step tables show `offset_free_mpc_gaussian`
  - safety-gate step tables show the configured teacher Gaussian behavior
  - no-gate actual intervention rates are `0.0`
  - no-gate diagnostic safety rates are populated
  - safety-gate fallback penalties remain active when actual intervention occurs
