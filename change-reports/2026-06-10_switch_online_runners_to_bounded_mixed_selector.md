# Change Report: Switch Online Disturbance Runners To Bounded Mixed Selector

Date: 2026-06-10

## Summary

Updated the shared disturbance runner implementation so the six online TD3 runners plus the Direct LMPC and OF-MPC disturbance baselines use the previous bounded Direct LMPC target selector.

The online safety gate, no-gate Direct LMPC monitor diagnostics, Direct LMPC baseline, and OF-MPC baseline diagnostics now use:

```python
target_mode = "bounded"
target_selector_variant = "bounded_mixed_u0p1_x0p1"
target_config = {
    "u_ref_weight": 0.1,
    "x_ref_weight": 0.1,
}
rho_lyap = 0.99
lyap_eps = 1e-3
lyap_tol = 1e-10
slack_penalty = 1e6
```

## Scope

- Changed only the shared online/baseline disturbance runner path in `utils/online_disturbance_runner.py`.
- Kept root runner entrypoints unchanged.
- Kept offline LMPC and OF-MPC pretraining checkpoint generation unchanged.
- Added run-summary metadata noting that pretrained checkpoint loading is unchanged while the online Direct LMPC gate/diagnostic selector now uses the bounded mixed variant.

## Expected Interpretation

New result bundles from the eight disturbance runners should be interpreted as a bounded-mixed-selector ablation relative to the June 10 governed-reference batch. Existing governed-reference result bundles remain historical and should not be mixed with the new selector runs unless the selector variant is reported.

## Validation

Completed static validation:

```powershell
python -m py_compile utils/online_disturbance_runner.py Simulation/run_rl_lyapunov.py
python -m py_compile OnlineTD3_LMPCPretrained_SafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py OnlineTD3_LMPCPretrained_NoSafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_ColdStart_NoSafetyGate.py DirectLyapunovMPC_DisturbanceRunner.py OffsetFreeMPC_DisturbanceRunner.py
```

Completed smoke validation with `C:\Users\hamediaa\.conda\envs\rl-env\python.exe`:

```powershell
python OnlineTD3_LMPCPretrained_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
python OnlineTD3_OFMPCPretrained_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
python OnlineTD3_LMPCPretrained_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
python OnlineTD3_OFMPCPretrained_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
python OnlineTD3_ColdStart_SafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
python OnlineTD3_ColdStart_NoSafetyGate.py --episodes 1 --set-points-len 5 --no-save-plots
python DirectLyapunovMPC_DisturbanceRunner.py --episodes 1 --set-points-len 5 --no-save-plots
python OffsetFreeMPC_DisturbanceRunner.py --episodes 1 --set-points-len 5 --no-save-plots
```

Latest smoke `run_summary.json` files confirmed:

- `target_mode="bounded"`
- `target_selector_variant="bounded_mixed_u0p1_x0p1"`
- `target_config={"u_ref_weight": 0.1, "x_ref_weight": 0.1}`
- `rho_lyap=0.99`
- `lyap_eps=0.001`
- `lyap_tol=1e-10`
- `slack_penalty=1e6`
