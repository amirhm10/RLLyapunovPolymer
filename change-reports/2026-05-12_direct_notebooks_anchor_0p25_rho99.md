# Direct Notebooks Anchor 0.25 and Rho 0.99

Date: 2026-05-12

## Summary

Updated the three direct notebooks to use `rho_lyap = 0.99` and anchor weights of `0.25`.

## Updated files

- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/DirectLyapunovSafetyGateRL_Pretrained.ipynb>)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/DirectLyapunovSafetyGateRL_ColdStart.ipynb>)
- [DirectLyapunovMPC_FourMethodDisturbance.ipynb](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/DirectLyapunovMPC_FourMethodDisturbance.ipynb>)

## Configuration changes

- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
  - kept `rho_lyap = 0.99`
  - changed `case_specs = direct_four_method_case_specs(anchor_weight=0.25, smoothness_weight=0.25)`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`
  - kept `rho_lyap = 0.99`
  - changed `case_specs = direct_four_method_case_specs(anchor_weight=0.25, smoothness_weight=0.25)`
- `DirectLyapunovMPC_FourMethodDisturbance.ipynb`
  - changed `rho_lyap` from `0.98` to `0.99`
  - changed `case_specs` from default four-method settings to `direct_four_method_case_specs(anchor_weight=0.25, smoothness_weight=0.25)`

## Validation

- Verified the new `rho_lyap` and anchor settings are present in all three notebooks.
- Parsed all three notebooks as JSON.
- Compiled every code cell in memory for all three notebooks.
