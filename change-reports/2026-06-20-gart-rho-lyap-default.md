# GART Rho Lyap Default

## Objective

Set the shared GART Lyapunov contraction factor default to `rho_lyap = 0.99`
so the active online TD3 runners inherit the looser final GART-LMPC setting.

## Files Changed

- `utils/gart_defaults.py`

## Change

`GART_FINAL_RHO_LYAP` now defaults to `0.99` instead of `0.98`.

The root online TD3 runners set:

```python
RHO_LYAP = GART_FINAL_RHO_LYAP
```

so this applies to:

- `OnlineTD3_ColdStart_NoSafetyGate.py`
- `OnlineTD3_ColdStart_SafetyGate.py`
- `OnlineTD3_OFMPCPretrained_NoSafetyGate.py`
- `OnlineTD3_OFMPCPretrained_SafetyGate.py`

## Mathematical Interpretation

The first-step Lyapunov gate uses:

$$
V_{k+1} \le \rho_{\mathrm{lyap}} V_k + \varepsilon_{\mathrm{lyap}}.
$$

Raising $\rho_{\mathrm{lyap}}$ from `0.98` to `0.99` loosens the allowed
one-step contraction while preserving $\rho_{\mathrm{lyap}} < 1$.

## Validation

Passed:

```powershell
python -m py_compile utils/gart_defaults.py OnlineTD3_ColdStart_NoSafetyGate.py OnlineTD3_ColdStart_SafetyGate.py OnlineTD3_OFMPCPretrained_NoSafetyGate.py OnlineTD3_OFMPCPretrained_SafetyGate.py utils/online_disturbance_runner.py
```

The successful local run redirected `PYTHONPYCACHEPREFIX` to a Windows temp
directory to avoid the known OneDrive `.validation-pyc` pycache permission issue.
