# 2026-05-11 RL Notebooks rho99 raw ysp weights 0p5

## What changed

Updated both direct safety-gate RL notebooks:

- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`

The configuration changes are:

- `rho_lyap = 0.99`
- `direct_tracking_use_target_output = False`
- `case_specs = direct_four_method_case_specs(anchor_weight=0.5, smoothness_weight=0.5)`

## Why

The temporary switch to tracking `y_s` produced catastrophic training behavior in the current RL runs. This change reverts the direct tracking stage back to raw `y_sp` and applies the requested stronger regularization weights for:

- previous-input anchoring: `u_ref_weight = 0.5`
- previous-state smoothing: `x_ref_weight = 0.5`

while also moving the Lyapunov contraction factor to `rho_lyap = 0.99`.

## Validation

- Both notebooks remain valid JSON.
- Both notebooks now contain:
  - `rho_lyap = 0.99`
  - `direct_tracking_use_target_output=False`
  - `direct_four_method_case_specs(anchor_weight=0.5, smoothness_weight=0.5)`
