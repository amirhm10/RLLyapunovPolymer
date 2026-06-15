# GART Final Configuration Default Alignment

## Summary

Aligned low-level GART target defaults and final-runner overrides with the selected final method.

## Changes

- Changed raw `GARTTargetConfig` dataclass defaults to the final proof values:
  - `rho = 0.98`
  - `eps = 1.0e-3`
- Added `du_s_max_abs` / `du_s_max_override` support to `make_gart_target_config(...)`.
- Made the final runner target-input motion bound explicit:

$$
d_{u_s}=[0.998,\;0.740].
$$

This avoids relying on the discovery fallback `0.05 * u_width` for the final published configuration.

## Validation

- Added tests for final raw dataclass defaults.
- Added tests for scalar and component-wise `du_s_max_abs` overrides.
- Ran low-cost compile and direct config-builder assertions.
