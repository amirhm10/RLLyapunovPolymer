# Direct Nominal Mode Audit Guard

## Summary

Verified and hardened the direct Lyapunov MPC rollout so nominal runs are
explicitly enforced and auditable from exported results.

## Findings

- The notebook already passes `plant_mode = "nominal"` into
  `run_direct_output_disturbance_lyapunov_mpc(...)`.
- In the rollout, disturbance updates are only applied under
  `mode == "disturb"`.
- With `mode = "nominal"`, neither the pre-step nor post-step disturbance hook
  updates `Qi`, `Qs`, or `hA`.

## Changes

- Added validation that rollout `mode` must be either `nominal` or `disturb`.
- Reset `system.Qi`, `system.Qs`, and `system.hA` to the passed nominal values
  at run entry, including after the entry-state reset.
- Added `plant_mode`, `disturbance_after_step`, nominal plant parameters, and
  final plant parameters to:
  - raw rollout results,
  - debug bundles,
  - summary records,
  - comparison records,
  - per-step records.

## Validation

- Ran `python -m py_compile Lyapunov/direct_lyapunov_mpc.py`.
