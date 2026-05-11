# Direct Notebooks Shaped Reward

## Why
- The direct Lyapunov study notebooks were still using the one-step quadratic reward.
- The repository already contains the newer shaped relative-band reward used in the main RL training notebooks.
- This change aligns the three direct notebook entrypoints with that shaped reward design.

## What Changed
- Updated [`DirectLyapunovSafetyGateRL_Pretrained.ipynb`](../DirectLyapunovSafetyGateRL_Pretrained.ipynb) to import `make_reward_fn_relative_QR` and build the shaped reward instead of `make_reward_fn_mpc_quadratic`.
- Updated [`DirectLyapunovSafetyGateRL_ColdStart.ipynb`](../DirectLyapunovSafetyGateRL_ColdStart.ipynb) the same way.
- Updated [`DirectLyapunovMPC_FrozenOutputDisturbance.ipynb`](../DirectLyapunovMPC_FrozenOutputDisturbance.ipynb) the same way.

## Reward Settings Used
- `k_rel = [0.003, 0.0003]`
- `band_floor_phys = [0.006, 0.07]`
- `tau_frac = 0.7`
- `gamma_out = 0.5`
- `gamma_in = 0.5`
- `beta = 7.0`
- `gate = "geom"`
- `lam_in = 1.0`
- `bonus_kind = "exp"`
- `bonus_k = 12.0`
- `bonus_p = 0.6`
- `bonus_c = 20.0`

These match the shaped reward configuration already used in the other polymer RL notebooks, while preserving each direct notebook's local `Qy_diag` and `Rdu_diag`.

## Validation
- Parsed all three notebooks as JSON after editing.
- Compiled every code cell in the three edited notebooks with Python `compile(...)`.
- Did not run the notebooks end-to-end.

## Notes
- `reward_config` is still exported through the existing debug bundles, now containing the shaped reward parameters.
- No controller backend Python modules were changed in this update.
