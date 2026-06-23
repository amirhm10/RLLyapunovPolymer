# Relax Online Reward Band

## Objective

Slightly widen the near-setpoint reward band so the online TD3 and GART-LMPC runs receive positive shaped reward over a more physically reasonable neighborhood of the setpoint.

## Changes

- Updated the reward band in the online TD3 reward builder:
  - `band_floor_phys = [0.006, 0.08]`
- Updated the GART-LMPC baseline reward builder to the same band.
- Kept `beta = 5.0` unchanged.

## Interpretation

The reward still uses:

$$
r = -(\ell_y + \ell_u) + b
$$

where the bonus term is gated by whether the tracking error is inside the physical band. The wider band makes the near-setpoint bonus active over a less restrictive region.

## Validation

- `python -X pycache_prefix=... -m py_compile utils/online_disturbance_runner.py experiments/run_gart_target_selector_study.py`
- Runtime config check in the `rl` conda environment confirmed:
  - TD3 band: `[0.006, 0.08]`
  - GART band: `[0.006, 0.08]`
  - `beta = 5.0`
  - perfect-tracking, zero-move step reward is approximately `0.32` for the current two setpoints.
