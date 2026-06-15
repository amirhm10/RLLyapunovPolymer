# Set GART dx_s Bound To Absolute 0.1

## Summary

Changed the selected GART raw controller from the previous `dx_rate_scale = 5` multiplier to an explicit absolute target-state rate bound:

$$
|x_{s,i}(k)-x_{s,i}(k-1)| \le 0.1,\quad i=1,\ldots,7.
$$

## Implementation

- Added `dx_s_max_abs` support to `make_gart_target_config(...)`.
- A scalar `dx_s_max_abs` is expanded into a length-7 component-wise vector.
- A length-7 `dx_s_max_abs` vector can also be passed later if state-specific bounds are needed.
- Updated the selected root runner case to:
  `gart_target_raw_dxabs0p1_headroom_0p01_dy2_no_umid`.
- Kept the rest of the selected configuration unchanged:
  - raw objective
  - hard contraction
  - `eps = 1e-4`
  - `dy_rate_scale = 2`
  - `input_headroom_frac = 0.01`
  - no `u_mid` tie-breaker
  - no $x_s$ or $y_s$ smoothing

## Interpretation

The implementation is component-wise. The scalar `0.1` is not a single scalar penalty; it is converted to:

$$
dx_{s,\max} = [0.1,\ldots,0.1].
$$

This is equivalent to an infinity-norm bound only because all components share the same value:

$$
\|x_s(k)-x_s(k-1)\|_\infty \le 0.1.
$$

If the uniform value is too restrictive for some physical state coordinates and too loose for others, the next step should be a state-specific vector rather than returning to the large learned multiplier.
