# Offset-Free MPC Steady-State Debug Summary

- Case: `disturbance`
- Steps analyzed: `1600`
- Configured solver mode: `auto`
- Least-squares used on `0` steps
- Requested-mode fallbacks: `0`
- Box analysis enabled: `True`

## Model Structure

| n | p | q | M_shape | G_shape | I_minus_A_invertible | M_square | G_square | rank_M | rank_G | classification |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 7.000000 | 2.000000 | 2.000000 | (9, 9) | (2, 2) | true | true | true | 9.000000 | 2.000000 | square |

## Global Linear Algebra Diagnostics

| cond_I_minus_A | cond_M | cond_G | smallest_sv_M | smallest_sv_G | configured_solver_mode |
| --- | --- | --- | --- | --- | --- |
| 33.619937 | 1205.832131 | 7.051874 | 0.001532 | 0.003256 | auto |

## Solver Mode Counts

| solver_mode | count |
| --- | --- |
| stacked_exact | 1600.000000 |

## Summary Statistics

| residual_dyn_norm_mean | residual_dyn_norm_std | residual_dyn_norm_max | residual_out_norm_mean | residual_out_norm_std | residual_out_norm_max | residual_total_norm_mean | residual_total_norm_std | residual_total_norm_max | u_applied_minus_u_s_norm_mean | u_applied_minus_u_s_norm_std | u_applied_minus_u_s_norm_max | y_current_minus_y_s_norm_mean | y_current_minus_y_s_norm_std | y_current_minus_y_s_norm_max | xhat_minus_x_s_norm_mean | xhat_minus_x_s_norm_std | xhat_minus_x_s_norm_max | rhs_output_norm_mean | rhs_output_norm_std | rhs_output_norm_max | u_s_dev_norm_mean | u_s_dev_norm_std | u_s_dev_norm_max | x_s_norm_mean | x_s_norm_std | x_s_norm_max | reduced_rhs_exact_residual_norm_mean | reduced_rhs_exact_residual_norm_std | reduced_rhs_exact_residual_norm_max | reduced_rhs_bounded_residual_norm_mean | reduced_rhs_bounded_residual_norm_std | reduced_rhs_bounded_residual_norm_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4.9286e-17 | 1.6938e-16 | 3.5534e-15 | 1.6974e-16 | 4.6449e-16 | 6.2172e-15 | 2.2659e-16 | 5.1868e-16 | 6.4643e-15 | 143.949669 | 276.861383 | 1466.304182 | 0.526607 | 0.963233 | 6.281438 | 41.832274 | 73.325312 | 493.301054 | 0.635495 | 1.179574 | 6.269257 | 145.970064 | 275.886106 | 1468.441281 | 46.387412 | 89.553563 | 493.939657 | 9.7488e-17 | 2.4281e-16 | 2.3915e-15 | 0.549651 | 1.150781 | 6.099560 |

## Sampled Per-Step Diagnostics

| k | residual_dyn_norm | residual_out_norm | residual_total_norm | exact_solution | used_lstsq | solver_mode | u_applied_minus_u_s_norm | y_current_minus_y_s_norm | xhat_minus_x_s_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 9.3675e-17 | 8.8818e-16 | 9.4076e-16 | true | false | stacked_exact | 823.756320 | 2.798133 | 274.896971 |
| 20.000000 | 1.4572e-16 | 4.4409e-16 | 6.2413e-16 | true | false | stacked_exact | 604.412565 | 1.655747 | 119.852576 |
| 40.000000 | 6.9389e-18 | 6.6613e-16 | 7.6638e-16 | true | false | stacked_exact | 373.610619 | 1.594822 | 118.217141 |
| 60.000000 | 1.3878e-17 | 2.2204e-16 | 3.8653e-16 | true | false | stacked_exact | 294.313336 | 1.171150 | 100.857098 |
| 80.000000 | 2.1684e-17 | 1.1102e-16 | 2.0355e-16 | true | false | stacked_exact | 197.859101 | 0.780401 | 64.744730 |
| 100.000000 | 1.1224e-16 | 2.2204e-16 | 2.6038e-16 | true | false | stacked_exact | 130.725072 | 0.521941 | 43.309458 |
| 120.000000 | 2.9924e-17 | 0.000000 | 3.9944e-17 | true | false | stacked_exact | 90.486912 | 0.356881 | 29.644270 |
| 140.000000 | 1.7347e-18 | 0.000000 | 3.5586e-17 | true | false | stacked_exact | 64.463145 | 0.256625 | 20.555836 |
| 160.000000 | 1.3878e-17 | 5.5511e-17 | 6.6578e-17 | true | false | stacked_exact | 48.021417 | 0.196899 | 14.977511 |
| 180.000000 | 2.1684e-19 | 0.000000 | 2.6072e-17 | true | false | stacked_exact | 37.896776 | 0.161192 | 11.608685 |
| 200.000000 | 3.6863e-18 | 0.000000 | 2.0007e-17 | true | false | stacked_exact | 31.665414 | 0.139880 | 9.582825 |
| 220.000000 | 2.8189e-18 | 1.1102e-16 | 1.1395e-16 | true | false | stacked_exact | 27.854934 | 0.127228 | 8.386223 |
| 240.000000 | 3.1442e-18 | 5.7220e-17 | 6.3868e-17 | true | false | stacked_exact | 25.561068 | 0.119817 | 7.689955 |
| 260.000000 | 5.5511e-17 | 5.5511e-17 | 6.9197e-17 | true | false | stacked_exact | 24.212046 | 0.115626 | 7.292187 |
| 280.000000 | 5.5613e-17 | 0.000000 | 6.6739e-17 | true | false | stacked_exact | 23.456180 | 0.113435 | 7.075696 |
| 300.000000 | 5.5524e-17 | 5.5511e-17 | 8.1313e-17 | true | false | stacked_exact | 23.071464 | 0.112502 | 6.969333 |
| 320.000000 | 3.3610e-18 | 5.5511e-17 | 6.1891e-17 | true | false | stacked_exact | 22.924098 | 0.112371 | 6.931533 |
| 340.000000 | 1.6263e-18 | 2.7756e-17 | 3.1460e-17 | true | false | stacked_exact | 22.926244 | 0.112753 | 6.937337 |
| 360.000000 | 8.6736e-19 | 0.000000 | 1.6381e-17 | true | false | stacked_exact | 23.027912 | 0.113479 | 6.971163 |
| 380.000000 | 6.3968e-18 | 2.7756e-17 | 3.7644e-17 | true | false | stacked_exact | 23.194839 | 0.114442 | 7.023719 |
| 400.000000 | 2.2898e-16 | 9.9301e-16 | 1.2943e-15 | true | false | stacked_exact | 1376.770671 | 6.281438 | 480.916170 |
| 420.000000 | 1.9860e-15 | 0.000000 | 1.3927e-15 | true | false | stacked_exact | 989.360131 | 1.824722 | 109.491477 |
| 440.000000 | 1.3878e-17 | 2.2204e-16 | 2.2342e-16 | true | false | stacked_exact | 117.464056 | 1.939557 | 152.298680 |
| 460.000000 | 2.2209e-16 | 4.4409e-16 | 5.2283e-16 | true | false | stacked_exact | 287.540007 | 0.740626 | 115.131319 |
| 480.000000 | 3.7297e-17 | 2.2204e-16 | 2.2731e-16 | true | false | stacked_exact | 208.116441 | 0.631187 | 46.213709 |
| 500.000000 | 2.2302e-16 | 0.000000 | 1.1836e-16 | true | false | stacked_exact | 90.786528 | 0.454393 | 37.583194 |
| 520.000000 | 2.2204e-16 | 0.000000 | 2.8633e-16 | true | false | stacked_exact | 87.882574 | 0.289215 | 29.676537 |
| 540.000000 | 8.6736e-19 | 0.000000 | 1.2245e-17 | true | false | stacked_exact | 56.142438 | 0.205474 | 16.839169 |
| 560.000000 | 1.6263e-18 | 6.2063e-17 | 6.3820e-17 | true | false | stacked_exact | 32.291266 | 0.125406 | 12.081888 |
| 580.000000 | 3.2526e-19 | 5.5511e-17 | 5.6621e-17 | true | false | stacked_exact | 21.503031 | 0.066854 | 7.672777 |
| 600.000000 | 5.4210e-20 | 0.000000 | 4.9906e-18 | true | false | stacked_exact | 9.874162 | 0.026223 | 4.213378 |
| 620.000000 | 8.6739e-19 | 0.000000 | 6.1433e-18 | true | false | stacked_exact | 2.162213 | 0.009358 | 2.429401 |
| 640.000000 | 1.9516e-18 | 1.7347e-18 | 3.9610e-18 | true | false | stacked_exact | 3.952102 | 0.031875 | 2.036852 |
| 660.000000 | 2.2226e-18 | 1.7347e-18 | 3.0920e-18 | true | false | stacked_exact | 8.357786 | 0.049817 | 2.833017 |
| 680.000000 | 1.8431e-18 | 3.4694e-18 | 3.9366e-18 | true | false | stacked_exact | 11.596746 | 0.063426 | 3.624161 |
| 700.000000 | 7.2382e-18 | 2.7972e-17 | 2.8572e-17 | true | false | stacked_exact | 14.031899 | 0.073684 | 4.310617 |
| 720.000000 | 3.0358e-18 | 2.7756e-17 | 2.8500e-17 | true | false | stacked_exact | 15.926422 | 0.081578 | 4.856870 |
| 740.000000 | 3.3610e-18 | 0.000000 | 7.1566e-18 | true | false | stacked_exact | 17.369846 | 0.087735 | 5.280042 |
| 760.000000 | 2.1684e-18 | 2.7756e-17 | 2.8110e-17 | true | false | stacked_exact | 18.506964 | 0.092617 | 5.619683 |
| 780.000000 | 2.7105e-18 | 2.7756e-17 | 2.7903e-17 | true | false | stacked_exact | 19.418374 | 0.096587 | 5.891831 |
| 800.000000 | 9.0206e-17 | 8.8818e-16 | 1.2295e-15 | true | false | stacked_exact | 1418.376391 | 6.270481 | 493.301054 |
| 820.000000 | 9.7145e-17 | 1.7764e-15 | 1.8367e-15 | true | false | stacked_exact | 752.392770 | 1.077999 | 119.129130 |
| 840.000000 | 6.0715e-17 | 0.000000 | 2.9767e-16 | true | false | stacked_exact | 374.381148 | 1.570672 | 129.564867 |
| 860.000000 | 8.8861e-16 | 8.8818e-16 | 1.5599e-15 | true | false | stacked_exact | 341.319860 | 1.201304 | 116.433890 |
| 880.000000 | 4.4411e-16 | 4.4409e-16 | 6.3063e-16 | true | false | stacked_exact | 233.675832 | 0.873067 | 72.661131 |
| 900.000000 | 4.3368e-18 | 1.5701e-16 | 2.5701e-16 | true | false | stacked_exact | 159.871614 | 0.636801 | 51.801580 |
| 920.000000 | 1.7781e-17 | 1.1102e-16 | 1.2159e-16 | true | false | stacked_exact | 115.788668 | 0.461096 | 37.464072 |
| 940.000000 | 9.1073e-18 | 0.000000 | 5.4839e-17 | true | false | stacked_exact | 83.595257 | 0.336971 | 26.784239 |
| 960.000000 | 6.5052e-18 | 1.1102e-16 | 1.1970e-16 | true | false | stacked_exact | 60.696715 | 0.248097 | 19.472276 |
| 980.000000 | 0.000000 | 1.1102e-16 | 1.2561e-16 | true | false | stacked_exact | 44.283627 | 0.182631 | 14.237478 |
| 1000.000000 | 1.7347e-18 | 5.5511e-17 | 6.2146e-17 | true | false | stacked_exact | 32.143514 | 0.133435 | 10.376573 |
| 1020.000000 | 7.3726e-18 | 0.000000 | 2.2658e-17 | true | false | stacked_exact | 23.012924 | 0.095834 | 7.488181 |
| 1040.000000 | 4.3368e-19 | 2.7756e-17 | 3.7802e-17 | true | false | stacked_exact | 16.045638 | 0.066710 | 5.292387 |
| 1060.000000 | 3.1086e-17 | 2.8610e-17 | 4.8279e-17 | true | false | stacked_exact | 10.659550 | 0.043943 | 3.605455 |
| 1080.000000 | 1.6263e-19 | 2.7756e-17 | 2.9247e-17 | true | false | stacked_exact | 6.454970 | 0.026060 | 2.308276 |
| 1100.000000 | 2.9273e-18 | 1.5516e-17 | 2.1517e-17 | true | false | stacked_exact | 3.150365 | 0.012127 | 1.336661 |
| 1120.000000 | 7.0473e-19 | 1.3878e-17 | 1.4431e-17 | true | false | stacked_exact | 0.580338 | 0.003848 | 0.735773 |
| 1140.000000 | 1.3893e-17 | 6.9389e-18 | 1.0635e-17 | true | false | stacked_exact | 1.583297 | 0.009726 | 0.730427 |
| 1160.000000 | 6.7763e-20 | 1.7347e-18 | 2.2026e-18 | true | false | stacked_exact | 3.232591 | 0.016638 | 1.072643 |
| 1180.000000 | 3.9302e-19 | 3.4694e-18 | 4.1546e-18 | true | false | stacked_exact | 4.555778 | 0.022329 | 1.428612 |
| 1200.000000 | 3.4694e-17 | 4.5288e-15 | 4.5956e-15 | true | false | stacked_exact | 1404.480507 | 6.275866 | 489.158247 |
| 1220.000000 | 1.7765e-15 | 8.8818e-16 | 1.9910e-15 | true | false | stacked_exact | 999.620124 | 1.853046 | 114.336557 |
| 1240.000000 | 3.2960e-17 | 2.2888e-16 | 2.3435e-16 | true | false | stacked_exact | 111.191875 | 1.953166 | 155.168158 |
| 1260.000000 | 2.2207e-16 | 4.4409e-16 | 6.5779e-16 | true | false | stacked_exact | 304.463635 | 0.802390 | 117.594342 |
| 1280.000000 | 6.9389e-18 | 2.4825e-16 | 2.8661e-16 | true | false | stacked_exact | 226.059797 | 0.736243 | 50.024664 |
| 1300.000000 | 2.2213e-16 | 1.1102e-16 | 2.4584e-16 | true | false | stacked_exact | 116.487957 | 0.575617 | 43.219315 |
| 1320.000000 | 1.3010e-18 | 2.2204e-16 | 2.2831e-16 | true | false | stacked_exact | 120.110550 | 0.449927 | 38.422687 |
| 1340.000000 | 1.1276e-17 | 0.000000 | 6.6394e-17 | true | false | stacked_exact | 90.102921 | 0.377862 | 26.350685 |
| 1360.000000 | 5.6400e-17 | 0.000000 | 8.2899e-17 | true | false | stacked_exact | 68.190865 | 0.301127 | 22.169597 |
| 1380.000000 | 1.4095e-17 | 1.1102e-16 | 1.2389e-16 | true | false | stacked_exact | 57.714647 | 0.244183 | 18.120565 |
| 1400.000000 | 4.7705e-18 | 5.5511e-17 | 6.0160e-17 | true | false | stacked_exact | 45.442766 | 0.198373 | 14.186224 |
| 1420.000000 | 3.4694e-18 | 5.7220e-17 | 6.6936e-17 | true | false | stacked_exact | 36.507068 | 0.159676 | 11.605710 |
| 1440.000000 | 5.5526e-17 | 5.5511e-17 | 8.1025e-17 | true | false | stacked_exact | 29.769380 | 0.129345 | 9.372623 |
| 1460.000000 | 5.5512e-17 | 2.7756e-17 | 7.3635e-17 | true | false | stacked_exact | 24.008491 | 0.104987 | 7.563272 |
| 1480.000000 | 1.3882e-17 | 2.7756e-17 | 3.1911e-17 | true | false | stacked_exact | 19.581310 | 0.085504 | 6.176974 |
| 1500.000000 | 5.9631e-19 | 0.000000 | 1.2580e-17 | true | false | stacked_exact | 16.079361 | 0.070167 | 5.054265 |
| 1520.000000 | 1.3967e-17 | 2.7756e-17 | 3.9542e-17 | true | false | stacked_exact | 13.284511 | 0.058052 | 4.170446 |
| 1540.000000 | 6.9693e-18 | 0.000000 | 8.4264e-18 | true | false | stacked_exact | 11.103689 | 0.048537 | 3.478418 |
| 1560.000000 | 4.6079e-19 | 0.000000 | 1.2113e-17 | true | false | stacked_exact | 9.388803 | 0.041095 | 2.932928 |
| 1580.000000 | 4.8789e-19 | 6.9389e-18 | 1.0560e-17 | true | false | stacked_exact | 8.044003 | 0.035280 | 2.506837 |

## Box-Constrained Analysis Summary

| pct_exact_bounded | pct_exact_unbounded_fallback_bounded_ls | pct_exact_unsolved_fallback_bounded_ls | pct_failed | pct_exact_solutions_inside_bounds | avg_exact_bound_violation_inf | max_exact_bound_violation_inf | avg_bounded_residual_norm | max_bounded_residual_norm | avg_us_exact_minus_us_bounded_inf | max_us_exact_minus_us_bounded_inf | avg_xs_exact_minus_xs_bounded_inf | max_xs_exact_minus_xs_bounded_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13.875000 | 86.125000 | 0.000000 | 0.000000 | 13.875000 | 120.136930 | 1234.945938 | 0.549651 | 6.099560 | 120.883877 | 1234.945938 | 30.423158 | 362.949918 |

## Box Solve Mode Counts

| solve_mode | count |
| --- | --- |
| exact_bounded | 222.000000 |
| exact_unbounded_fallback_bounded_ls | 1378.000000 |

## Per-Input Bound Activity

| input_index | fraction_lower_bound_active | fraction_upper_bound_active | average_exact_violation_below_lower | average_exact_violation_above_upper |
| --- | --- | --- | --- | --- |
| 0.000000 | 0.228750 | 0.476875 | 58.758810 | 61.353003 |
| 1.000000 | 0.179375 | 0.071250 | 30.195974 | 28.676621 |

## Box Event Table

| event_kind | event_anchor | k | solve_mode | exact_eq_residual_state_inf | exact_eq_residual_output_inf | bounded_residual_norm | dhat_delta_inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| setpoint_change | 400.000000 | 395.000000 | exact_unbounded_fallback_bounded_ls | 3.3610e-18 | 2.7756e-17 | 0.062409 | 9.5174e-05 |
| setpoint_change | 400.000000 | 396.000000 | exact_unbounded_fallback_bounded_ls | 6.3968e-18 | 2.7756e-17 | 0.062461 | 9.5327e-05 |
| setpoint_change | 400.000000 | 397.000000 | exact_unbounded_fallback_bounded_ls | 2.7105e-18 | 0.000000 | 0.062514 | 9.4731e-05 |
| setpoint_change | 400.000000 | 398.000000 | exact_unbounded_fallback_bounded_ls | 4.0115e-18 | 1.3878e-17 | 0.062567 | 9.5059e-05 |
| setpoint_change | 400.000000 | 399.000000 | exact_unbounded_fallback_bounded_ls | 4.9873e-18 | 2.7756e-17 | 0.062620 | 9.5765e-05 |
| setpoint_change | 400.000000 | 400.000000 | exact_unbounded_fallback_bounded_ls | 2.2898e-16 | 8.8818e-16 | 6.094292 | 9.6132e-05 |
| setpoint_change | 400.000000 | 401.000000 | exact_unbounded_fallback_bounded_ls | 2.5674e-16 | 2.6645e-15 | 6.094321 | 9.6323e-05 |
| setpoint_change | 400.000000 | 402.000000 | exact_unbounded_fallback_bounded_ls | 1.9429e-16 | 3.9968e-15 | 5.976994 | 0.196579 |
| setpoint_change | 400.000000 | 403.000000 | exact_unbounded_fallback_bounded_ls | 9.0206e-17 | 1.3323e-15 | 5.769154 | 0.361868 |
| setpoint_change | 400.000000 | 404.000000 | exact_unbounded_fallback_bounded_ls | 1.2490e-16 | 5.7732e-15 | 5.516277 | 0.469146 |
| setpoint_change | 400.000000 | 405.000000 | exact_unbounded_fallback_bounded_ls | 2.3592e-16 | 4.4409e-16 | 5.253466 | 0.532862 |
| setpoint_change | 800.000000 | 795.000000 | exact_unbounded_fallback_bounded_ls | 5.8547e-18 | 5.5511e-17 | 0.041843 | 1.4982e-04 |
| setpoint_change | 800.000000 | 796.000000 | exact_unbounded_fallback_bounded_ls | 7.5894e-19 | 0.000000 | 0.041977 | 1.4916e-04 |
| setpoint_change | 800.000000 | 797.000000 | exact_unbounded_fallback_bounded_ls | 3.2526e-18 | 6.9389e-18 | 0.042110 | 1.4832e-04 |
| setpoint_change | 800.000000 | 798.000000 | exact_unbounded_fallback_bounded_ls | 6.0715e-18 | 2.7756e-17 | 0.042242 | 1.4684e-04 |
| setpoint_change | 800.000000 | 799.000000 | exact_unbounded_fallback_bounded_ls | 5.4210e-19 | 5.5511e-17 | 0.042373 | 1.4643e-04 |
| setpoint_change | 800.000000 | 800.000000 | exact_unbounded_fallback_bounded_ls | 9.0206e-17 | 8.8818e-16 | 6.048586 | 1.4579e-04 |
| setpoint_change | 800.000000 | 801.000000 | exact_unbounded_fallback_bounded_ls | 1.3184e-16 | 0.000000 | 6.048589 | 1.4597e-04 |
| setpoint_change | 800.000000 | 802.000000 | exact_unbounded_fallback_bounded_ls | 1.1796e-16 | 1.3323e-15 | 5.886967 | 0.294180 |
| setpoint_change | 800.000000 | 803.000000 | exact_unbounded_fallback_bounded_ls | 6.9389e-18 | 1.7764e-15 | 5.620814 | 0.526554 |
