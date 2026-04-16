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
| 5.0797e-17 | 2.1080e-16 | 3.5531e-15 | 1.5597e-16 | 4.3156e-16 | 5.8411e-15 | 2.1263e-16 | 4.9772e-16 | 5.8588e-15 | 134.293237 | 282.356942 | 1447.591762 | 0.474384 | 0.987075 | 6.274134 | 38.804451 | 74.803567 | 487.529539 | 0.588407 | 1.205687 | 6.251886 | 136.823780 | 281.247733 | 1448.219762 | 43.710369 | 91.110413 | 487.779785 | 8.8343e-17 | 2.3858e-16 | 1.9984e-15 | 0.521889 | 1.170144 | 6.082218 |

## Sampled Per-Step Diagnostics

| k | residual_dyn_norm | residual_out_norm | residual_total_norm | exact_solution | used_lstsq | solver_mode | u_applied_minus_u_s_norm | y_current_minus_y_s_norm | xhat_minus_x_s_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 9.3675e-17 | 8.8818e-16 | 9.4076e-16 | true | false | stacked_exact | 823.756320 | 2.798133 | 274.896971 |
| 20.000000 | 1.7347e-17 | 8.8818e-16 | 9.1888e-16 | true | false | stacked_exact | 602.100189 | 1.641790 | 119.973564 |
| 40.000000 | 1.2143e-17 | 4.4409e-16 | 5.0816e-16 | true | false | stacked_exact | 367.414498 | 1.558279 | 116.992029 |
| 60.000000 | 4.4755e-16 | 2.2204e-16 | 5.8172e-16 | true | false | stacked_exact | 284.986235 | 1.119852 | 98.510603 |
| 80.000000 | 2.3419e-17 | 1.1102e-16 | 2.2048e-16 | true | false | stacked_exact | 186.034445 | 0.717057 | 61.848747 |
| 100.000000 | 8.2399e-18 | 1.1102e-16 | 1.2629e-16 | true | false | stacked_exact | 116.689286 | 0.448214 | 39.898778 |
| 120.000000 | 1.2143e-17 | 1.1102e-16 | 1.3060e-16 | true | false | stacked_exact | 74.578023 | 0.274678 | 25.600093 |
| 140.000000 | 1.3227e-17 | 1.1102e-16 | 1.2395e-16 | true | false | stacked_exact | 46.997679 | 0.168003 | 16.044629 |
| 160.000000 | 3.6863e-18 | 0.000000 | 1.3042e-17 | true | false | stacked_exact | 29.427059 | 0.103268 | 10.054322 |
| 180.000000 | 1.7347e-18 | 0.000000 | 1.0696e-17 | true | false | stacked_exact | 18.437486 | 0.063713 | 6.298850 |
| 200.000000 | 1.4183e-17 | 6.9389e-18 | 1.8487e-17 | true | false | stacked_exact | 11.550972 | 0.039476 | 3.940457 |
| 220.000000 | 1.5721e-18 | 1.3878e-17 | 1.7455e-17 | true | false | stacked_exact | 7.240567 | 0.024577 | 2.466695 |
| 240.000000 | 7.0473e-19 | 1.3878e-17 | 1.7653e-17 | true | false | stacked_exact | 4.549847 | 0.015377 | 1.547171 |
| 260.000000 | 6.7763e-19 | 0.000000 | 3.6142e-18 | true | false | stacked_exact | 2.871645 | 0.009680 | 0.973759 |
| 280.000000 | 9.7578e-19 | 1.3878e-17 | 1.5531e-17 | true | false | stacked_exact | 1.825189 | 0.006146 | 0.616242 |
| 300.000000 | 2.7105e-19 | 6.9389e-18 | 7.6408e-18 | true | false | stacked_exact | 1.172881 | 0.003952 | 0.393401 |
| 320.000000 | 1.3889e-17 | 6.9389e-18 | 1.8817e-17 | true | false | stacked_exact | 0.766220 | 0.002590 | 0.254705 |
| 340.000000 | 1.0842e-19 | 3.4694e-18 | 6.9055e-18 | true | false | stacked_exact | 0.513313 | 0.001743 | 0.168683 |
| 360.000000 | 3.5968e-18 | 6.9389e-18 | 8.3281e-18 | true | false | stacked_exact | 0.355948 | 0.001218 | 0.115127 |
| 380.000000 | 3.4780e-18 | 0.000000 | 3.5811e-18 | true | false | stacked_exact | 0.257819 | 8.9367e-04 | 0.082107 |
| 400.000000 | 6.9389e-18 | 5.8411e-15 | 5.8588e-15 | true | false | stacked_exact | 1398.921593 | 6.274016 | 487.512523 |
| 420.000000 | 7.6328e-17 | 8.8818e-16 | 1.0199e-15 | true | false | stacked_exact | 1017.566532 | 1.934038 | 117.250148 |
| 440.000000 | 3.9031e-17 | 0.000000 | 9.9892e-17 | true | false | stacked_exact | 149.141075 | 2.007205 | 152.912609 |
| 460.000000 | 2.2238e-16 | 4.9651e-16 | 5.7708e-16 | true | false | stacked_exact | 305.048490 | 0.828055 | 121.731010 |
| 480.000000 | 1.6480e-17 | 2.2204e-16 | 3.1779e-16 | true | false | stacked_exact | 226.718581 | 0.704955 | 52.475180 |
| 500.000000 | 1.7781e-17 | 0.000000 | 5.5815e-17 | true | false | stacked_exact | 102.836451 | 0.506170 | 40.832488 |
| 520.000000 | 2.2220e-16 | 1.1102e-16 | 1.9149e-16 | true | false | stacked_exact | 95.234131 | 0.323457 | 32.374516 |
| 540.000000 | 1.1128e-16 | 0.000000 | 1.1536e-16 | true | false | stacked_exact | 63.122800 | 0.233698 | 18.819155 |
| 560.000000 | 1.0842e-18 | 1.1102e-16 | 1.1779e-16 | true | false | stacked_exact | 38.342211 | 0.154466 | 13.568566 |
| 580.000000 | 1.8431e-18 | 5.5511e-17 | 6.0067e-17 | true | false | stacked_exact | 28.215029 | 0.100364 | 9.345256 |
| 600.000000 | 1.3885e-17 | 0.000000 | 3.0927e-17 | true | false | stacked_exact | 18.067321 | 0.066781 | 5.879430 |
| 620.000000 | 1.4014e-17 | 2.7756e-17 | 3.4458e-17 | true | false | stacked_exact | 11.563172 | 0.042625 | 3.974730 |
| 640.000000 | 8.1315e-19 | 1.3878e-17 | 1.4809e-17 | true | false | stacked_exact | 7.757564 | 0.027168 | 2.585731 |
| 660.000000 | 5.4210e-20 | 1.5516e-17 | 1.8223e-17 | true | false | stacked_exact | 4.858974 | 0.017204 | 1.637884 |
| 680.000000 | 2.7105e-19 | 3.4694e-18 | 8.6782e-18 | true | false | stacked_exact | 3.055120 | 0.010556 | 1.052398 |
| 700.000000 | 2.4395e-19 | 0.000000 | 5.3413e-18 | true | false | stacked_exact | 1.924267 | 0.006418 | 0.656129 |
| 720.000000 | 2.2023e-20 | 1.7347e-18 | 3.5141e-18 | true | false | stacked_exact | 1.170418 | 0.003857 | 0.404414 |
| 740.000000 | 1.2705e-21 | 1.3986e-17 | 1.4410e-17 | true | false | stacked_exact | 0.710541 | 0.002266 | 0.248218 |
| 760.000000 | 3.2187e-20 | 0.000000 | 1.1192e-18 | true | false | stacked_exact | 0.426115 | 0.001313 | 0.149156 |
| 780.000000 | 2.2023e-20 | 0.000000 | 5.6976e-18 | true | false | stacked_exact | 0.249070 | 7.4529e-04 | 0.088619 |
| 800.000000 | 3.5528e-15 | 1.7764e-15 | 2.6535e-15 | true | false | stacked_exact | 1398.972315 | 6.274134 | 487.529539 |
| 820.000000 | 4.4443e-16 | 8.8818e-16 | 1.1581e-15 | true | false | stacked_exact | 751.423061 | 1.071312 | 116.448950 |
| 840.000000 | 9.0206e-17 | 0.000000 | 2.7007e-16 | true | false | stacked_exact | 376.199277 | 1.575416 | 128.153809 |
| 860.000000 | 6.9389e-18 | 2.2204e-16 | 2.7995e-16 | true | false | stacked_exact | 336.013779 | 1.170173 | 117.800466 |
| 880.000000 | 4.4426e-16 | 4.4409e-16 | 6.9347e-16 | true | false | stacked_exact | 214.886402 | 0.763708 | 69.437088 |
| 900.000000 | 4.3368e-18 | 0.000000 | 7.1692e-17 | true | false | stacked_exact | 130.413083 | 0.484828 | 44.730798 |
| 920.000000 | 4.3368e-18 | 1.2413e-16 | 1.3439e-16 | true | false | stacked_exact | 82.953509 | 0.295742 | 28.695887 |
| 940.000000 | 1.1102e-16 | 1.1102e-16 | 1.3772e-16 | true | false | stacked_exact | 51.594395 | 0.178900 | 17.695433 |
| 960.000000 | 2.8200e-17 | 1.3878e-17 | 6.0868e-17 | true | false | stacked_exact | 31.771170 | 0.108594 | 10.936575 |
| 980.000000 | 8.6736e-19 | 0.000000 | 2.9729e-17 | true | false | stacked_exact | 19.623505 | 0.066003 | 6.760617 |
| 1000.000000 | 1.4014e-17 | 6.9389e-18 | 2.0282e-17 | true | false | stacked_exact | 12.111842 | 0.040246 | 4.166249 |
| 1020.000000 | 3.2526e-19 | 1.5516e-17 | 2.1793e-17 | true | false | stacked_exact | 7.473531 | 0.024648 | 2.568395 |
| 1040.000000 | 4.3368e-19 | 1.3878e-17 | 1.4025e-17 | true | false | stacked_exact | 4.622463 | 0.015166 | 1.585911 |
| 1060.000000 | 5.4210e-20 | 2.7972e-17 | 3.0133e-17 | true | false | stacked_exact | 2.870846 | 0.009387 | 0.982111 |
| 1080.000000 | 3.2526e-19 | 0.000000 | 6.3247e-18 | true | false | stacked_exact | 1.795357 | 0.005862 | 0.611417 |
| 1100.000000 | 1.3892e-17 | 1.3878e-17 | 1.6081e-17 | true | false | stacked_exact | 1.135938 | 0.003710 | 0.384047 |
| 1120.000000 | 1.3888e-17 | 7.7579e-18 | 1.7055e-17 | true | false | stacked_exact | 0.731865 | 0.002397 | 0.244855 |
| 1140.000000 | 1.0029e-18 | 0.000000 | 1.3482e-18 | true | false | stacked_exact | 0.484414 | 0.001598 | 0.159661 |
| 1160.000000 | 2.1684e-19 | 6.9389e-18 | 8.5628e-18 | true | false | stacked_exact | 0.332991 | 0.001112 | 0.107946 |
| 1180.000000 | 3.4800e-18 | 0.000000 | 5.7451e-18 | true | false | stacked_exact | 0.240744 | 8.1716e-04 | 0.076552 |
| 1200.000000 | 1.2490e-16 | 8.8818e-16 | 1.2862e-15 | true | false | stacked_exact | 1398.933674 | 6.274011 | 487.516437 |
| 1220.000000 | 9.1987e-16 | 1.8310e-15 | 2.0810e-15 | true | false | stacked_exact | 1017.556105 | 1.933954 | 117.231472 |
| 1240.000000 | 2.4286e-17 | 5.5511e-17 | 1.2670e-16 | true | false | stacked_exact | 149.126927 | 2.007334 | 152.918550 |
| 1260.000000 | 2.2207e-16 | 2.2204e-16 | 3.7469e-16 | true | false | stacked_exact | 305.103417 | 0.828261 | 121.745027 |
| 1280.000000 | 8.6736e-18 | 0.000000 | 1.4691e-16 | true | false | stacked_exact | 226.740626 | 0.705119 | 52.476977 |
| 1300.000000 | 3.0358e-18 | 2.2204e-16 | 2.3202e-16 | true | false | stacked_exact | 102.864530 | 0.506276 | 40.842393 |
| 1320.000000 | 4.7705e-18 | 1.1102e-16 | 1.4798e-16 | true | false | stacked_exact | 95.265006 | 0.323581 | 32.382060 |
| 1340.000000 | 5.6733e-17 | 1.1102e-16 | 1.2584e-16 | true | false | stacked_exact | 63.140821 | 0.233796 | 18.824373 |
| 1360.000000 | 7.8063e-18 | 5.5511e-17 | 6.0456e-17 | true | false | stacked_exact | 38.356798 | 0.154534 | 13.573777 |
| 1380.000000 | 2.7921e-17 | 5.7220e-17 | 6.5835e-17 | true | false | stacked_exact | 28.228065 | 0.100419 | 9.349133 |
| 1400.000000 | 4.3368e-19 | 0.000000 | 1.6933e-17 | true | false | stacked_exact | 18.076622 | 0.066821 | 5.882533 |
| 1420.000000 | 1.3939e-17 | 1.3878e-17 | 1.8039e-17 | true | false | stacked_exact | 11.570094 | 0.042653 | 3.976885 |
| 1440.000000 | 7.3184e-19 | 1.5516e-17 | 1.6844e-17 | true | false | stacked_exact | 7.770366 | 0.027213 | 2.588517 |
| 1460.000000 | 3.5237e-19 | 1.5516e-17 | 1.6145e-17 | true | false | stacked_exact | 4.871219 | 0.017260 | 1.640763 |
| 1480.000000 | 4.6079e-19 | 1.3878e-17 | 1.5092e-17 | true | false | stacked_exact | 3.063939 | 0.010602 | 1.055328 |
| 1500.000000 | 2.5750e-19 | 0.000000 | 3.6269e-18 | true | false | stacked_exact | 1.932014 | 0.006451 | 0.658514 |
| 1520.000000 | 9.6562e-20 | 1.3986e-17 | 1.4602e-17 | true | false | stacked_exact | 1.175916 | 0.003881 | 0.406051 |
| 1540.000000 | 5.0822e-21 | 1.4305e-17 | 1.5569e-17 | true | false | stacked_exact | 0.714209 | 0.002281 | 0.249477 |
| 1560.000000 | 4.4046e-20 | 3.4694e-18 | 3.8177e-18 | true | false | stacked_exact | 0.428891 | 0.001323 | 0.150004 |
| 1580.000000 | 7.2845e-20 | 5.2042e-18 | 5.9233e-18 | true | false | stacked_exact | 0.250809 | 7.5209e-04 | 0.089206 |

## Box-Constrained Analysis Summary

| pct_exact_bounded | pct_exact_unbounded_fallback_bounded_ls | pct_exact_unsolved_fallback_bounded_ls | pct_failed | pct_exact_solutions_inside_bounds | avg_exact_bound_violation_inf | max_exact_bound_violation_inf | avg_bounded_residual_norm | max_bounded_residual_norm | avg_us_exact_minus_us_bounded_inf | max_us_exact_minus_us_bounded_inf | avg_xs_exact_minus_xs_bounded_inf | max_xs_exact_minus_xs_bounded_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 43.812500 | 56.187500 | 0.000000 | 0.000000 | 43.812500 | 113.194504 | 1216.723735 | 0.521889 | 6.082218 | 113.846605 | 1221.461842 | 28.677963 | 359.902838 |

## Box Solve Mode Counts

| solve_mode | count |
| --- | --- |
| exact_bounded | 701.000000 |
| exact_unbounded_fallback_bounded_ls | 899.000000 |

## Per-Input Bound Activity

| input_index | fraction_lower_bound_active | fraction_upper_bound_active | average_exact_violation_below_lower | average_exact_violation_above_upper |
| --- | --- | --- | --- | --- |
| 0.000000 | 0.220625 | 0.263750 | 59.165477 | 54.022940 |
| 1.000000 | 0.132500 | 0.071875 | 30.201432 | 26.595327 |

## Box Event Table

| event_kind | event_anchor | k | solve_mode | exact_eq_residual_state_inf | exact_eq_residual_output_inf | bounded_residual_norm | dhat_delta_inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| setpoint_change | 400.000000 | 395.000000 | exact_bounded | 4.0658e-19 | 6.9389e-18 | 0.000000 | 8.3022e-06 |
| setpoint_change | 400.000000 | 396.000000 | exact_bounded | 1.3878e-17 | 0.000000 | 0.000000 | 8.0959e-06 |
| setpoint_change | 400.000000 | 397.000000 | exact_bounded | 3.2526e-19 | 0.000000 | 0.000000 | 7.8999e-06 |
| setpoint_change | 400.000000 | 398.000000 | exact_bounded | 1.3878e-17 | 6.9389e-18 | 0.000000 | 7.7164e-06 |
| setpoint_change | 400.000000 | 399.000000 | exact_bounded | 1.3878e-17 | 3.4694e-18 | 0.000000 | 7.5421e-06 |
| setpoint_change | 400.000000 | 400.000000 | exact_unbounded_fallback_bounded_ls | 6.9389e-18 | 5.7732e-15 | 6.082204 | 7.3729e-06 |
| setpoint_change | 400.000000 | 401.000000 | exact_unbounded_fallback_bounded_ls | 1.4572e-16 | 3.1086e-15 | 6.082207 | 7.2059e-06 |
| setpoint_change | 400.000000 | 402.000000 | exact_unbounded_fallback_bounded_ls | 2.4286e-16 | 1.7764e-15 | 5.967190 | 0.198520 |
| setpoint_change | 400.000000 | 403.000000 | exact_unbounded_fallback_bounded_ls | 3.4694e-17 | 2.6645e-15 | 5.764690 | 0.364432 |
| setpoint_change | 400.000000 | 404.000000 | exact_unbounded_fallback_bounded_ls | 2.2204e-16 | 2.6645e-15 | 5.520376 | 0.471191 |
| setpoint_change | 400.000000 | 405.000000 | exact_unbounded_fallback_bounded_ls | 2.0123e-16 | 1.3323e-15 | 5.269137 | 0.533803 |
| setpoint_change | 800.000000 | 795.000000 | exact_bounded | 1.3878e-17 | 3.4694e-18 | 0.000000 | 1.4292e-05 |
| setpoint_change | 800.000000 | 796.000000 | exact_bounded | 1.1858e-20 | 0.000000 | 0.000000 | 1.3933e-05 |
| setpoint_change | 800.000000 | 797.000000 | exact_bounded | 6.9457e-20 | 0.000000 | 0.000000 | 1.3584e-05 |
| setpoint_change | 800.000000 | 798.000000 | exact_bounded | 4.3368e-19 | 0.000000 | 0.000000 | 1.3246e-05 |
| setpoint_change | 800.000000 | 799.000000 | exact_bounded | 1.0164e-20 | 0.000000 | 0.000000 | 1.2917e-05 |
| setpoint_change | 800.000000 | 800.000000 | exact_unbounded_fallback_bounded_ls | 3.5527e-15 | 1.7764e-15 | 6.061613 | 1.2597e-05 |
| setpoint_change | 800.000000 | 801.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 3.9968e-15 | 6.061619 | 1.2286e-05 |
| setpoint_change | 800.000000 | 802.000000 | exact_unbounded_fallback_bounded_ls | 1.0408e-16 | 1.7764e-15 | 5.894235 | 0.295503 |
| setpoint_change | 800.000000 | 803.000000 | exact_unbounded_fallback_bounded_ls | 2.4980e-16 | 8.8818e-16 | 5.617220 | 0.529494 |
