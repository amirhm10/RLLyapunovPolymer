# Offset-Free MPC Steady-State Debug Summary

- Case: `first_step_contraction_bounded_frozen_dhat`
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
| 6.1929e-17 | 2.6328e-16 | 3.5563e-15 | 1.6209e-16 | 4.1807e-16 | 4.5288e-15 | 2.3091e-16 | 5.2257e-16 | 5.6357e-15 | 146.482763 | 272.061700 | 1425.180219 | 0.531845 | 0.955664 | 6.284358 | 41.635444 | 71.461974 | 489.226071 | 0.642595 | 1.154080 | 6.278449 | 148.533555 | 271.036041 | 1424.316546 | 47.417544 | 88.052206 | 486.551976 | 1.0001e-16 | 2.4403e-16 | 2.1869e-15 | 0.635713 | 1.174879 | 6.288032 |

## Sampled Per-Step Diagnostics

| k | residual_dyn_norm | residual_out_norm | residual_total_norm | exact_solution | used_lstsq | solver_mode | u_applied_minus_u_s_norm | y_current_minus_y_s_norm | xhat_minus_x_s_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 9.3675e-17 | 8.8818e-16 | 9.4076e-16 | true | false | stacked_exact | 823.756320 | 2.798133 | 274.896971 |
| 20.000000 | 1.7347e-17 | 0.000000 | 2.0978e-16 | true | false | stacked_exact | 605.461801 | 1.656460 | 116.254809 |
| 40.000000 | 2.7756e-17 | 2.2204e-16 | 2.3412e-16 | true | false | stacked_exact | 377.289166 | 1.594035 | 116.680357 |
| 60.000000 | 2.2286e-16 | 8.8818e-16 | 1.0044e-15 | true | false | stacked_exact | 297.072894 | 1.171539 | 100.060621 |
| 80.000000 | 3.2960e-17 | 1.1102e-16 | 2.1380e-16 | true | false | stacked_exact | 200.773668 | 0.780679 | 63.960726 |
| 100.000000 | 1.1237e-16 | 1.1102e-16 | 1.6930e-16 | true | false | stacked_exact | 133.656185 | 0.521798 | 42.758408 |
| 120.000000 | 1.3010e-17 | 1.1102e-16 | 1.4735e-16 | true | false | stacked_exact | 93.269506 | 0.356554 | 29.280478 |
| 140.000000 | 3.9031e-18 | 1.1444e-16 | 1.2761e-16 | true | false | stacked_exact | 67.177931 | 0.256213 | 20.310209 |
| 160.000000 | 2.1684e-18 | 5.5511e-17 | 7.2470e-17 | true | false | stacked_exact | 50.703876 | 0.196482 | 14.812342 |
| 180.000000 | 2.1684e-19 | 2.7756e-17 | 2.8556e-17 | true | false | stacked_exact | 40.558783 | 0.160811 | 11.497243 |
| 200.000000 | 2.8200e-17 | 5.5511e-17 | 6.7505e-17 | true | false | stacked_exact | 34.324758 | 0.139554 | 9.508473 |
| 220.000000 | 4.1200e-18 | 5.5511e-17 | 5.5758e-17 | true | false | stacked_exact | 30.516865 | 0.126962 | 8.335440 |
| 240.000000 | 5.5619e-17 | 0.000000 | 7.6624e-17 | true | false | stacked_exact | 28.228463 | 0.119607 | 7.655767 |
| 260.000000 | 1.5179e-18 | 0.000000 | 2.0211e-17 | true | false | stacked_exact | 26.887414 | 0.115464 | 7.269218 |
| 280.000000 | 2.9273e-18 | 0.000000 | 1.9903e-17 | true | false | stacked_exact | 26.135609 | 0.113307 | 7.059774 |
| 300.000000 | 2.9838e-17 | 2.7756e-17 | 4.3984e-17 | true | false | stacked_exact | 25.757111 | 0.112401 | 6.958899 |
| 320.000000 | 5.6073e-17 | 1.3878e-17 | 3.7281e-17 | true | false | stacked_exact | 25.612711 | 0.112290 | 6.924518 |
| 340.000000 | 5.5680e-17 | 8.3267e-17 | 9.8279e-17 | true | false | stacked_exact | 25.621259 | 0.112697 | 6.933199 |
| 360.000000 | 2.7759e-17 | 8.3267e-17 | 8.8110e-17 | true | false | stacked_exact | 25.724370 | 0.113440 | 6.968627 |
| 380.000000 | 2.8690e-17 | 5.7220e-17 | 6.9991e-17 | true | false | stacked_exact | 25.893728 | 0.114416 | 7.022443 |
| 400.000000 | 1.4572e-16 | 4.5288e-15 | 4.6092e-15 | true | false | stacked_exact | 1374.317913 | 6.281441 | 480.917028 |
| 420.000000 | 0.000000 | 9.9301e-16 | 1.1005e-15 | true | false | stacked_exact | 985.478189 | 1.828532 | 103.081526 |
| 440.000000 | 5.5842e-17 | 1.1102e-16 | 1.1713e-16 | true | false | stacked_exact | 127.848225 | 1.929027 | 147.469904 |
| 460.000000 | 3.6429e-17 | 4.4409e-16 | 4.5000e-16 | true | false | stacked_exact | 288.521264 | 0.737382 | 115.493734 |
| 480.000000 | 4.4531e-16 | 0.000000 | 3.2836e-16 | true | false | stacked_exact | 211.021118 | 0.630291 | 45.279380 |
| 500.000000 | 2.2209e-16 | 3.3307e-16 | 4.1367e-16 | true | false | stacked_exact | 95.395357 | 0.454414 | 36.857052 |
| 520.000000 | 2.2351e-16 | 1.1102e-16 | 2.0215e-16 | true | false | stacked_exact | 90.998110 | 0.289303 | 29.490646 |
| 540.000000 | 6.7221e-18 | 0.000000 | 1.8456e-17 | true | false | stacked_exact | 59.659408 | 0.205246 | 16.599604 |
| 560.000000 | 8.6736e-19 | 5.5511e-17 | 6.0372e-17 | true | false | stacked_exact | 35.948889 | 0.125437 | 11.904455 |
| 580.000000 | 2.0600e-18 | 0.000000 | 2.0774e-17 | true | false | stacked_exact | 25.013525 | 0.066766 | 7.569054 |
| 600.000000 | 1.6263e-19 | 2.8610e-17 | 3.1476e-17 | true | false | stacked_exact | 13.552989 | 0.026044 | 4.127452 |
| 620.000000 | 1.4307e-17 | 6.9389e-18 | 1.9463e-17 | true | false | stacked_exact | 6.054428 | 0.009491 | 2.389948 |
| 640.000000 | 2.7105e-20 | 6.9389e-18 | 1.1020e-17 | true | false | stacked_exact | 3.187076 | 0.032022 | 2.054253 |
| 660.000000 | 6.5052e-19 | 1.7347e-18 | 1.9520e-18 | true | false | stacked_exact | 5.826501 | 0.049947 | 2.862197 |
| 680.000000 | 1.4095e-18 | 0.000000 | 2.9156e-17 | true | false | stacked_exact | 24.952623 | 0.309660 | 19.949481 |
| 700.000000 | 4.3368e-19 | 1.1102e-16 | 1.3738e-16 | true | false | stacked_exact | 62.056422 | 0.323664 | 30.247178 |
| 720.000000 | 3.0358e-18 | 0.000000 | 4.9929e-17 | true | false | stacked_exact | 75.303271 | 0.328388 | 21.199498 |
| 740.000000 | 4.3368e-18 | 5.5511e-17 | 6.0015e-17 | true | false | stacked_exact | 47.472664 | 0.251239 | 15.268860 |
| 760.000000 | 1.1143e-16 | 5.5511e-17 | 1.0475e-16 | true | false | stacked_exact | 42.366409 | 0.175147 | 13.873114 |
| 780.000000 | 1.6263e-18 | 5.7220e-17 | 6.1956e-17 | true | false | stacked_exact | 31.343291 | 0.118437 | 9.009107 |
| 800.000000 | 7.6328e-17 | 2.2204e-15 | 2.3715e-15 | true | false | stacked_exact | 1379.210591 | 6.284358 | 482.059339 |
| 820.000000 | 1.6306e-16 | 2.6645e-15 | 2.6755e-15 | true | false | stacked_exact | 722.272565 | 0.974065 | 109.427880 |
| 840.000000 | 3.4694e-17 | 2.2204e-16 | 3.0438e-16 | true | false | stacked_exact | 350.591317 | 1.446166 | 119.199391 |
| 860.000000 | 6.9389e-18 | 8.8818e-16 | 9.6009e-16 | true | false | stacked_exact | 321.655316 | 1.107523 | 108.256920 |
| 880.000000 | 4.4412e-16 | 4.5776e-16 | 5.4643e-16 | true | false | stacked_exact | 221.741201 | 0.811299 | 66.965352 |
| 900.000000 | 1.1951e-16 | 3.3307e-16 | 3.7150e-16 | true | false | stacked_exact | 152.782916 | 0.595117 | 47.896003 |
| 920.000000 | 1.0842e-17 | 3.3307e-16 | 3.4974e-16 | true | false | stacked_exact | 111.518755 | 0.432558 | 34.772328 |
| 940.000000 | 2.6021e-18 | 1.1102e-16 | 1.1842e-16 | true | false | stacked_exact | 81.350860 | 0.316738 | 24.904105 |
| 960.000000 | 5.5538e-17 | 1.1102e-16 | 1.2689e-16 | true | false | stacked_exact | 59.817355 | 0.233281 | 18.122774 |
| 980.000000 | 5.4210e-18 | 0.000000 | 4.1915e-17 | true | false | stacked_exact | 44.330958 | 0.171512 | 13.246983 |
| 1000.000000 | 2.8200e-17 | 0.000000 | 4.4536e-17 | true | false | stacked_exact | 32.846306 | 0.124928 | 9.635066 |
| 1020.000000 | 2.7867e-17 | 2.7756e-17 | 4.1888e-17 | true | false | stacked_exact | 24.187771 | 0.089233 | 6.923750 |
| 1040.000000 | 5.0958e-18 | 5.5511e-17 | 5.6899e-17 | true | false | stacked_exact | 17.567597 | 0.061536 | 4.857684 |
| 1060.000000 | 2.2768e-18 | 2.7756e-17 | 2.9566e-17 | true | false | stacked_exact | 12.442089 | 0.039862 | 3.268855 |
| 1080.000000 | 2.7756e-17 | 1.3878e-17 | 3.4975e-17 | true | false | stacked_exact | 8.436204 | 0.022842 | 2.050137 |
| 1100.000000 | 1.2468e-18 | 2.7756e-17 | 3.0529e-17 | true | false | stacked_exact | 5.284081 | 0.009697 | 1.153546 |
| 1120.000000 | 2.8731e-18 | 0.000000 | 7.5619e-18 | true | false | stacked_exact | 2.795529 | 0.004326 | 0.673422 |
| 1140.000000 | 6.9389e-18 | 1.3878e-17 | 1.6801e-17 | true | false | stacked_exact | 0.866337 | 0.011292 | 0.799164 |
| 1160.000000 | 7.5894e-19 | 6.9389e-18 | 7.9584e-18 | true | false | stacked_exact | 0.885036 | 0.017959 | 1.160701 |
| 1180.000000 | 5.6921e-19 | 6.9389e-18 | 7.6632e-18 | true | false | stacked_exact | 2.103027 | 0.023405 | 1.507056 |
| 1200.000000 | 3.5563e-15 | 1.2561e-15 | 3.7928e-15 | true | false | stacked_exact | 1402.192790 | 6.275810 | 489.226071 |
| 1220.000000 | 3.1225e-17 | 8.8818e-16 | 9.7708e-16 | true | false | stacked_exact | 995.941422 | 1.856360 | 107.622801 |
| 1240.000000 | 1.1102e-16 | 0.000000 | 1.1872e-16 | true | false | stacked_exact | 121.608868 | 1.943651 | 150.385023 |
| 1260.000000 | 9.1617e-16 | 8.8818e-16 | 1.6049e-15 | true | false | stacked_exact | 305.416204 | 0.799715 | 118.066611 |
| 1280.000000 | 5.2042e-18 | 0.000000 | 2.1136e-16 | true | false | stacked_exact | 229.089770 | 0.735891 | 49.142152 |
| 1300.000000 | 4.3368e-18 | 0.000000 | 1.1027e-16 | true | false | stacked_exact | 121.054288 | 0.576498 | 42.609087 |
| 1320.000000 | 3.0358e-18 | 5.5511e-17 | 1.1185e-16 | true | false | stacked_exact | 123.170551 | 0.450504 | 38.296007 |
| 1340.000000 | 1.8648e-17 | 0.000000 | 6.7216e-17 | true | false | stacked_exact | 93.625320 | 0.378158 | 26.175948 |
| 1360.000000 | 8.2399e-18 | 2.7756e-17 | 4.6880e-17 | true | false | stacked_exact | 71.758425 | 0.301613 | 22.038884 |
| 1380.000000 | 1.1108e-16 | 2.2204e-16 | 2.5672e-16 | true | false | stacked_exact | 61.085967 | 0.244445 | 18.038872 |
| 1400.000000 | 1.1120e-16 | 5.5511e-17 | 1.3352e-16 | true | false | stacked_exact | 48.880349 | 0.198535 | 14.114794 |
| 1420.000000 | 2.7898e-17 | 5.5511e-17 | 6.3632e-17 | true | false | stacked_exact | 39.926994 | 0.159809 | 11.545681 |
| 1440.000000 | 2.8189e-18 | 5.7220e-17 | 6.6625e-17 | true | false | stacked_exact | 33.168428 | 0.129408 | 9.325594 |
| 1460.000000 | 2.8508e-17 | 0.000000 | 1.9867e-17 | true | false | stacked_exact | 27.420323 | 0.105016 | 7.524252 |
| 1480.000000 | 5.5648e-17 | 2.7756e-17 | 5.2703e-17 | true | false | stacked_exact | 22.995160 | 0.085514 | 6.143998 |
| 1500.000000 | 5.5512e-17 | 0.000000 | 2.8713e-17 | true | false | stacked_exact | 19.501102 | 0.070157 | 5.027620 |
| 1520.000000 | 1.0300e-18 | 2.7756e-17 | 3.3152e-17 | true | false | stacked_exact | 16.719198 | 0.058033 | 4.148218 |
| 1540.000000 | 2.4395e-18 | 5.5511e-17 | 5.7097e-17 | true | false | stacked_exact | 14.552420 | 0.048512 | 3.460451 |
| 1560.000000 | 1.6805e-18 | 2.7756e-17 | 3.4436e-17 | true | false | stacked_exact | 12.851864 | 0.041067 | 2.918162 |
| 1580.000000 | 5.9631e-19 | 1.3878e-17 | 2.9867e-17 | true | false | stacked_exact | 11.523942 | 0.035251 | 2.495095 |

## Box-Constrained Analysis Summary

| pct_exact_bounded | pct_exact_unbounded_fallback_bounded_ls | pct_exact_unsolved_fallback_bounded_ls | pct_failed | pct_exact_solutions_inside_bounds | avg_exact_bound_violation_inf | max_exact_bound_violation_inf | avg_bounded_residual_norm | max_bounded_residual_norm | avg_us_exact_minus_us_bounded_inf | max_us_exact_minus_us_bounded_inf | avg_xs_exact_minus_xs_bounded_inf | max_xs_exact_minus_xs_bounded_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.562500 | 91.437500 | 0.000000 | 0.000000 | 8.562500 | 121.696713 | 1194.940372 | 0.635713 | 6.288032 | 129.976505 | 1202.513157 | 32.774402 | 364.817347 |

## Box Solve Mode Counts

| solve_mode | count |
| --- | --- |
| exact_bounded | 137.000000 |
| exact_unbounded_fallback_bounded_ls | 1463.000000 |

## Per-Input Bound Activity

| input_index | fraction_lower_bound_active | fraction_upper_bound_active | average_exact_violation_below_lower | average_exact_violation_above_upper |
| --- | --- | --- | --- | --- |
| 0.000000 | 0.000000 | 0.000000 | 61.377625 | 60.060733 |
| 1.000000 | 0.000000 | 0.000000 | 32.607665 | 28.296410 |

## Box Event Table

| event_kind | event_anchor | k | solve_mode | exact_eq_residual_state_inf | exact_eq_residual_output_inf | bounded_residual_norm | dhat_delta_inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| setpoint_change | 400.000000 | 395.000000 | exact_unbounded_fallback_bounded_ls | 2.1684e-19 | 2.7756e-17 | 0.135670 | 9.1889e-05 |
| setpoint_change | 400.000000 | 396.000000 | exact_unbounded_fallback_bounded_ls | 8.6736e-19 | 0.000000 | 0.135727 | 9.1817e-05 |
| setpoint_change | 400.000000 | 397.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 2.7756e-17 | 0.135784 | 9.2350e-05 |
| setpoint_change | 400.000000 | 398.000000 | exact_unbounded_fallback_bounded_ls | 5.2042e-18 | 5.5511e-17 | 0.135842 | 9.2592e-05 |
| setpoint_change | 400.000000 | 399.000000 | exact_unbounded_fallback_bounded_ls | 2.7756e-17 | 2.7756e-17 | 0.135901 | 9.2927e-05 |
| setpoint_change | 400.000000 | 400.000000 | exact_unbounded_fallback_bounded_ls | 1.4572e-16 | 4.4409e-15 | 6.288032 | 9.2878e-05 |
| setpoint_change | 400.000000 | 401.000000 | exact_unbounded_fallback_bounded_ls | 6.9389e-18 | 3.9968e-15 | 6.279065 | 9.3653e-05 |
| setpoint_change | 400.000000 | 402.000000 | exact_unbounded_fallback_bounded_ls | 1.3184e-16 | 2.2204e-15 | 6.147530 | 0.193084 |
| setpoint_change | 400.000000 | 403.000000 | exact_unbounded_fallback_bounded_ls | 2.7756e-16 | 1.3323e-15 | 5.921835 | 0.355460 |
| setpoint_change | 400.000000 | 404.000000 | exact_unbounded_fallback_bounded_ls | 0.000000 | 1.3323e-15 | 5.648278 | 0.460957 |
| setpoint_change | 400.000000 | 405.000000 | exact_unbounded_fallback_bounded_ls | 2.0817e-16 | 0.000000 | 5.361992 | 0.523889 |
| setpoint_change | 800.000000 | 795.000000 | exact_unbounded_fallback_bounded_ls | 7.5894e-19 | 0.000000 | 0.072850 | 0.002249 |
| setpoint_change | 800.000000 | 796.000000 | exact_unbounded_fallback_bounded_ls | 1.3878e-17 | 2.7756e-17 | 0.071438 | 0.002210 |
| setpoint_change | 800.000000 | 797.000000 | exact_unbounded_fallback_bounded_ls | 1.3878e-17 | 1.3878e-17 | 0.070134 | 0.002164 |
| setpoint_change | 800.000000 | 798.000000 | exact_unbounded_fallback_bounded_ls | 1.3878e-17 | 0.000000 | 0.068934 | 0.002112 |
| setpoint_change | 800.000000 | 799.000000 | exact_unbounded_fallback_bounded_ls | 4.3368e-19 | 0.000000 | 0.067832 | 0.002054 |
| setpoint_change | 800.000000 | 800.000000 | exact_unbounded_fallback_bounded_ls | 7.6328e-17 | 2.2204e-15 | 6.215234 | 0.001993 |
| setpoint_change | 800.000000 | 801.000000 | exact_unbounded_fallback_bounded_ls | 2.0817e-16 | 1.3323e-15 | 6.206130 | 0.001928 |
| setpoint_change | 800.000000 | 802.000000 | exact_unbounded_fallback_bounded_ls | 1.8735e-16 | 8.8818e-16 | 6.017819 | 0.293510 |
| setpoint_change | 800.000000 | 803.000000 | exact_unbounded_fallback_bounded_ls | 1.3184e-16 | 3.9968e-15 | 5.711742 | 0.522718 |
