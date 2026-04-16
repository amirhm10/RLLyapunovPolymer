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
| 1.5291e-16 | 3.8603e-16 | 3.5604e-15 | 4.7562e-16 | 6.9669e-16 | 6.2172e-15 | 6.4573e-16 | 8.0002e-16 | 7.0679e-15 | 399.242955 | 377.972463 | 1630.318860 | 1.901044 | 1.802593 | 7.987864 | 151.473840 | 142.332322 | 641.210931 | 2.621084 | 2.761438 | 14.342706 | 399.846297 | 376.261809 | 1634.797598 | 129.057373 | 121.579370 | 499.361246 | 2.4798e-16 | 3.7733e-16 | 2.7013e-15 | 2.458709 | 2.719141 | 14.077809 |

## Sampled Per-Step Diagnostics

| k | residual_dyn_norm | residual_out_norm | residual_total_norm | exact_solution | used_lstsq | solver_mode | u_applied_minus_u_s_norm | y_current_minus_y_s_norm | xhat_minus_x_s_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 9.3675e-17 | 8.8818e-16 | 9.4076e-16 | true | false | stacked_exact | 823.653903 | 2.798133 | 274.896971 |
| 20.000000 | 2.4286e-17 | 8.8818e-16 | 9.7256e-16 | true | false | stacked_exact | 559.828684 | 1.765979 | 104.916812 |
| 40.000000 | 1.2143e-17 | 1.3506e-15 | 1.3709e-15 | true | false | stacked_exact | 398.842172 | 1.727125 | 131.560639 |
| 60.000000 | 6.9389e-18 | 4.4409e-16 | 4.5496e-16 | true | false | stacked_exact | 310.317854 | 1.230174 | 103.501958 |
| 80.000000 | 1.0408e-17 | 0.000000 | 8.3159e-17 | true | false | stacked_exact | 204.749767 | 0.809185 | 65.478270 |
| 100.000000 | 2.6021e-18 | 1.1102e-16 | 1.2839e-16 | true | false | stacked_exact | 136.562207 | 0.534977 | 44.006565 |
| 120.000000 | 8.2399e-18 | 1.1102e-16 | 1.4761e-16 | true | false | stacked_exact | 94.929885 | 0.362195 | 29.859885 |
| 140.000000 | 1.3010e-18 | 2.2204e-16 | 2.2892e-16 | true | false | stacked_exact | 67.985437 | 0.258683 | 20.590512 |
| 160.000000 | 1.1493e-17 | 5.5511e-17 | 5.7900e-17 | true | false | stacked_exact | 51.120799 | 0.197500 | 14.955188 |
| 180.000000 | 8.6736e-18 | 1.1102e-16 | 1.2214e-16 | true | false | stacked_exact | 40.763152 | 0.161170 | 11.561177 |
| 200.000000 | 5.5819e-17 | 5.5511e-17 | 7.3099e-17 | true | false | stacked_exact | 34.413536 | 0.139627 | 9.532195 |
| 220.000000 | 2.8322e-17 | 0.000000 | 2.9208e-17 | true | false | stacked_exact | 31.135719 | 0.118593 | 9.570245 |
| 240.000000 | 1.1447e-16 | 0.000000 | 9.9732e-17 | true | false | stacked_exact | 33.380546 | 0.117663 | 15.515233 |
| 260.000000 | 1.7764e-15 | 2.2204e-16 | 1.6170e-15 | true | false | stacked_exact | 431.106314 | 3.676614 | 318.386065 |
| 280.000000 | 9.7145e-17 | 1.1102e-16 | 2.0515e-16 | true | false | stacked_exact | 800.347031 | 5.900782 | 223.301241 |
| 300.000000 | 2.4286e-16 | 6.2804e-16 | 7.0827e-16 | true | false | stacked_exact | 1073.687394 | 7.120682 | 558.119028 |
| 320.000000 | 5.5511e-17 | 8.0059e-16 | 9.8064e-16 | true | false | stacked_exact | 880.008445 | 3.500469 | 321.532369 |
| 340.000000 | 9.1940e-17 | 9.9301e-16 | 1.9618e-15 | true | false | stacked_exact | 693.375552 | 3.585309 | 464.184757 |
| 360.000000 | 7.8063e-18 | 8.9509e-16 | 9.2065e-16 | true | false | stacked_exact | 187.602576 | 2.053308 | 254.906666 |
| 380.000000 | 1.3879e-17 | 0.000000 | 2.1502e-17 | true | false | stacked_exact | 42.344276 | 1.095319 | 63.099531 |
| 400.000000 | 6.2450e-17 | 8.8818e-16 | 1.1623e-15 | true | false | stacked_exact | 1326.692512 | 6.160176 | 456.101949 |
| 420.000000 | 1.1796e-16 | 1.7764e-15 | 1.8130e-15 | true | false | stacked_exact | 979.641771 | 1.874352 | 105.559546 |
| 440.000000 | 1.0408e-17 | 0.000000 | 9.1098e-17 | true | false | stacked_exact | 145.084453 | 1.921321 | 143.030534 |
| 460.000000 | 2.2767e-16 | 0.000000 | 2.9612e-16 | true | false | stacked_exact | 295.264903 | 0.792871 | 117.162944 |
| 480.000000 | 2.2551e-17 | 4.9651e-16 | 7.5812e-16 | true | false | stacked_exact | 423.457597 | 2.482838 | 263.788566 |
| 500.000000 | 2.2118e-17 | 2.2204e-16 | 3.5644e-16 | true | false | stacked_exact | 205.378024 | 2.124560 | 189.973151 |
| 520.000000 | 9.0772e-16 | 8.8818e-16 | 1.5447e-15 | true | false | stacked_exact | 983.096138 | 3.887291 | 254.651423 |
| 540.000000 | 0.000000 | 4.4409e-16 | 5.2150e-16 | true | false | stacked_exact | 589.626020 | 3.892779 | 223.058068 |
| 560.000000 | 5.5511e-17 | 4.4409e-16 | 9.3507e-16 | true | false | stacked_exact | 603.733302 | 2.667355 | 227.974941 |
| 580.000000 | 1.9082e-17 | 4.4409e-16 | 5.7437e-16 | true | false | stacked_exact | 358.575298 | 1.669872 | 111.576523 |
| 600.000000 | 2.8623e-17 | 4.5776e-16 | 4.6363e-16 | true | false | stacked_exact | 211.042530 | 0.958286 | 81.904968 |
| 620.000000 | 2.6021e-18 | 2.2204e-16 | 2.4115e-16 | true | false | stacked_exact | 157.266809 | 0.581099 | 51.763623 |
| 640.000000 | 1.7781e-17 | 0.000000 | 7.3076e-17 | true | false | stacked_exact | 97.924384 | 0.395128 | 31.845846 |
| 660.000000 | 7.3726e-18 | 1.1444e-16 | 1.2294e-16 | true | false | stacked_exact | 68.942022 | 0.279091 | 23.415699 |
| 680.000000 | 6.9389e-18 | 0.000000 | 1.2553e-17 | true | false | stacked_exact | 52.766623 | 0.218487 | 16.938254 |
| 700.000000 | 5.2042e-18 | 3.9252e-17 | 4.3115e-17 | true | false | stacked_exact | 40.041864 | 0.181557 | 13.085675 |
| 720.000000 | 6.2149e-17 | 2.7756e-17 | 5.8922e-17 | true | false | stacked_exact | 32.481968 | 0.151916 | 12.360490 |
| 740.000000 | 3.7947e-18 | 5.5511e-17 | 1.1851e-16 | true | false | stacked_exact | 53.244825 | 0.628144 | 45.200397 |
| 760.000000 | 9.9747e-18 | 2.2204e-16 | 2.6581e-16 | true | false | stacked_exact | 212.710676 | 2.918657 | 136.559464 |
| 780.000000 | 3.4694e-17 | 4.4409e-16 | 1.6435e-15 | true | false | stacked_exact | 1038.898818 | 5.392211 | 467.186243 |
| 800.000000 | 1.8735e-16 | 1.9860e-15 | 2.1925e-15 | true | false | stacked_exact | 1089.799987 | 6.676641 | 540.267163 |
| 820.000000 | 6.3317e-17 | 9.0366e-16 | 9.8372e-16 | true | false | stacked_exact | 241.334933 | 2.380859 | 298.495451 |
| 840.000000 | 1.3878e-17 | 1.1102e-16 | 2.2343e-16 | true | false | stacked_exact | 192.386837 | 1.627708 | 123.923650 |
| 860.000000 | 4.9728e-16 | 4.4409e-16 | 6.5740e-16 | true | false | stacked_exact | 267.980261 | 0.981328 | 93.275018 |
| 880.000000 | 2.2204e-16 | 3.1402e-16 | 5.7149e-16 | true | false | stacked_exact | 314.896529 | 1.187000 | 127.215252 |
| 900.000000 | 8.6736e-19 | 0.000000 | 9.4978e-17 | true | false | stacked_exact | 43.360380 | 0.823795 | 55.030150 |
| 920.000000 | 1.0408e-17 | 5.5511e-17 | 7.1337e-17 | true | false | stacked_exact | 36.362825 | 0.263973 | 28.182356 |
| 940.000000 | 6.0715e-18 | 5.5511e-17 | 7.4283e-17 | true | false | stacked_exact | 41.715500 | 0.120685 | 12.046684 |
| 960.000000 | 8.6736e-19 | 8.3267e-17 | 1.3489e-16 | true | false | stacked_exact | 135.854414 | 1.129979 | 117.548417 |
| 980.000000 | 1.1103e-16 | 1.1102e-16 | 1.9501e-16 | true | false | stacked_exact | 43.102190 | 1.044567 | 31.286618 |
| 1000.000000 | 4.3368e-19 | 2.2204e-16 | 2.2407e-16 | true | false | stacked_exact | 105.541444 | 0.569722 | 52.706667 |
| 1020.000000 | 3.0358e-18 | 0.000000 | 3.0202e-17 | true | false | stacked_exact | 76.235354 | 0.379992 | 21.724226 |
| 1040.000000 | 5.2042e-18 | 1.1102e-16 | 1.2346e-16 | true | false | stacked_exact | 102.201038 | 0.307173 | 26.807880 |
| 1060.000000 | 1.7347e-17 | 1.7764e-15 | 1.8086e-15 | true | false | stacked_exact | 761.579049 | 5.649314 | 395.621892 |
| 1080.000000 | 8.3267e-17 | 8.8818e-16 | 8.9657e-16 | true | false | stacked_exact | 511.636400 | 1.123704 | 58.728900 |
| 1100.000000 | 2.4286e-17 | 2.2204e-16 | 2.3342e-16 | true | false | stacked_exact | 277.814216 | 1.609705 | 93.029228 |
| 1120.000000 | 2.2551e-17 | 1.1322e-15 | 1.1673e-15 | true | false | stacked_exact | 311.774073 | 1.315573 | 109.294856 |
| 1140.000000 | 2.6021e-18 | 6.6613e-16 | 6.9008e-16 | true | false | stacked_exact | 229.159280 | 0.975594 | 70.082764 |
| 1160.000000 | 1.3010e-17 | 1.1102e-16 | 1.8682e-16 | true | false | stacked_exact | 166.997371 | 0.722105 | 53.050895 |
| 1180.000000 | 9.5410e-18 | 0.000000 | 5.7352e-17 | true | false | stacked_exact | 126.495070 | 0.528902 | 39.885132 |
| 1200.000000 | 1.1102e-16 | 0.000000 | 6.7319e-16 | true | false | stacked_exact | 1307.594856 | 6.325510 | 459.990950 |
| 1220.000000 | 1.7777e-15 | 8.8818e-16 | 2.1884e-15 | true | false | stacked_exact | 925.985573 | 1.654982 | 88.815054 |
| 1240.000000 | 3.0358e-18 | 1.1102e-16 | 1.1441e-16 | true | false | stacked_exact | 74.969657 | 1.835303 | 143.989012 |
| 1260.000000 | 4.9440e-17 | 4.4409e-16 | 4.9584e-16 | true | false | stacked_exact | 268.549867 | 0.666417 | 107.042126 |
| 1280.000000 | 8.6736e-18 | 9.9301e-16 | 1.2248e-15 | true | false | stacked_exact | 566.726302 | 3.818683 | 374.737768 |
| 1300.000000 | 6.0715e-18 | 2.2888e-16 | 2.5835e-16 | true | false | stacked_exact | 183.591406 | 1.531898 | 183.012670 |
| 1320.000000 | 6.2450e-17 | 0.000000 | 6.5990e-17 | true | false | stacked_exact | 615.489417 | 6.612716 | 209.808381 |
| 1340.000000 | 2.3245e-16 | 8.8818e-16 | 1.1002e-15 | true | false | stacked_exact | 910.495275 | 4.481849 | 440.387956 |
| 1360.000000 | 0.000000 | 4.4409e-16 | 4.6894e-16 | true | false | stacked_exact | 551.686477 | 2.678263 | 139.668401 |
| 1380.000000 | 1.6653e-16 | 0.000000 | 8.4724e-16 | true | false | stacked_exact | 757.100474 | 3.460923 | 320.212823 |
| 1400.000000 | 9.5410e-18 | 5.5511e-17 | 5.9479e-17 | true | false | stacked_exact | 196.718318 | 1.021872 | 107.890239 |
| 1420.000000 | 1.9516e-18 | 0.000000 | 1.6173e-17 | true | false | stacked_exact | 36.158233 | 0.491396 | 64.731655 |
| 1440.000000 | 4.6838e-17 | 8.9207e-16 | 8.9783e-16 | true | false | stacked_exact | 300.379527 | 0.974190 | 152.004295 |
| 1460.000000 | 8.1532e-17 | 8.8818e-16 | 9.2054e-16 | true | false | stacked_exact | 366.740854 | 2.085441 | 88.083613 |
| 1480.000000 | 1.3010e-17 | 0.000000 | 1.5491e-16 | true | false | stacked_exact | 162.232137 | 0.962373 | 110.353329 |
| 1500.000000 | 1.6653e-16 | 8.8818e-16 | 1.3939e-15 | true | false | stacked_exact | 1332.722950 | 5.518973 | 398.663791 |
| 1520.000000 | 5.2042e-17 | 0.000000 | 2.2138e-16 | true | false | stacked_exact | 512.338629 | 2.470223 | 70.193348 |
| 1540.000000 | 9.0366e-16 | 3.5527e-15 | 3.7435e-15 | true | false | stacked_exact | 1336.665591 | 6.060375 | 636.718610 |
| 1560.000000 | 4.8572e-17 | 0.000000 | 2.2172e-16 | true | false | stacked_exact | 368.537969 | 1.766748 | 373.246796 |
| 1580.000000 | 2.1511e-16 | 1.7764e-15 | 1.7979e-15 | true | false | stacked_exact | 1420.260735 | 3.260106 | 282.433225 |

## Box-Constrained Analysis Summary

| pct_exact_bounded | pct_exact_unbounded_fallback_bounded_ls | pct_exact_unsolved_fallback_bounded_ls | pct_failed | pct_exact_solutions_inside_bounds | avg_exact_bound_violation_inf | max_exact_bound_violation_inf | avg_bounded_residual_norm | max_bounded_residual_norm | avg_us_exact_minus_us_bounded_inf | max_us_exact_minus_us_bounded_inf | avg_xs_exact_minus_xs_bounded_inf | max_xs_exact_minus_xs_bounded_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.187500 | 99.812500 | 0.000000 | 0.000000 | 0.187500 | 346.965246 | 1527.157487 | 2.458709 | 14.077809 | 349.098593 | 1527.157487 | 94.447494 | 365.786368 |

## Box Solve Mode Counts

| solve_mode | count |
| --- | --- |
| exact_bounded | 3.000000 |
| exact_unbounded_fallback_bounded_ls | 1597.000000 |

## Per-Input Bound Activity

| input_index | fraction_lower_bound_active | fraction_upper_bound_active | average_exact_violation_below_lower | average_exact_violation_above_upper |
| --- | --- | --- | --- | --- |
| 0.000000 | 0.223125 | 0.678750 | 124.020588 | 207.313269 |
| 1.000000 | 0.452500 | 0.261875 | 78.852459 | 94.781856 |

## Box Event Table

| event_kind | event_anchor | k | solve_mode | exact_eq_residual_state_inf | exact_eq_residual_output_inf | bounded_residual_norm | dhat_delta_inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| setpoint_change | 400.000000 | 395.000000 | exact_unbounded_fallback_bounded_ls | 6.2884e-18 | 5.5511e-17 | 0.300040 | 0.035498 |
| setpoint_change | 400.000000 | 396.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 1.6653e-16 | 0.288839 | 0.036823 |
| setpoint_change | 400.000000 | 397.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 1.1102e-16 | 0.279816 | 0.037473 |
| setpoint_change | 400.000000 | 398.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 2.2204e-16 | 0.271881 | 0.037521 |
| setpoint_change | 400.000000 | 399.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 5.5511e-17 | 0.262340 | 0.037044 |
| setpoint_change | 400.000000 | 400.000000 | exact_unbounded_fallback_bounded_ls | 6.2450e-17 | 8.8818e-16 | 5.769542 | 0.036112 |
| setpoint_change | 400.000000 | 401.000000 | exact_unbounded_fallback_bounded_ls | 8.8818e-16 | 4.4409e-16 | 5.796313 | 0.034798 |
| setpoint_change | 400.000000 | 402.000000 | exact_unbounded_fallback_bounded_ls | 8.8818e-16 | 1.3323e-15 | 5.706409 | 0.161337 |
| setpoint_change | 400.000000 | 403.000000 | exact_unbounded_fallback_bounded_ls | 4.8572e-17 | 1.3323e-15 | 5.526294 | 0.326857 |
| setpoint_change | 400.000000 | 404.000000 | exact_unbounded_fallback_bounded_ls | 2.4980e-16 | 8.8818e-16 | 5.300806 | 0.435066 |
| setpoint_change | 400.000000 | 405.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 4.4409e-15 | 5.064273 | 0.500082 |
| setpoint_change | 800.000000 | 795.000000 | exact_unbounded_fallback_bounded_ls | 1.3184e-16 | 4.4409e-16 | 3.794022 | 0.166355 |
| setpoint_change | 800.000000 | 796.000000 | exact_unbounded_fallback_bounded_ls | 2.2204e-16 | 8.8818e-16 | 3.838493 | 0.154088 |
| setpoint_change | 800.000000 | 797.000000 | exact_unbounded_fallback_bounded_ls | 4.4409e-16 | 4.4409e-16 | 3.840553 | 0.150531 |
| setpoint_change | 800.000000 | 798.000000 | exact_unbounded_fallback_bounded_ls | 4.3368e-17 | 0.000000 | 3.800263 | 0.146779 |
| setpoint_change | 800.000000 | 799.000000 | exact_unbounded_fallback_bounded_ls | 1.1102e-16 | 0.000000 | 3.719201 | 0.142789 |
| setpoint_change | 800.000000 | 800.000000 | exact_unbounded_fallback_bounded_ls | 1.8735e-16 | 1.7764e-15 | 8.472502 | 0.138499 |
| setpoint_change | 800.000000 | 801.000000 | exact_unbounded_fallback_bounded_ls | 2.0123e-16 | 2.6645e-15 | 8.428084 | 0.133830 |
| setpoint_change | 800.000000 | 802.000000 | exact_unbounded_fallback_bounded_ls | 2.2551e-16 | 3.5527e-15 | 8.085150 | 0.482603 |
| setpoint_change | 800.000000 | 803.000000 | exact_unbounded_fallback_bounded_ls | 1.4919e-16 | 3.5527e-15 | 7.525821 | 0.766403 |
