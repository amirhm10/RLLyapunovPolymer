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
| 1.3152e-16 | 3.8074e-16 | 3.5652e-15 | 3.3824e-16 | 5.6691e-16 | 4.4409e-15 | 4.8489e-16 | 6.9490e-16 | 6.0253e-15 | 293.758872 | 329.098801 | 1475.617799 | 1.439056 | 1.729068 | 9.957227 | 112.563773 | 127.412690 | 705.210042 | 1.818129 | 2.531634 | 17.240114 | 294.058133 | 328.154202 | 1472.197680 | 94.893814 | 105.753262 | 495.919067 | 1.7659e-16 | 2.9906e-16 | 2.2204e-15 | 1.689017 | 2.484019 | 16.975734 |

## Sampled Per-Step Diagnostics

| k | residual_dyn_norm | residual_out_norm | residual_total_norm | exact_solution | used_lstsq | solver_mode | u_applied_minus_u_s_norm | y_current_minus_y_s_norm | xhat_minus_x_s_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | 9.3675e-17 | 8.8818e-16 | 9.4076e-16 | true | false | stacked_exact | 823.653903 | 2.798133 | 274.896971 |
| 20.000000 | 4.6132e-16 | 4.4409e-16 | 7.2348e-16 | true | false | stacked_exact | 557.459431 | 1.750008 | 105.046904 |
| 40.000000 | 8.6736e-17 | 6.6613e-16 | 7.0347e-16 | true | false | stacked_exact | 392.637891 | 1.690409 | 130.146420 |
| 60.000000 | 1.3878e-17 | 1.1102e-15 | 1.1927e-15 | true | false | stacked_exact | 301.014011 | 1.178921 | 101.127686 |
| 80.000000 | 2.6021e-18 | 0.000000 | 1.2913e-16 | true | false | stacked_exact | 192.913265 | 0.745624 | 62.565667 |
| 100.000000 | 1.0842e-17 | 1.1102e-16 | 1.2504e-16 | true | false | stacked_exact | 122.472596 | 0.461017 | 40.559524 |
| 120.000000 | 8.6736e-19 | 1.1102e-16 | 1.2395e-16 | true | false | stacked_exact | 78.980517 | 0.279734 | 25.802367 |
| 140.000000 | 1.9516e-18 | 5.5511e-17 | 7.7305e-17 | true | false | stacked_exact | 50.475765 | 0.169845 | 16.075669 |
| 160.000000 | 4.3368e-19 | 5.5511e-17 | 6.6027e-17 | true | false | stacked_exact | 32.475314 | 0.103675 | 10.031755 |
| 180.000000 | 5.5680e-17 | 2.7756e-17 | 4.4955e-17 | true | false | stacked_exact | 21.261491 | 0.063522 | 6.253073 |
| 200.000000 | 8.6736e-19 | 2.7756e-17 | 3.2521e-17 | true | false | stacked_exact | 14.264341 | 0.039093 | 3.890592 |
| 220.000000 | 4.8789e-19 | 2.7756e-17 | 2.8031e-17 | true | false | stacked_exact | 9.910788 | 0.024171 | 2.421830 |
| 240.000000 | 2.1142e-18 | 1.3878e-17 | 1.9611e-17 | true | false | stacked_exact | 7.213203 | 0.015016 | 1.510067 |
| 260.000000 | 7.3726e-18 | 4.4755e-16 | 4.5382e-16 | true | false | stacked_exact | 94.307959 | 0.710841 | 72.228269 |
| 280.000000 | 8.6736e-19 | 0.000000 | 6.7549e-18 | true | false | stacked_exact | 35.784742 | 0.754710 | 38.665928 |
| 300.000000 | 2.3852e-17 | 2.2204e-16 | 2.4833e-16 | true | false | stacked_exact | 117.646667 | 0.622284 | 48.107438 |
| 320.000000 | 5.5716e-17 | 5.5511e-17 | 8.8549e-17 | true | false | stacked_exact | 93.169404 | 0.483773 | 30.490032 |
| 340.000000 | 5.7784e-17 | 1.1444e-16 | 1.3985e-16 | true | false | stacked_exact | 74.643803 | 0.377475 | 25.594127 |
| 360.000000 | 3.4694e-18 | 5.5511e-17 | 7.1967e-17 | true | false | stacked_exact | 59.619161 | 0.286068 | 20.400257 |
| 380.000000 | 2.8892e-17 | 5.5511e-17 | 6.6703e-17 | true | false | stacked_exact | 45.012085 | 0.211839 | 15.441738 |
| 400.000000 | 1.6653e-16 | 0.000000 | 1.0186e-15 | true | false | stacked_exact | 1430.982216 | 6.262591 | 498.519781 |
| 420.000000 | 8.3267e-17 | 1.8310e-15 | 1.8557e-15 | true | false | stacked_exact | 936.500252 | 1.956548 | 100.575535 |
| 440.000000 | 9.5410e-18 | 4.4755e-16 | 4.9030e-16 | true | false | stacked_exact | 170.152583 | 1.912737 | 150.752592 |
| 460.000000 | 8.8822e-16 | 8.8818e-16 | 1.3781e-15 | true | false | stacked_exact | 369.922669 | 1.068911 | 129.099072 |
| 480.000000 | 7.6328e-17 | 9.9301e-16 | 1.3447e-15 | true | false | stacked_exact | 581.844857 | 2.700232 | 266.442084 |
| 500.000000 | 1.0408e-17 | 4.9651e-16 | 4.9722e-16 | true | false | stacked_exact | 297.922749 | 2.000260 | 191.560550 |
| 520.000000 | 2.4286e-17 | 4.4409e-16 | 1.8137e-15 | true | false | stacked_exact | 1009.957369 | 4.217647 | 339.913037 |
| 540.000000 | 6.2805e-16 | 4.4409e-16 | 8.0690e-16 | true | false | stacked_exact | 497.784049 | 2.025998 | 58.416788 |
| 560.000000 | 4.8572e-17 | 0.000000 | 1.7413e-15 | true | false | stacked_exact | 804.767118 | 4.008599 | 456.330797 |
| 580.000000 | 4.7705e-18 | 2.2888e-16 | 2.4017e-16 | true | false | stacked_exact | 99.336039 | 1.329623 | 179.628932 |
| 600.000000 | 6.5919e-17 | 1.7764e-15 | 1.8020e-15 | true | false | stacked_exact | 729.550469 | 9.676471 | 281.530838 |
| 620.000000 | 1.0061e-16 | 4.4409e-16 | 1.3181e-15 | true | false | stacked_exact | 822.740855 | 4.931909 | 673.993051 |
| 640.000000 | 4.4519e-16 | 0.000000 | 6.9555e-16 | true | false | stacked_exact | 779.226870 | 2.733396 | 146.547206 |
| 660.000000 | 5.6400e-17 | 0.000000 | 6.8890e-17 | true | false | stacked_exact | 71.711902 | 1.559723 | 110.429309 |
| 680.000000 | 1.7347e-18 | 0.000000 | 1.0944e-16 | true | false | stacked_exact | 161.891160 | 0.391210 | 70.529758 |
| 700.000000 | 1.1112e-16 | 5.5511e-17 | 1.4739e-16 | true | false | stacked_exact | 74.531000 | 0.380369 | 57.618653 |
| 720.000000 | 1.3010e-17 | 1.1102e-16 | 1.2769e-16 | true | false | stacked_exact | 115.384166 | 0.231590 | 13.968835 |
| 740.000000 | 6.9600e-18 | 3.4694e-18 | 1.3028e-17 | true | false | stacked_exact | 12.313658 | 0.217102 | 15.248117 |
| 760.000000 | 2.6021e-18 | 0.000000 | 1.2339e-17 | true | false | stacked_exact | 25.754424 | 0.080556 | 13.183188 |
| 780.000000 | 1.2143e-17 | 2.2204e-16 | 2.8171e-16 | true | false | stacked_exact | 103.103562 | 1.206412 | 27.630589 |
| 800.000000 | 1.4225e-16 | 2.3915e-15 | 2.5550e-15 | true | false | stacked_exact | 1173.344992 | 5.832564 | 402.373426 |
| 820.000000 | 6.9389e-17 | 1.3323e-15 | 1.3476e-15 | true | false | stacked_exact | 586.196596 | 0.693247 | 108.935873 |
| 840.000000 | 6.1583e-17 | 2.2204e-16 | 2.4069e-16 | true | false | stacked_exact | 214.038281 | 0.814025 | 88.745164 |
| 860.000000 | 1.9082e-17 | 2.2204e-16 | 2.7103e-16 | true | false | stacked_exact | 199.347112 | 0.542788 | 69.604332 |
| 880.000000 | 1.8215e-17 | 0.000000 | 9.9815e-17 | true | false | stacked_exact | 128.183542 | 0.376213 | 39.203146 |
| 900.000000 | 8.6736e-18 | 2.2204e-16 | 2.2713e-16 | true | false | stacked_exact | 77.066879 | 0.250373 | 24.886577 |
| 920.000000 | 1.0842e-18 | 0.000000 | 5.5034e-17 | true | false | stacked_exact | 49.727328 | 0.157024 | 15.967096 |
| 940.000000 | 2.8322e-17 | 2.7756e-17 | 5.9833e-17 | true | false | stacked_exact | 31.914751 | 0.096776 | 9.824032 |
| 960.000000 | 3.4694e-18 | 0.000000 | 9.2710e-18 | true | false | stacked_exact | 20.638037 | 0.059498 | 6.060514 |
| 980.000000 | 2.2768e-18 | 2.7756e-17 | 3.0006e-17 | true | false | stacked_exact | 13.771445 | 0.036468 | 3.746008 |
| 1000.000000 | 2.7867e-17 | 2.7756e-17 | 3.7324e-17 | true | false | stacked_exact | 9.538562 | 0.022371 | 2.308379 |
| 1020.000000 | 2.1142e-18 | 1.3878e-17 | 1.4968e-17 | true | false | stacked_exact | 6.935818 | 0.013769 | 1.424407 |
| 1040.000000 | 6.5052e-19 | 2.2204e-16 | 2.2317e-16 | true | false | stacked_exact | 45.463054 | 0.370160 | 29.341110 |
| 1060.000000 | 1.9949e-17 | 1.1102e-16 | 1.2952e-16 | true | false | stacked_exact | 391.635163 | 4.276068 | 278.767957 |
| 1080.000000 | 1.2143e-16 | 8.8818e-16 | 1.0898e-15 | true | false | stacked_exact | 932.082195 | 6.222059 | 460.740084 |
| 1100.000000 | 1.0408e-17 | 1.3323e-15 | 1.4681e-15 | true | false | stacked_exact | 930.267101 | 4.112469 | 349.478437 |
| 1120.000000 | 1.0582e-16 | 2.2204e-16 | 2.6219e-16 | true | false | stacked_exact | 455.457245 | 2.125447 | 149.677412 |
| 1140.000000 | 3.4694e-17 | 5.5511e-16 | 5.5992e-16 | true | false | stacked_exact | 260.984699 | 1.100971 | 99.102645 |
| 1160.000000 | 2.6021e-17 | 2.2204e-16 | 2.2499e-16 | true | false | stacked_exact | 153.011038 | 0.528641 | 54.896440 |
| 1180.000000 | 1.4311e-17 | 1.1102e-16 | 1.5349e-16 | true | false | stacked_exact | 83.541383 | 0.257048 | 29.834949 |
| 1200.000000 | 8.9348e-16 | 4.4409e-16 | 1.8086e-15 | true | false | stacked_exact | 1353.807859 | 6.325733 | 475.586306 |
| 1220.000000 | 1.5613e-16 | 8.8818e-16 | 1.1880e-15 | true | false | stacked_exact | 992.073919 | 1.920854 | 104.498728 |
| 1240.000000 | 6.0715e-18 | 3.3307e-16 | 3.3427e-16 | true | false | stacked_exact | 150.233720 | 1.978144 | 143.561355 |
| 1260.000000 | 4.5103e-17 | 0.000000 | 2.6844e-16 | true | false | stacked_exact | 300.389171 | 0.823051 | 120.517092 |
| 1280.000000 | 1.7902e-15 | 9.1551e-16 | 2.4064e-15 | true | false | stacked_exact | 411.554377 | 2.361130 | 253.543480 |
| 1300.000000 | 4.3368e-19 | 0.000000 | 2.2266e-16 | true | false | stacked_exact | 185.150255 | 2.130637 | 185.669658 |
| 1320.000000 | 1.1581e-16 | 0.000000 | 4.2682e-16 | true | false | stacked_exact | 502.324081 | 6.525317 | 357.038281 |
| 1340.000000 | 8.6736e-18 | 8.8818e-16 | 9.0565e-16 | true | false | stacked_exact | 281.111848 | 2.775175 | 376.603157 |
| 1360.000000 | 1.3878e-17 | 0.000000 | 3.7930e-16 | true | false | stacked_exact | 639.809467 | 2.273427 | 176.989587 |
| 1380.000000 | 1.3444e-17 | 2.7756e-17 | 6.6414e-17 | true | false | stacked_exact | 76.664080 | 1.165068 | 78.773677 |
| 1400.000000 | 4.4538e-16 | 4.4409e-16 | 5.5555e-16 | true | false | stacked_exact | 249.501103 | 0.817735 | 88.671156 |
| 1420.000000 | 1.5179e-18 | 1.1444e-16 | 1.2380e-16 | true | false | stacked_exact | 39.126770 | 1.123702 | 47.808306 |
| 1440.000000 | 6.4185e-17 | 2.2204e-16 | 3.9543e-16 | true | false | stacked_exact | 362.362787 | 1.696352 | 142.796424 |
| 1460.000000 | 4.1633e-17 | 2.4825e-16 | 3.4163e-16 | true | false | stacked_exact | 267.832072 | 1.471221 | 75.839582 |
| 1480.000000 | 2.4286e-17 | 2.2204e-16 | 2.6339e-16 | true | false | stacked_exact | 215.880224 | 1.157691 | 77.475071 |
| 1500.000000 | 4.4414e-16 | 0.000000 | 3.6961e-16 | true | false | stacked_exact | 208.664611 | 0.935069 | 66.624108 |
| 1520.000000 | 1.9082e-17 | 2.2888e-16 | 2.5557e-16 | true | false | stacked_exact | 159.764623 | 0.729461 | 49.559034 |
| 1540.000000 | 9.5410e-18 | 0.000000 | 1.0521e-16 | true | false | stacked_exact | 123.841879 | 0.542902 | 39.784710 |
| 1560.000000 | 2.6021e-18 | 0.000000 | 3.1298e-17 | true | false | stacked_exact | 95.975159 | 0.397998 | 29.871650 |
| 1580.000000 | 7.3726e-18 | 0.000000 | 6.1611e-17 | true | false | stacked_exact | 70.202382 | 0.284374 | 21.652870 |

## Box-Constrained Analysis Summary

| pct_exact_bounded | pct_exact_unbounded_fallback_bounded_ls | pct_exact_unsolved_fallback_bounded_ls | pct_failed | pct_exact_solutions_inside_bounds | avg_exact_bound_violation_inf | max_exact_bound_violation_inf | avg_bounded_residual_norm | max_bounded_residual_norm | avg_us_exact_minus_us_bounded_inf | max_us_exact_minus_us_bounded_inf | avg_xs_exact_minus_xs_bounded_inf | max_xs_exact_minus_xs_bounded_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.125000 | 98.875000 | 0.000000 | 0.000000 | 1.125000 | 251.555767 | 1300.413626 | 1.689017 | 16.975734 | 253.208155 | 1300.413626 | 67.205972 | 365.840745 |

## Box Solve Mode Counts

| solve_mode | count |
| --- | --- |
| exact_bounded | 18.000000 |
| exact_unbounded_fallback_bounded_ls | 1582.000000 |

## Per-Input Bound Activity

| input_index | fraction_lower_bound_active | fraction_upper_bound_active | average_exact_violation_below_lower | average_exact_violation_above_upper |
| --- | --- | --- | --- | --- |
| 0.000000 | 0.373125 | 0.476875 | 105.168340 | 136.919079 |
| 1.000000 | 0.270625 | 0.251250 | 65.275637 | 63.068035 |

## Box Event Table

| event_kind | event_anchor | k | solve_mode | exact_eq_residual_state_inf | exact_eq_residual_output_inf | bounded_residual_norm | dhat_delta_inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| setpoint_change | 400.000000 | 395.000000 | exact_unbounded_fallback_bounded_ls | 5.8547e-18 | 2.7756e-17 | 0.068221 | 0.001713 |
| setpoint_change | 400.000000 | 396.000000 | exact_unbounded_fallback_bounded_ls | 8.6736e-19 | 2.7756e-17 | 0.066277 | 0.001687 |
| setpoint_change | 400.000000 | 397.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 0.000000 | 0.064356 | 0.001664 |
| setpoint_change | 400.000000 | 398.000000 | exact_unbounded_fallback_bounded_ls | 1.5179e-18 | 5.5511e-17 | 0.062458 | 0.001639 |
| setpoint_change | 400.000000 | 399.000000 | exact_unbounded_fallback_bounded_ls | 5.9631e-18 | 2.7756e-17 | 0.060584 | 0.001613 |
| setpoint_change | 400.000000 | 400.000000 | exact_unbounded_fallback_bounded_ls | 1.6653e-16 | 0.000000 | 6.122888 | 0.001589 |
| setpoint_change | 400.000000 | 401.000000 | exact_unbounded_fallback_bounded_ls | 8.3267e-17 | 1.7764e-15 | 6.122501 | 0.001565 |
| setpoint_change | 400.000000 | 402.000000 | exact_unbounded_fallback_bounded_ls | 2.7062e-16 | 4.4409e-16 | 5.711927 | 0.750387 |
| setpoint_change | 400.000000 | 403.000000 | exact_unbounded_fallback_bounded_ls | 0.000000 | 3.9968e-15 | 5.329305 | 0.855327 |
| setpoint_change | 400.000000 | 404.000000 | exact_unbounded_fallback_bounded_ls | 3.2613e-16 | 2.2204e-16 | 5.054689 | 0.814088 |
| setpoint_change | 400.000000 | 405.000000 | exact_unbounded_fallback_bounded_ls | 7.6328e-17 | 3.9968e-15 | 4.871963 | 0.755883 |
| setpoint_change | 800.000000 | 795.000000 | exact_unbounded_fallback_bounded_ls | 1.3878e-17 | 2.2204e-16 | 0.564259 | 0.049229 |
| setpoint_change | 800.000000 | 796.000000 | exact_unbounded_fallback_bounded_ls | 1.9082e-17 | 2.2204e-16 | 0.596091 | 0.056078 |
| setpoint_change | 800.000000 | 797.000000 | exact_unbounded_fallback_bounded_ls | 2.6888e-17 | 0.000000 | 0.622729 | 0.061516 |
| setpoint_change | 800.000000 | 798.000000 | exact_unbounded_fallback_bounded_ls | 1.3878e-17 | 4.4409e-16 | 0.649760 | 0.065598 |
| setpoint_change | 800.000000 | 799.000000 | exact_unbounded_fallback_bounded_ls | 2.3419e-17 | 4.4409e-16 | 0.678199 | 0.068390 |
| setpoint_change | 800.000000 | 800.000000 | exact_unbounded_fallback_bounded_ls | 1.4225e-16 | 2.2204e-15 | 5.620331 | 0.069938 |
| setpoint_change | 800.000000 | 801.000000 | exact_unbounded_fallback_bounded_ls | 3.4694e-17 | 0.000000 | 5.657922 | 0.070367 |
| setpoint_change | 800.000000 | 802.000000 | exact_unbounded_fallback_bounded_ls | 7.6328e-17 | 2.2204e-15 | 5.497538 | 0.227861 |
| setpoint_change | 800.000000 | 803.000000 | exact_unbounded_fallback_bounded_ls | 2.4286e-17 | 8.8818e-16 | 5.198941 | 0.460374 |
