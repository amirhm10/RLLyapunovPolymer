# Offset-Free MPC Steady-State Debug Summary

- Case: `first_step_contraction_bounded_frozen_dhat`
- Steps analyzed: `1600`
- Configured solver mode: `auto`
- Requested analysis target mode: `hybrid`
- Effective analysis target mode: `hybrid`
- Analysis labels: `x=x_s_exact`, `u=u_s_bounded`, `y=y_s_exact`, `d=d_s_exact`
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

| residual_dyn_norm_mean | residual_dyn_norm_std | residual_dyn_norm_max | residual_out_norm_mean | residual_out_norm_std | residual_out_norm_max | residual_total_norm_mean | residual_total_norm_std | residual_total_norm_max | u_applied_minus_u_s_norm_mean | u_applied_minus_u_s_norm_std | u_applied_minus_u_s_norm_max | y_current_minus_y_s_norm_mean | y_current_minus_y_s_norm_std | y_current_minus_y_s_norm_max | xhat_minus_x_s_norm_mean | xhat_minus_x_s_norm_std | xhat_minus_x_s_norm_max | rhs_output_norm_mean | rhs_output_norm_std | rhs_output_norm_max | u_s_dev_norm_mean | u_s_dev_norm_std | u_s_dev_norm_max | x_s_norm_mean | x_s_norm_std | x_s_norm_max | reduced_rhs_exact_residual_norm_mean | reduced_rhs_exact_residual_norm_std | reduced_rhs_exact_residual_norm_max | reduced_rhs_bounded_residual_norm_mean | reduced_rhs_bounded_residual_norm_std | reduced_rhs_bounded_residual_norm_max | analysis_u_applied_minus_u_s_norm_mean | analysis_u_applied_minus_u_s_norm_std | analysis_u_applied_minus_u_s_norm_max | analysis_y_current_minus_y_s_norm_mean | analysis_y_current_minus_y_s_norm_std | analysis_y_current_minus_y_s_norm_max | analysis_xhat_minus_x_s_norm_mean | analysis_xhat_minus_x_s_norm_std | analysis_xhat_minus_x_s_norm_max | analysis_rhs_output_norm_mean | analysis_rhs_output_norm_std | analysis_rhs_output_norm_max | analysis_u_s_dev_norm_mean | analysis_u_s_dev_norm_std | analysis_u_s_dev_norm_max | analysis_x_s_norm_mean | analysis_x_s_norm_std | analysis_x_s_norm_max | analysis_G_u_s_minus_rhs_norm_mean | analysis_G_u_s_minus_rhs_norm_std | analysis_G_u_s_minus_rhs_norm_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 6.2796e-17 | 2.8127e-16 | 3.5652e-15 | 1.7690e-16 | 4.3648e-16 | 5.3291e-15 | 2.4660e-16 | 5.4860e-16 | 7.1584e-15 | 146.211128 | 269.076301 | 1430.632585 | 0.530741 | 0.936694 | 6.285872 | 41.377915 | 69.799062 | 489.230086 | 0.640410 | 1.144209 | 6.278456 | 148.238111 | 268.043061 | 1425.563126 | 47.248064 | 86.836173 | 486.555898 | 9.4692e-17 | 2.2216e-16 | 1.9984e-15 | 0.621085 | 1.139082 | 6.129689 | 1.095009 | 1.920634 | 13.514799 | 0.530741 | 0.936694 | 6.285872 | 41.377915 | 69.799062 | 489.230086 | 0.640410 | 1.144209 | 6.278456 | 4.939569 | 1.529566 | 12.266122 | 47.248064 | 86.836173 | 486.555898 | 0.621085 | 1.139082 | 6.129689 |

## Sampled Per-Step Diagnostics

| k | analysis_target_variant | residual_dyn_norm | residual_out_norm | residual_total_norm | exact_solution | used_lstsq | solver_mode | analysis_u_applied_minus_u_s_norm | analysis_y_current_minus_y_s_norm | analysis_xhat_minus_x_s_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.000000 | hybrid | 9.3675e-17 | 8.8818e-16 | 9.4076e-16 | true | false | stacked_exact | 1.693153 | 2.798133 | 274.896971 |
| 20.000000 | hybrid | 7.6328e-17 | 0.000000 | 4.0820e-16 | true | false | stacked_exact | 4.585647 | 1.659080 | 115.662107 |
| 40.000000 | hybrid | 6.5919e-17 | 8.8818e-16 | 9.3489e-16 | true | false | stacked_exact | 2.409330 | 1.599883 | 117.350241 |
| 60.000000 | hybrid | 2.2423e-16 | 4.4409e-16 | 4.7961e-16 | true | false | stacked_exact | 1.208065 | 1.173945 | 100.199474 |
| 80.000000 | hybrid | 6.6787e-17 | 6.6613e-16 | 6.7855e-16 | true | false | stacked_exact | 0.937870 | 0.781749 | 64.011229 |
| 100.000000 | hybrid | 4.3368e-18 | 0.000000 | 1.0521e-16 | true | false | stacked_exact | 0.638453 | 0.522241 | 42.803776 |
| 120.000000 | hybrid | 1.3010e-18 | 3.3766e-16 | 3.4470e-16 | true | false | stacked_exact | 0.456478 | 0.356702 | 29.299867 |
| 140.000000 | hybrid | 4.3368e-18 | 1.1102e-16 | 1.1624e-16 | true | false | stacked_exact | 0.363999 | 0.256265 | 20.317369 |
| 160.000000 | hybrid | 2.3852e-18 | 5.5511e-17 | 5.5740e-17 | true | false | stacked_exact | 0.306548 | 0.196496 | 14.816589 |
| 180.000000 | hybrid | 8.2399e-18 | 0.000000 | 2.5217e-17 | true | false | stacked_exact | 0.271571 | 0.160817 | 11.499149 |
| 200.000000 | hybrid | 2.9145e-17 | 5.5511e-17 | 6.7915e-17 | true | false | stacked_exact | 0.251113 | 0.139557 | 9.508983 |
| 220.000000 | hybrid | 6.2101e-17 | 0.000000 | 7.7933e-17 | true | false | stacked_exact | 0.239054 | 0.126962 | 8.336096 |
| 240.000000 | hybrid | 1.7347e-18 | 0.000000 | 2.0907e-17 | true | false | stacked_exact | 0.232146 | 0.119605 | 7.656326 |
| 260.000000 | hybrid | 2.7972e-17 | 0.000000 | 3.0308e-17 | true | false | stacked_exact | 0.228539 | 0.115462 | 7.269474 |
| 280.000000 | hybrid | 2.8746e-17 | 2.7756e-17 | 4.3838e-17 | true | false | stacked_exact | 0.226945 | 0.113312 | 7.059893 |
| 300.000000 | hybrid | 4.7705e-18 | 0.000000 | 1.0150e-17 | true | false | stacked_exact | 0.226621 | 0.112405 | 6.958663 |
| 320.000000 | hybrid | 2.2768e-18 | 5.5511e-17 | 6.1713e-17 | true | false | stacked_exact | 0.227128 | 0.112288 | 6.924160 |
| 340.000000 | hybrid | 2.6021e-18 | 5.5511e-17 | 6.1599e-17 | true | false | stacked_exact | 0.228200 | 0.112682 | 6.931681 |
| 360.000000 | hybrid | 3.4694e-18 | 0.000000 | 1.1118e-17 | true | false | stacked_exact | 0.229654 | 0.113423 | 6.967511 |
| 380.000000 | hybrid | 6.2138e-17 | 2.7756e-17 | 7.6764e-17 | true | false | stacked_exact | 0.231385 | 0.114400 | 7.021760 |
| 400.000000 | hybrid | 4.1633e-16 | 2.2204e-15 | 2.6226e-15 | true | false | stacked_exact | 8.590751 | 6.281434 | 480.917302 |
| 420.000000 | hybrid | 3.4694e-17 | 1.7764e-15 | 1.8426e-15 | true | false | stacked_exact | 9.670519 | 1.707156 | 94.827276 |
| 440.000000 | hybrid | 2.2551e-17 | 0.000000 | 4.8793e-17 | true | false | stacked_exact | 1.351255 | 1.860557 | 145.875477 |
| 460.000000 | hybrid | 1.7347e-18 | 4.4409e-16 | 5.8575e-16 | true | false | stacked_exact | 0.691973 | 0.804357 | 115.607140 |
| 480.000000 | hybrid | 4.3368e-18 | 6.6613e-16 | 6.8074e-16 | true | false | stacked_exact | 1.484718 | 0.669283 | 45.996645 |
| 500.000000 | hybrid | 2.1684e-18 | 2.2204e-16 | 2.3119e-16 | true | false | stacked_exact | 0.350147 | 0.475983 | 39.051248 |
| 520.000000 | hybrid | 4.3368e-19 | 1.1102e-16 | 1.4366e-16 | true | false | stacked_exact | 0.189729 | 0.317158 | 30.840263 |
| 540.000000 | hybrid | 5.5606e-17 | 2.7756e-17 | 8.5320e-17 | true | false | stacked_exact | 0.243209 | 0.227000 | 17.941690 |
| 560.000000 | hybrid | 2.7769e-17 | 0.000000 | 3.5925e-17 | true | false | stacked_exact | 0.048954 | 0.141901 | 13.088738 |
| 580.000000 | hybrid | 6.7221e-18 | 1.3878e-17 | 2.8904e-17 | true | false | stacked_exact | 0.038936 | 0.080316 | 8.428659 |
| 600.000000 | hybrid | 2.6021e-18 | 2.7756e-17 | 3.6133e-17 | true | false | stacked_exact | 0.072104 | 0.036103 | 4.761890 |
| 620.000000 | hybrid | 1.7618e-19 | 2.7756e-17 | 3.5534e-17 | true | false | stacked_exact | 0.123684 | 0.005276 | 2.701463 |
| 640.000000 | hybrid | 2.3039e-19 | 3.4694e-18 | 8.3593e-18 | true | false | stacked_exact | 3.290997 | 0.026158 | 1.948582 |
| 660.000000 | hybrid | 4.6079e-19 | 3.4694e-18 | 8.0232e-18 | true | false | stacked_exact | 4.982974 | 0.045543 | 2.626412 |
| 680.000000 | hybrid | 2.2768e-18 | 1.1102e-16 | 1.2311e-16 | true | false | stacked_exact | 1.412377 | 0.219965 | 27.708779 |
| 700.000000 | hybrid | 2.3852e-18 | 0.000000 | 1.4838e-17 | true | false | stacked_exact | 0.061724 | 0.291689 | 23.673038 |
| 720.000000 | hybrid | 1.8648e-17 | 1.1444e-16 | 1.1933e-16 | true | false | stacked_exact | 0.428463 | 0.343207 | 25.395106 |
| 740.000000 | hybrid | 9.9747e-18 | 0.000000 | 4.1515e-17 | true | false | stacked_exact | 0.335465 | 0.270893 | 15.126682 |
| 760.000000 | hybrid | 2.1684e-18 | 5.5511e-17 | 7.5694e-17 | true | false | stacked_exact | 0.087958 | 0.189382 | 14.655170 |
| 780.000000 | hybrid | 3.5779e-18 | 5.5511e-17 | 6.7343e-17 | true | false | stacked_exact | 0.071742 | 0.130818 | 10.037127 |
| 800.000000 | hybrid | 3.0531e-16 | 2.3915e-15 | 2.6531e-15 | true | false | stacked_exact | 9.053023 | 6.285872 | 481.522863 |
| 820.000000 | hybrid | 1.5959e-16 | 1.3506e-15 | 1.3666e-15 | true | false | stacked_exact | 7.219403 | 0.967265 | 109.293172 |
| 840.000000 | hybrid | 4.5103e-17 | 4.4409e-16 | 6.1710e-16 | true | false | stacked_exact | 2.433760 | 1.438445 | 118.736195 |
| 860.000000 | hybrid | 1.5613e-17 | 4.4409e-16 | 6.2586e-16 | true | false | stacked_exact | 1.049537 | 1.101205 | 107.777133 |
| 880.000000 | hybrid | 4.4416e-16 | 2.2204e-16 | 4.8793e-16 | true | false | stacked_exact | 1.106148 | 0.807006 | 66.647021 |
| 900.000000 | hybrid | 1.1224e-16 | 1.1102e-16 | 1.5056e-16 | true | false | stacked_exact | 0.770021 | 0.592175 | 47.672231 |
| 920.000000 | hybrid | 4.3368e-18 | 1.1102e-16 | 1.5414e-16 | true | false | stacked_exact | 0.558594 | 0.430538 | 34.611636 |
| 940.000000 | hybrid | 9.1073e-18 | 1.1102e-16 | 1.5570e-16 | true | false | stacked_exact | 0.431824 | 0.315308 | 24.792063 |
| 960.000000 | hybrid | 2.1684e-18 | 1.1102e-16 | 1.2151e-16 | true | false | stacked_exact | 0.330694 | 0.232236 | 18.041601 |
| 980.000000 | hybrid | 5.4210e-18 | 1.1102e-16 | 1.1592e-16 | true | false | stacked_exact | 0.254413 | 0.170730 | 13.186938 |
| 1000.000000 | hybrid | 5.2042e-18 | 5.5511e-17 | 5.5944e-17 | true | false | stacked_exact | 0.196444 | 0.124332 | 9.589998 |
| 1020.000000 | hybrid | 5.5957e-17 | 3.1032e-17 | 4.5048e-17 | true | false | stacked_exact | 0.151297 | 0.088772 | 6.889423 |
| 1040.000000 | hybrid | 4.3368e-19 | 2.7756e-17 | 2.8084e-17 | true | false | stacked_exact | 0.115894 | 0.061177 | 4.831248 |
| 1060.000000 | hybrid | 8.6736e-19 | 0.000000 | 3.6257e-18 | true | false | stacked_exact | 0.087959 | 0.039580 | 3.248461 |
| 1080.000000 | hybrid | 3.1984e-18 | 1.5516e-17 | 1.6300e-17 | true | false | stacked_exact | 0.065798 | 0.022620 | 2.034629 |
| 1100.000000 | hybrid | 2.1142e-18 | 2.7756e-17 | 2.9337e-17 | true | false | stacked_exact | 0.048157 | 0.009533 | 1.142901 |
| 1120.000000 | hybrid | 1.6263e-18 | 1.4305e-17 | 1.4606e-17 | true | false | stacked_exact | 2.763093 | 0.004392 | 0.670992 |
| 1140.000000 | hybrid | 4.6079e-19 | 2.1943e-17 | 2.2075e-17 | true | false | stacked_exact | 0.842844 | 0.011399 | 0.803932 |
| 1160.000000 | hybrid | 9.7578e-19 | 2.1104e-17 | 2.1413e-17 | true | false | stacked_exact | 0.903997 | 0.018048 | 1.166134 |
| 1180.000000 | hybrid | 5.6921e-19 | 3.4694e-18 | 3.7836e-18 | true | false | stacked_exact | 2.119466 | 0.023477 | 1.511798 |
| 1200.000000 | hybrid | 1.4572e-16 | 8.8818e-16 | 1.8485e-15 | true | false | stacked_exact | 8.494883 | 6.275806 | 489.230086 |
| 1220.000000 | hybrid | 1.7764e-15 | 9.9301e-16 | 1.5120e-15 | true | false | stacked_exact | 9.610165 | 1.753518 | 100.377006 |
| 1240.000000 | hybrid | 2.1684e-17 | 1.1102e-16 | 1.3088e-16 | true | false | stacked_exact | 1.348970 | 1.881645 | 148.710624 |
| 1260.000000 | hybrid | 8.8835e-16 | 4.4409e-16 | 9.1819e-16 | true | false | stacked_exact | 0.580195 | 0.850474 | 117.887398 |
| 1280.000000 | hybrid | 4.4235e-17 | 2.2204e-16 | 2.5938e-16 | true | false | stacked_exact | 1.701226 | 0.763694 | 49.410338 |
| 1300.000000 | hybrid | 2.2248e-16 | 2.2204e-16 | 3.2229e-16 | true | false | stacked_exact | 0.565116 | 0.593611 | 44.473848 |
| 1320.000000 | hybrid | 1.6914e-17 | 1.1102e-16 | 1.6314e-16 | true | false | stacked_exact | 0.459597 | 0.472670 | 39.345013 |
| 1340.000000 | hybrid | 8.6736e-19 | 2.2888e-16 | 2.5227e-16 | true | false | stacked_exact | 0.521359 | 0.396129 | 27.287835 |
| 1360.000000 | hybrid | 9.9747e-18 | 0.000000 | 6.3495e-17 | true | false | stacked_exact | 0.315027 | 0.316582 | 23.132388 |
| 1380.000000 | hybrid | 1.1102e-16 | 0.000000 | 1.2498e-16 | true | false | stacked_exact | 0.256151 | 0.257254 | 18.868512 |
| 1400.000000 | hybrid | 1.3878e-17 | 1.6883e-16 | 1.7244e-16 | true | false | stacked_exact | 0.201583 | 0.208929 | 14.837577 |
| 1420.000000 | hybrid | 3.6863e-18 | 5.5511e-17 | 6.2258e-17 | true | false | stacked_exact | 0.138553 | 0.168273 | 12.145483 |
| 1440.000000 | hybrid | 1.6263e-18 | 1.3878e-17 | 1.7910e-17 | true | false | stacked_exact | 0.100341 | 0.136245 | 9.803936 |
| 1460.000000 | hybrid | 8.6736e-19 | 1.3878e-17 | 1.7765e-17 | true | false | stacked_exact | 0.067536 | 0.110472 | 7.913086 |
| 1480.000000 | hybrid | 3.9031e-18 | 5.5511e-17 | 6.3006e-17 | true | false | stacked_exact | 0.041418 | 0.089852 | 6.454793 |
| 1500.000000 | hybrid | 1.8431e-18 | 5.5511e-17 | 5.7629e-17 | true | false | stacked_exact | 0.024122 | 0.073587 | 5.273822 |
| 1520.000000 | hybrid | 2.1684e-19 | 0.000000 | 1.4298e-17 | true | false | stacked_exact | 0.016987 | 0.060731 | 4.342961 |
| 1540.000000 | hybrid | 5.5511e-17 | 1.9626e-17 | 4.3471e-17 | true | false | stacked_exact | 0.021142 | 0.050625 | 3.613697 |
| 1560.000000 | hybrid | 1.0300e-18 | 1.9626e-17 | 2.1242e-17 | true | false | stacked_exact | 0.028312 | 0.042718 | 3.038507 |
| 1580.000000 | hybrid | 7.1591e-18 | 2.8610e-17 | 3.3225e-17 | true | false | stacked_exact | 0.034951 | 0.036539 | 2.588721 |

## Box-Constrained Analysis Summary

| pct_exact_bounded | pct_exact_unbounded_fallback_bounded_ls | pct_exact_unsolved_fallback_bounded_ls | pct_failed | pct_exact_solutions_inside_bounds | avg_exact_bound_violation_inf | max_exact_bound_violation_inf | avg_bounded_residual_norm | max_bounded_residual_norm | avg_us_exact_minus_us_bounded_inf | max_us_exact_minus_us_bounded_inf | avg_xs_exact_minus_xs_bounded_inf | max_xs_exact_minus_xs_bounded_inf |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8.625000 | 91.375000 | 0.000000 | 0.000000 | 8.625000 | 121.683930 | 1193.360836 | 0.621085 | 6.129689 | 129.568351 | 1199.708626 | 32.467897 | 361.138639 |

## Box Solve Mode Counts

| solve_mode | count |
| --- | --- |
| exact_bounded | 138.000000 |
| exact_unbounded_fallback_bounded_ls | 1462.000000 |

## Per-Input Bound Activity

| input_index | fraction_lower_bound_active | fraction_upper_bound_active | average_exact_violation_below_lower | average_exact_violation_above_upper |
| --- | --- | --- | --- | --- |
| 0.000000 | 0.000000 | 6.2500e-04 | 61.473287 | 59.957887 |
| 1.000000 | 0.000000 | 0.000000 | 32.111882 | 28.236190 |

## Box Event Table

| event_kind | event_anchor | k | solve_mode | exact_eq_residual_state_inf | exact_eq_residual_output_inf | bounded_residual_norm | dhat_delta_inf |
| --- | --- | --- | --- | --- | --- | --- | --- |
| setpoint_change | 400.000000 | 395.000000 | exact_unbounded_fallback_bounded_ls | 2.7756e-17 | 2.7756e-17 | 0.131468 | 9.1889e-05 |
| setpoint_change | 400.000000 | 396.000000 | exact_unbounded_fallback_bounded_ls | 2.7756e-17 | 2.7756e-17 | 0.131524 | 9.2466e-05 |
| setpoint_change | 400.000000 | 397.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 8.3267e-17 | 0.131580 | 9.2621e-05 |
| setpoint_change | 400.000000 | 398.000000 | exact_unbounded_fallback_bounded_ls | 2.9273e-18 | 5.5511e-17 | 0.131636 | 9.2494e-05 |
| setpoint_change | 400.000000 | 399.000000 | exact_unbounded_fallback_bounded_ls | 4.0115e-18 | 2.7756e-17 | 0.131693 | 9.2634e-05 |
| setpoint_change | 400.000000 | 400.000000 | exact_unbounded_fallback_bounded_ls | 4.1633e-16 | 2.2204e-15 | 6.129689 | 9.3037e-05 |
| setpoint_change | 400.000000 | 401.000000 | exact_unbounded_fallback_bounded_ls | 3.5527e-15 | 8.8818e-16 | 6.110597 | 9.3465e-05 |
| setpoint_change | 400.000000 | 402.000000 | exact_unbounded_fallback_bounded_ls | 5.5511e-17 | 1.7764e-15 | 5.865095 | 0.416510 |
| setpoint_change | 400.000000 | 403.000000 | exact_unbounded_fallback_bounded_ls | 3.5527e-15 | 1.7764e-15 | 5.567594 | 0.552948 |
| setpoint_change | 400.000000 | 404.000000 | exact_unbounded_fallback_bounded_ls | 1.3878e-17 | 2.2204e-15 | 5.278200 | 0.598399 |
| setpoint_change | 400.000000 | 405.000000 | exact_unbounded_fallback_bounded_ls | 1.5266e-16 | 2.4425e-15 | 5.017075 | 0.613339 |
| setpoint_change | 800.000000 | 795.000000 | exact_unbounded_fallback_bounded_ls | 3.5779e-18 | 0.000000 | 0.080677 | 0.002307 |
| setpoint_change | 800.000000 | 796.000000 | exact_unbounded_fallback_bounded_ls | 4.2284e-18 | 5.5511e-17 | 0.078755 | 0.002315 |
| setpoint_change | 800.000000 | 797.000000 | exact_unbounded_fallback_bounded_ls | 2.8189e-18 | 0.000000 | 0.076933 | 0.002312 |
| setpoint_change | 800.000000 | 798.000000 | exact_unbounded_fallback_bounded_ls | 5.0958e-18 | 5.5511e-17 | 0.075213 | 0.002299 |
| setpoint_change | 800.000000 | 799.000000 | exact_unbounded_fallback_bounded_ls | 8.6736e-19 | 5.5511e-17 | 0.073599 | 0.002277 |
| setpoint_change | 800.000000 | 800.000000 | exact_unbounded_fallback_bounded_ls | 3.0531e-16 | 2.2204e-15 | 6.064601 | 0.002246 |
| setpoint_change | 800.000000 | 801.000000 | exact_unbounded_fallback_bounded_ls | 1.0408e-16 | 3.5527e-15 | 6.056168 | 0.002207 |
| setpoint_change | 800.000000 | 802.000000 | exact_unbounded_fallback_bounded_ls | 1.5266e-16 | 1.7764e-15 | 5.883289 | 0.294011 |
| setpoint_change | 800.000000 | 803.000000 | exact_unbounded_fallback_bounded_ls | 8.3267e-17 | 4.4409e-16 | 5.603385 | 0.523316 |
