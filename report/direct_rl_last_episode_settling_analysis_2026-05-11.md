# Direct RL Last-Episode Settling Analysis

Date: 2026-05-11

## Scope

This rewritten note updates the earlier settling analysis using the newer direct safety-gate RL runs with `rho_lyap = 0.99`.

The two questions remain:

1. Why did the final episode appear not to settle in the earlier runs?
2. Is the final episode still receiving exploration noise?

The new evidence comes from:

- pretrained latest complete run: `20260511_171056`
- cold-start latest complete run: `20260511_170643`

and is compared against the earlier runs used in the previous version of this note:

- pretrained earlier comparison run: `20260511_104912`
- cold-start earlier comparison run: `20260511_104852`

## Files inspected

- [DirectLyapunovSafetyGateRL_Pretrained.ipynb](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/DirectLyapunovSafetyGateRL_Pretrained.ipynb>)
- [DirectLyapunovSafetyGateRL_ColdStart.ipynb](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/DirectLyapunovSafetyGateRL_ColdStart.ipynb>)
- [Simulation/run_rl_lyapunov.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Simulation/run_rl_lyapunov.py>)
- [utils/helpers.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/utils/helpers.py>)
- [pretrained rho=0.99 comparison table](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260511_171056/comparison_table.csv>)
- [cold-start rho=0.99 comparison table](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260511_170643/comparison_table.csv>)
- [pretrained rho=0.98 comparison table](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_pretrained/20260511_104912/comparison_table.csv>)
- [cold-start rho=0.98 comparison table](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Data/debug_exports/rl_direct_safety_gate_four_method_two_setpoint_disturb_cold_start/20260511_104852/comparison_table.csv>)

## What changed between the old and new comparison sets

The relevant configuration change is not inferred from notebook memory. It is recorded directly inside the saved bundles:

- earlier comparison runs: `rho_lyap = 0.98`
- newer comparison runs: `rho_lyap = 0.99`

So this update is specifically a `rho_lyap` comparison, not a vague "latest run" comparison.

## Last-episode noise question

The answer is still no at the code-path level.

The final cycle is forced to be a test cycle in [utils/helpers.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/utils/helpers.py>), and test cycles map to deterministic behavior in [Simulation/run_rl_lyapunov.py](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/Simulation/run_rl_lyapunov.py>).

So the final episode should still be interpreted as deterministic evaluation, not as an exploratory rollout.

One remaining limitation is that the exported safety step tables still do not include an explicit `behavior_noise_mode` column, so this remains a code-level conclusion rather than a direct saved-column confirmation.

## Main updated conclusion

The new `rho_lyap = 0.99` results materially change the earlier interpretation.

For the bounded-hard cases that originally motivated the settling concern, `rho_lyap = 0.99` clearly improves final-episode settling, especially for output 2. That means your hypothesis was directionally right: the older non-settling behavior was significantly influenced by the contraction setting.

However, the effect is not universal across all four-method variants:

- bounded-hard improves in both pretrained and cold-start
- several other second-setpoint tails also improve
- some first-setpoint tails worsen
- the combined `u_prev + x_s_prev` case does not improve consistently

So `rho_lyap = 0.99` is a major part of the story, but not the only part.

## Figure

The figure below compares final-episode tail-settling error for output 2 across all four cases, for pretrained and cold-start, under `rho_lyap = 0.98` versus `rho_lyap = 0.99`.

![Final-episode settling comparison for rho](../last_episode_settling_rho99_compare_2026-05-11.png)

Figure 1. Final-episode tail mean absolute error to raw setpoint for output 2. Lower is better.

## Quantitative results

### Bounded-hard case: clear improvement with `rho_lyap = 0.99`

Final 50-step tail MAE to raw setpoint for output 2:

| Study | Seg 1, rho=0.98 | Seg 1, rho=0.99 | Seg 2, rho=0.98 | Seg 2, rho=0.99 |
| --- | ---: | ---: | ---: | ---: |
| Pretrained | 0.1431 | 0.0732 | 0.1593 | 0.1312 |
| Cold start | 0.1631 | 0.0981 | 0.1682 | 0.0744 |

This is the strongest updated result in the note. The bounded-hard final episode is substantially better with `rho_lyap = 0.99`, especially in the cold-start second setpoint segment.

### Segment-2 tail settling across all cases

Final 50-step tail MAE to raw setpoint for output 2:

| Case | Pretrained rho=0.98 | Pretrained rho=0.99 | Cold rho=0.98 | Cold rho=0.99 |
| --- | ---: | ---: | ---: | ---: |
| `bounded_hard` | 0.1593 | 0.1312 | 0.1682 | 0.0744 |
| `bounded_hard_u_prev_0p1` | 0.1380 | 0.1040 | 0.0527 | 0.0348 |
| `bounded_hard_xs_prev_0p1` | 0.0893 | 0.0577 | 0.0412 | 0.1565 |
| `bounded_hard_u_prev_0p1_xs_prev_0p1` | 0.0699 | 0.0618 | 0.0318 | 0.1028 |

Interpretation:

- In the second setpoint tail, `rho_lyap = 0.99` improves 6 of 8 study-case combinations.
- The clearest improvements are the bounded-hard cases and the pretrained `u_prev` and `x_s_prev` cases.
- The two regressions are both in cold-start regularized variants:
  - `bounded_hard_xs_prev_0p1`
  - `bounded_hard_u_prev_0p1_xs_prev_0p1`

### Segment-1 tail settling is more mixed

For the first setpoint tail, `rho_lyap = 0.99` improves only 3 of 8 study-case combinations.

That means the benefit of higher `rho_lyap` is more consistent near the second final target than near the first final target.

### Whole-run metrics can disagree with final-episode settling

Two cases are especially important here:

- pretrained `bounded_hard_u_prev_0p1` with `rho_lyap = 0.99`
- cold-start `bounded_hard_u_prev_0p1_xs_prev_0p1` with `rho_lyap = 0.99`

Their whole-run averages are poor:

- pretrained `u_prev`: `reward_mean = -346.7`, `output_rmse_mean = 7.31`
- cold-start combined: `reward_mean = -581.2`, `output_rmse_mean = 9.48`

But that does **not** mean the final episode tail is equally poor in every segment. So for this report's question, whole-run reward is a misleading proxy. The settling question has to be answered from the final-episode tail itself.

## Scientific interpretation

The updated evidence supports the following interpretation:

1. The earlier report was too cautious to center `rho_lyap` because it only had the `rho_lyap = 0.98` comparison set.
2. With the newer `rho_lyap = 0.99` runs available, the bounded-hard settling issue is clearly reduced.
3. So `rho_lyap` was indeed one of the important causes of the observed non-settling.
4. But the regularized variants still show case-dependent behavior, which means the final-episode shape is still also influenced by target regularization, accepted-versus-corrected action mix, and fallback interaction.

## Updated answer to the two original questions

### 1. Why were we not reaching and settling in the last episode?

For the bounded-hard family, the new evidence indicates that `rho_lyap = 0.98` was a meaningful part of the problem. Increasing it to `0.99` improves the final tail noticeably.

For the regularized variants, the answer is more nuanced. Some improve with `rho_lyap = 0.99`, while others do not. So those cases still have additional structure beyond the contraction factor alone.

### 2. Are we adding noise in the last episode?

Still no by design. The last episode is intended to be deterministic evaluation.

## Recommended next step

The most useful next experiment is now narrower than before.

Instead of asking whether the final episode is noisy, the better question is:

- why do the cold-start `x_s_prev` and combined regularized cases fail to benefit from `rho_lyap = 0.99` the way bounded-hard does?

The next targeted audit should therefore compare, for those two remaining problematic variants:

1. target mismatch near the last 100 steps,
2. accepted-candidate versus fallback counts in the last episode,
3. whether the selected target is moving more than in the improved bounded-hard case.

## Files changed

- [report/direct_rl_last_episode_settling_analysis_2026-05-11.md](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/direct_rl_last_episode_settling_analysis_2026-05-11.md>)
- [last_episode_settling_rho99_compare_2026-05-11.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/last_episode_settling_rho99_compare_2026-05-11.png>)

## Bottom line

The new complete `rho_lyap = 0.99` runs support your hypothesis for the main bounded-hard settling issue. The earlier apparent last-episode non-settling was significantly tied to using `rho_lyap = 0.98`. But `rho_lyap = 0.99` is not a complete universal fix, because a subset of the regularized variants still shows mixed or worse tail behavior.
