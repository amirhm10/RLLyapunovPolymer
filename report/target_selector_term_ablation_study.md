# Target Selector Term Ablation Study

## Purpose

This study isolates the refined Step A target-selector objective terms while keeping the rest of the safety-MPC pipeline fixed. The goal is to compare which selector terms matter most for the resulting closed-loop behavior, target quality, and safety-filter activity.

The notebook for this study is:

- `LyapunovSafetyFilterMPCTargetSelectorTermAblation.ipynb`

It mirrors the structure of `LyapunovSafetyFilterMPC.ipynb` and changes only the selector objective activation mask.

## Fixed Controller Settings

Across all study runs, the following are held fixed:

- the plant, disturbance scenario, and setpoint schedule
- the upstream offset-free MPC formulation
- the safety-filter backend and acceptance logic
- the refined selector constraints
- selector warm start behavior
- effective-target reuse (`target_backup_policy`)
- tracking-target policy
- debug export and plotting pipeline

The only intentional change between runs is which selector objective terms are active.

One explicit study-level exception is intentional:

- fallback MPC is disabled in the ablation notebook, so the comparison isolates selector-term changes without backup-MPC recovery masking the ablation effect

## Selector Objective Terms

The refined Step A selector solves a steady-state optimization with the decision variables `(x_s, u_s)` and fixed disturbance estimate `d_s = d_hat_k`.

Its objective contains five named terms:

1. `target_tracking`
   \[
   \|r_s - y_{sp}\|_{Q_r}^2
   \]
2. `u_applied_anchor`
   \[
   \|u_s - u_{\mathrm{applied},k}\|_{R_{u,\mathrm{ref}}}^2
   \]
3. `u_prev_smoothing`
   \[
   \|u_s - u_{s,\mathrm{prev}}\|_{R_{\Delta u,\mathrm{sel}}}^2
   \]
4. `x_prev_smoothing`
   \[
   \|x_s - x_{s,\mathrm{prev}}\|_{Q_{\Delta x}}^2
   \]
5. `xhat_anchor`
   \[
   \|x_s - \hat{x}_k\|_{Q_{x,\mathrm{ref}}}^2
   \]

The study uses an exact `term_activation` mask in the selector config. A masked-off term is omitted from the optimization objective entirely and is logged as `0.0` in the exported selector objective-term traces.

## Ablation Cases

The notebook runs the following fixed sweep:

- `all_terms_on`
- `objective_zero`
- `only_target_tracking`
- `only_u_applied_anchor`
- `only_u_prev_smoothing`
- `only_x_prev_smoothing`
- `only_xhat_anchor`

Interpretation:

- `all_terms_on` is the current refined-selector baseline.
- `objective_zero` keeps the selector constraints active but removes all five objective terms.
- each `only_*` case keeps exactly one objective term active and disables the other four.

## Comparison Outputs

For each study case, the notebook:

- runs the normal safety-MPC closed loop
- saves the standard safety debug export in its own subfolder
- appends one row to a comparison table

The comparison table includes both control and selector metrics:

- `study_name`
- `active_terms`
- `reward_mean`
- `target_error_inf_mean`
- `target_error_inf_max`
- `n_target_success`
- `n_qcqp_attempted`
- `n_qcqp_hard_accepted`
- `mode_counts`
- `output0_rmse`
- `output1_rmse`
- `debug_dir`

The RMSE columns are computed in physical output units using the post-step output trajectory against the physical setpoint schedule.

The notebook also saves compact comparison plots:

- mean reward by study
- per-output RMSE by study
- maximum target-error infinity norm by study

## How To Read The Results

This study is about **importance under closed-loop use**, not only raw objective magnitude.

Important interpretation rules:

- A term having a large logged value in the baseline run does not automatically mean it is the most important term.
- A term is more important if removing everything except that term or removing all terms except it changes:
  - tracking quality
  - reward
  - selector success
  - target mismatch
  - safety-filter intervention behavior
- `objective_zero` is a structural reference case. It shows what the selector does when only the steady-state and bound constraints remain.

The most informative comparison is usually:

- `all_terms_on` versus `objective_zero`
- `all_terms_on` versus each `only_*` run

This makes it easier to separate:

- constraint-driven selector behavior
- target-tracking-driven selector behavior
- smoothing and anchoring effects

## Implementation Notes

The exact term mask is implemented in:

- `Lyapunov/target_selector.py`

The study notebook uses the same safety export utilities as the standard safety-MPC notebook:

- `Lyapunov/safety_debug.py`

No other controller logic is intentionally changed by this ablation study.
