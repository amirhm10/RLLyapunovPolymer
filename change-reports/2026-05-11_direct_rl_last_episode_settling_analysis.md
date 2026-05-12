# Direct RL Last-Episode Settling Analysis

Date: 2026-05-11

## Summary

Rewrote the direct RL last-episode settling note using the newer complete `rho_lyap = 0.99` runs for both pretrained and cold-start studies, and compared them directly against the earlier `rho_lyap = 0.98` runs.

## Updated artifacts

- [report/direct_rl_last_episode_settling_analysis_2026-05-11.md](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/report/direct_rl_last_episode_settling_analysis_2026-05-11.md>)
- [last_episode_settling_rho99_compare_2026-05-11.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/last_episode_settling_rho99_compare_2026-05-11.png>)
- [last_episode_output_tracking_rho99_compare_2026-05-11.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/last_episode_output_tracking_rho99_compare_2026-05-11.png>)
- [pretrained_anchor_bias_tracking_2026-05-12.png](</c:/Users/HAMEDI/OneDrive - McMaster University/PythonProjects/Lyapunov_polymer/pretrained_anchor_bias_tracking_2026-05-12.png>)

The new figure is embedded directly in the report.

## Main conclusions captured in the report

- The last episode is still structurally a deterministic test episode, so behavior noise is not the primary explanation.
- The new `rho_lyap = 0.99` runs materially improve the bounded-hard final-episode settling behavior that motivated the earlier concern.
- In the second setpoint tail for output 2, `rho_lyap = 0.99` improves 6 of 8 study-case combinations.
- A new output-tracking figure and RMSE comparison were added so the report now compares actual bounded-hard trajectories and whole-run tracking, not just tail-settling bars.
- The latest pretrained anchored run shows that jitter can be removed while still leaving a steady-state offset, because the selected target is being pulled away from `y_sp` by the anchor term.
- The `rho_lyap` change is therefore a major part of the old settling problem, but not a universal fix for every regularized variant.

## Validation

- Inspected the saved bundle configs to confirm the old comparison set used `rho_lyap = 0.98` and the new set used `rho_lyap = 0.99`.
- Recomputed final-episode tail metrics from the saved bundles for all four cases in both studies.
- Generated a new settling comparison figure from the updated runs.
- Added a new pretrained anchor-bias figure and tail-bias table using the latest complete pretrained run `20260511_213850`.
