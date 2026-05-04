# StatsControl2026 Project 3 RL Math Slide

## Summary

Added a new Project 3 mathematics slide to `StatsControl2026/stats_control_2026_slides.tex` that explains the direct-Lyapunov RL deployment loop in equation form.

## What Changed

- defined the Project 3 RL state as the augmented estimate, setpoint, and previous input
- showed the actor map from TD3 action to bounded plant input
- added the direct target calculator equation
- added the Lyapunov acceptance inequality used to certify the RL proposal
- defined the fallback rule that sends rejected actions to direct Lyapunov MPC
- stated that the reward function and replay buffer are intentionally the same as in Project 2
- added compact TD3 replay/update equations to show that learning stays off-policy and unchanged

## Presentation Intent

This slide is method-only. It clarifies that Project 3 changes the plant-side action acceptance logic, not the reward design or replay structure.

## Validation

- compiled successfully with:
  - `pdflatex -interaction=nonstopmode -halt-on-error stats_control_2026_slides.tex`
- visually checked the rendered slide page after compile
