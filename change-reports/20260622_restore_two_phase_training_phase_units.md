# Restore Two-Phase Training Phase Units

## Objective

Preserve the old online TD3 meaning of warmup and handoff episode counts after the two-phase runner changed console/reporting windows from 800-step cycles to 400-step reporting windows.

## Issue

The new two-phase profile keeps Phase 1 as true 800-step learning episodes:

- 400 steps at setpoint 1
- 400 steps at setpoint 2

However the rollout compatibility layer reports every 400 steps. The downstream TD3 phase normalizer multiplies `behavior_clone_teacher_episodes` and `handoff_episodes` by `time_in_sub_episodes`, so a configured value of `10` became `10 * 400 = 4000` steps instead of the old `10 * 800 = 8000` steps.

This was especially risky for pretrained no-gate runs because the pretrained actor is retained, the critic is reset, and the no-gate runner applies policy actions without safety intervention.

## Changes

- Added training-phase scaling in `RunOnlineTD3TwoPhaseStudy.py`.
- The shared runner now computes:

  `phase1_episode_len / reporting_window_steps = 800 / 400 = 2`

- The following training-phase counts are multiplied by this ratio before calling the old online TD3 loop:
  - `warmup_buffer_only_episodes`
  - `behavior_clone_teacher_episodes`
  - `handoff_episodes`
- Added metadata to the scaled override dict:
  - `configured_learning_episode_steps`
  - `rollout_reporting_window_steps`
  - `learning_episode_to_reporting_window_multiplier`

## Validation

- Checked all four TD3 two-phase runners:
  - `ofmpc_pretrained_no_safety_gate`: `10 -> 20` teacher windows and `10 -> 20` handoff windows.
  - `ofmpc_pretrained_safety_gate`: `10 -> 20` teacher windows and `10 -> 20` handoff windows.
  - `cold_start_no_safety_gate`: `10 -> 20` teacher windows and `10 -> 20` handoff windows.
  - `cold_start_safety_gate`: `10 -> 20` teacher windows and `10 -> 20` handoff windows.
- This restores the intended step durations:
  - teacher/critic warmup: `8000` steps.
  - handoff: `8000` steps.
  - full RL starts at step `16000`.
- In-memory Python syntax compile passed for `RunOnlineTD3TwoPhaseStudy.py`.
