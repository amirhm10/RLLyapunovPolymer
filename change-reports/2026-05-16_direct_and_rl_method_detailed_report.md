# 2026-05-16 Direct And RL Method Detailed Report

## Summary

Added a new consolidated report:

- [report/direct_lyapunov_direct_and_rl_method_2026-05-16.md](../report/direct_lyapunov_direct_and_rl_method_2026-05-16.md)

and updated:

- [report/README.md](../report/README.md)

## What the new report covers

- the direct Lyapunov MPC method without RL
- the direct Lyapunov safety-gated RL method with a pretrained TD3 initialization
- the direct Lyapunov safety-gated RL method with a cold-start TD3 agent
- the mathematical form of the direct target selector, Lyapunov gate, fallback MPC, reward, and replay transition
- the difference between offline MPC-to-TD3 pretraining and the online teacher-driven behavioral-cloning phase in the direct RL notebooks

## Key clarifications captured

- The pretrained notebook loads a saved TD3 checkpoint but still runs a 20-cycle online teacher-driven BC phase before full RL.
- The cold-start notebook currently uses the same BC phase schedule, but starts from fresh random weights.
- The online BC phase stores the executed safe action in both the replay buffer and BC buffer, not the raw teacher action before filtering.
- The Lyapunov certificate is centered on the admissible steady target, while the reward and current tracking objective still reference the raw requested setpoint.
- The current direct RL notebook calls do not activate hard move-bound checks in the gate.
- The report was reformatted to avoid raw LaTeX notation in `.md` files and use plain-text equations that render cleanly in a normal IDE Markdown preview.

## Validation

- source-level inspection only
- no code execution changes were made
