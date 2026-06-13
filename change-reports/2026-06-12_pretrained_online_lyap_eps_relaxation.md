# Pretrained Online Lyapunov Epsilon Relaxation

## Summary
Relaxed the Direct LMPC first-step contraction tolerance for pretrained online TD3 runners from the shared bounded-mixed default `1e-3` to `1e-2`.

## Rationale
- The critic-reset batch showed that the catastrophic early full-RL collapse was largely fixed, but safety-gate cases can still lose tracking during handoff.
- The active safety gate may worsen tracking if the contraction certificate is too strict relative to the moving bounded target and the freshly recalibrated online critic.
- This change isolates a more permissive practical contraction certificate for pretrained online TD3 only.

## Implementation Notes
- `PRETRAINED_ONLINE_LYAP_EPS = 1e-2` is used for all pretrained online TD3 presets.
- Cold-start online runners and MPC baselines keep the shared Direct LMPC default `LYAP_EPS = 1e-3`.
- No-gate pretrained runners use the same `1e-2` value for Direct LMPC monitor diagnostics, so monitor rates remain comparable to the active pretrained gate.
- Run configs record `lyap_eps`, `lyap_eps_default`, `lyap_eps_pretrained_online_override`, and `lyap_eps_override_reason`.

## Validation
- Static validation should compile `utils/online_disturbance_runner.py` and all six online TD3 root runners.
- Config checks should confirm pretrained presets report `lyap_eps = 0.01` while cold-start presets report `lyap_eps = 0.001`.
