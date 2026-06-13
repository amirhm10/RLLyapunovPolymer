# Restore Pretrained Online Lyapunov Epsilon Default

## Summary

Restored pretrained online TD3 disturbance runners to the bounded-mixed Direct LMPC default `lyap_eps=1e-3`. The previous pretrained-only `1e-2` relaxation is removed from future runs.

## Motivation

The `1e-2` pretrained relaxation made the handoff-calibrated batch harder to interpret because the batch changed both the handoff schedule and the Lyapunov contraction tolerance. Future pretrained runs should isolate the handoff and learning-phase design using the same bounded-mixed epsilon used by the direct LMPC defaults and cold-start runners.

## Code Changes

- Removed the pretrained-specific `PRETRAINED_ONLINE_LYAP_EPS = 1e-2` override.
- `_lyap_eps_for_preset(...)` now returns `LYAP_EPS` for every online TD3 preset.
- Run config metadata now records:
  - `lyap_eps = 0.001`
  - `lyap_eps_default = 0.001`
  - `lyap_eps_pretrained_online_override = None`
  - `lyap_eps_override_reason = "default bounded-mixed Direct LMPC epsilon"`

## Validation

Use:

```powershell
python -m py_compile utils/online_disturbance_runner.py
```

and verify pretrained and cold-start config probes both report `lyap_eps=0.001`.

## Historical Note

The handoff-calibrated analysis report remains a historical analysis of the `lyap_eps=1e-2` batch. New runs after this change should be interpreted as calibrated-handoff runs under the default bounded-mixed Lyapunov epsilon.

