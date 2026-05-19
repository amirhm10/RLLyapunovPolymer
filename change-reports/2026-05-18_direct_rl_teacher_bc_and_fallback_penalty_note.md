# Direct RL Teacher BC And Fallback Penalty Note

## Summary

Updated the direct safety-gated RL notebooks so the initial online behavioral-cloning block starts immediately from direct Lyapunov MPC teacher behavior, with no separate warmup phase and no BC exploration noise for the diagnostic run.

Also extended the direct method step-by-step report with a proposed fallback-activation penalty:

`J_fallback = gamma_fallback * I_fallback * ||u_rl - u_exec||_2^2`

## Files

- `DirectLyapunovSafetyGateRL_Pretrained.ipynb`
- `DirectLyapunovSafetyGateRL_ColdStart.ipynb`
- `report/direct_lyapunov_direct_and_rl_method_2026-05-16.md`

## Rationale

The latest pretrained result suggested the loaded actor was being trusted too early. The pretrained notebook previously allowed the BC phase to clone executed policy actions, which can amplify a mismatched checkpoint. The new diagnostic setup makes both pretrained and cold-start runs begin with the same direct-MPC teacher behavior.

The fallback penalty is documented as a next reward extension, not yet implemented. It would penalize the actor when the safety gate must replace the RL proposal with a different executed action, giving the critic a direct signal that large safety corrections are undesirable.

## Validation

- Validated both edited notebooks with `nbformat.validate`.
- Scanned the edited report for non-ASCII characters.
