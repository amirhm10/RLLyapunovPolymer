# GART Progress Prints

## Summary
Added direct-runner-style progress printing to the GART closed-loop rollout. The old governed-reference baseline already prints through the direct Lyapunov runner; the GART raw, mixed, and mixed-soft cases now print the same sub-episode reward and solver diagnostics.

## Printed Fields
- sub-episode index
- average reward over the completed sub-episode
- target mode
- Lyapunov mode
- plant mode
- solver success
- target stage
- Lyapunov contraction margin
- Lyapunov slack
- solver iteration count

## Validation
- `python -m py_compile experiments/run_gart_target_selector_study.py GARTLyapunovMPC.py`
