# Use GART-LMPC For Cold-Start Fallback And BC

## Summary
- Switched both cold-start online TD3 presets to use `gart_lmpc` teacher behavior during warmup, behavior cloning, and handoff.
- Switched the cold-start safety-gate fallback controller from Direct LMPC tracking to GART-LMPC.
- Kept RL state construction, TD3 action scaling, rewards, pretrained presets, and the existing GART target selector path unchanged.

## Implementation Notes
- `run_rl_train(...)` now accepts optional `gart_mpc_config` and `fallback_controller` inputs.
- GART-LMPC teacher and fallback actions reuse the GART target context from `prepare_direct_output_disturbance_step(...)`, then call `solve_gart_lmpc_step(...)`.
- The no-safety cold-start runner still has `safety_gate_active=False`; it uses GART-LMPC for BC/handoff teacher actions and GART target diagnostics only.
- GART-LMPC solver failures keep the solver's hold-previous behavior instead of falling back to Direct LMPC.

## Logging
- Run summaries now record `teacher_source`, `fallback_controller`, JSON-safe `gart_lmpc_config`, and the raw/hard GART-LMPC objective mode.
- Step debug records include a `fallback_controller` column in addition to `fallback_mode`.

## Validation
- Passed:
  ```powershell
  python -m py_compile Simulation/run_rl_lyapunov.py Lyapunov/direct_lyapunov_mpc.py utils/online_disturbance_runner.py utils/gart_defaults.py Lyapunov/safety_debug.py
  ```

## Follow-Up Checks
- Run `pytest tests/test_gart_target.py` when `pytest` is available.
- Run short cold-start smoke checks when the scientific Python stack is available.
