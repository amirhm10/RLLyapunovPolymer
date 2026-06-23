# Restore Sequential Two-Phase Seeds

Date: 2026-06-22

## Objective

Use simple sequential paired seeds for the stochastic two-phase TD3 runners. This keeps the experiment design easy to reproduce and makes `N_SEEDS = 5` mean seeds `0, 1, 2, 3, 4` in every TD3 method.

## Changes

- Removed custom `PAPER_SEEDS` tuples from the four TD3 two-phase runners.
- Set `SEEDS = None`, `SEED_START = 0`, and `N_SEEDS = 10` in each TD3 runner.
- Updated comments to explain that changing `N_SEEDS` controls the sequential seed count.
- Set the deterministic GART-LMPC reference seed to `0` and kept `N_SEEDS = 1`.

## Notes

- This preserves paired-seed fairness across TD3 methods because all four stochastic runners now use the same generated sequence.
- To run five TD3 seeds, set `N_SEEDS = 5`; this gives seeds `0, 1, 2, 3, 4`.
- To debug a specific case, set `SEEDS = (123,)` or another explicit tuple.

## Validation

Compiled the five two-phase runner files:

```powershell
$env:PYTHONPYCACHEPREFIX="$env:TEMP\codex_pycache"
C:\Users\HAMEDI\miniconda3\envs\rl\python.exe -m py_compile RunTwoPhase_OFMPCPretrained_SafetyGate.py RunTwoPhase_OFMPCPretrained_NoSafetyGate.py RunTwoPhase_ColdStart_SafetyGate.py RunTwoPhase_ColdStart_NoSafetyGate.py RunTwoPhase_GART_LMPC.py
```
