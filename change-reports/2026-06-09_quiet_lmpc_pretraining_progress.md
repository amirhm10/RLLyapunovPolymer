# Quiet LMPC Pretraining Solver Warnings And Add Chunk Progress

Date: 2026-06-09

## Summary

Reduced noisy LMPC pretraining console output while preserving diagnostics in result artifacts.

The Direct LMPC label-generation path can trigger repeated CVXPY warnings:

```text
Solution may be inaccurate. Try another solver, adjusting the solver settings, or solve with verbose=True for more information.
```

These warnings are now suppressed during LMPC pretraining label solves. Solver status and failure information remain available in `label_diagnostics.json`.

## Changes

- Updated `utils/lmpc_td3_workflow.py`.
  - Suppressed the CVXPY `Solution may be inaccurate` `UserWarning` inside `_label_candidate(...)`.
  - The suppression applies to both sequential and parallel LMPC label generation.
  - Added OF-MPC-style candidate chunk progress:

```text
Processing broad LMPC candidate chunk 1/20 (size=..., accepted=..., attempted=...)
```

- Updated `report/lmpc_td3_pretraining_process_2026-06-09.md`.
  - Documented the quiet warning behavior.
  - Documented that solver diagnostics are still written to artifacts.
  - Documented the chunk-progress print behavior.

## Validation

Static validation passed:

```powershell
python -m py_compile utils/lmpc_td3_workflow.py
```

Tiny sequential smoke validation passed:

```powershell
python PretrainTD3LyapunovMPC.py --lmpc-samples 1 --steady-samples 1 --candidate-chunk-size 2 --worker-batch-size 1 --label-n-jobs 1 --parallel-backend sequential --actor-epochs 1 --critic-epochs 1 --pretrain-batch-size 2 --device cpu
```

Result bundle:

```text
results/PretrainLMPCQuietSmoke/20260609_215836/
```

Observed console progress:

```text
Processing broad LMPC candidate chunk 1/10 (size=1, accepted=0/1, attempted=0/20)
Processing steady LMPC candidate chunk 1/10 (size=1, accepted=0/1, attempted=0/20)
```

Observed artifacts:

- accepted labels: `2`
- attempted candidates: `2`
- solve success rate: `1.0`
- actor BC loss entries: `1`
- critic TD loss entries: `1`
