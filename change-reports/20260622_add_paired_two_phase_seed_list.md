# Add Paired Two-Phase Seed List

Date: 2026-06-22

## Summary

Added an explicit fixed seed list to all five two-phase method runners so paper runs are paired across methods.

## Seed List

All runners now use:

```python
PAPER_SEEDS = (42, 7, 19, 73, 101, 203, 307, 401, 557, 809)
SEEDS = PAPER_SEEDS
N_SEEDS = len(PAPER_SEEDS)
```

This gives the same seed labels and randomization seeds for:

- OF-MPC-pretrained safety gate
- OF-MPC-pretrained no safety gate
- cold-start safety gate
- cold-start no safety gate
- GART-LMPC baseline

For quick sequential debugging, set `SEEDS = None` and use `N_SEEDS` plus `SEED_START`.

## Validation

- Imported all five runner modules and verified each resolves to `42,7,19,73,101,203,307,401,557,809`.
- Passed `py_compile` on all five two-phase runner files.
