# CSCHE Slide Archive And Pretraining Defaults

Date: 2026-06-10

## Summary

Archived the CSCHE 2026 presentation materials and the May 25 latest-run analysis report, then aligned the local TD3 pretraining runner defaults with the smaller three-layer, 256-unit pilot architecture.

## Files Added

- `CSCHE2026/slides.tex`
- `CSCHE2026/slides.pdf`
- `CSCHE2026/speaker_notes_15min.html`
- `CSCHE2026/figures/latest_runner_slide_figures.tex`
- `CSCHE2026/figures/*.png`
- `CSCHE2026/a-practical-mpc-pretrained-reinforcement-learning-framework-for-complex-process-systems (1).pdf`
- `report/latest_all_runners_extended_analysis_2026-05-25.md`

## Code Changes

- `PretrainTD3OffsetFreeMPC.py`
  - Reduces the default OF-MPC label count from 4.9 million to 2.0 million.
  - Uses `[256, 256, 256]` actor and critic defaults to match the current pilot scale.
- `ComparePretrainedTD3OffsetFreeMPC.py`
  - Uses `[256, 256, 256]` actor and critic defaults for checkpoint comparison.
- `ComparePretrainedTD3LyapunovMPC.py`
  - Uses `[256, 256, 256]` actor and critic defaults for checkpoint comparison.
  - Removes duplicate overwritten `[512, 512, 512, 512, 512]` constants.

## Scientific Context

The archived CSCHE materials summarize the pre-governed safety-gated RL story:

- OF-MPC demonstrations initialize a TD3 actor by behavior cloning.
- The online actor proposes an action, but a one-step Lyapunov contraction gate has final authority.
- Rejected actions are replaced by fallback LMPC.
- Pretrained gated RL gives the best full-window tracking among the three main methods in the May 25 analysis.
- The no-gate pretrained diagnostic tracks slightly better but would trigger the Lyapunov gate often, so it is not a certified controller.

This should be read alongside the governed-reference reports, which later made the target modification explicit through:

$$
y_{sp,k} \rightarrow r_k \rightarrow (x_{s,k},u_{s,k},y_{s,k}).
$$

The governed-reference result changes the mechanism interpretation. The target selector is no longer only a hidden local target. It becomes an explicit command governor plus steady target layer, while the stage objective should continue tracking the raw setpoint.

## Validation

Planned low-cost validation:

```powershell
python -m py_compile ComparePretrainedTD3LyapunovMPC.py ComparePretrainedTD3OffsetFreeMPC.py PretrainTD3OffsetFreeMPC.py
```
