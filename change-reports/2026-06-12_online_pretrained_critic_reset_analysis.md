# Online Pretrained Critic-Reset Analysis

## Summary
Added a code-side analysis report for the four pretrained online TD3 disturbance runners completed after the critic-reset and tiny online-BC-noise change.

## Artifacts
- Analysis script: `analysis/online_pretrained_critic_reset_analysis.py`
- Report: `report/online_pretrained_critic_reset_analysis_2026-06-12.md`
- Figures: `report/figures/2026-06-12_online_pretrained_critic_reset_analysis/`
- Tables: `report/tables/2026-06-12_online_pretrained_critic_reset_analysis/`

## Main Findings
- Resetting the pretrained critic while retaining the pretrained actor removes the catastrophic early full-RL collapse observed in the no-reset low-noise batch.
- LMPC-pretrained cases become the strongest current pretrained runs.
- OF-MPC-pretrained cases recover in early full RL and tail behavior, but show a localized handoff failure around episode 23.
- The next implementation target should keep critic reset but make handoff more conservative, likely by delaying TD3 actor-gradient updates during the handoff window.

## Validation
- `py_compile` passed for the analysis script.
- Generated figures and Markdown report were visually scanned for consistency with the computed tables.
