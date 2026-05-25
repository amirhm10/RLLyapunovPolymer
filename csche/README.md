# CSChE Lyapunov-Guided RL Presentation Draft

This folder contains the first CSChE conference-style Beamer draft for:

**Lyapunov-Guided Reinforcement Learning for Practical Polymer CSTR Control**

## Files

- `csche_lyapunov_rl_presentation.tex`  
  Main Beamer source.

- `figures/`  
  Copied and generated slide figures. These are derived from the latest local Lyapunov/RL report and saved result bundles.

- `scripts/generate_csche_figures.py`  
  Reproducible figure-preparation script. It copies verified report figures and generates compact CSChE-specific plots.

- `notes_for_speaker.md`  
  Speaker notes and slide-by-slide intent.

- `change_report_csche_first_draft.md`  
  Summary of files created, sources inspected, compile status, and unresolved issues.

## Evidence Sources

The first draft is based primarily on:

- `report/rl_agent_authority_bc_latest_analysis_2026-05-19.md`
- `report/rl_agent_authority_bc_latest_analysis_2026-05-19.html`
- `report/chatgpt_repo_deep_research_prompt_2026-05-20.md`
- recent `change-reports/` files from May 18, May 19, and May 20, 2026
- the active root scripts for direct LMPC, cold-start RL, and pretrained RL
- the latest analyzed result folders:
  - `results/ColdStart/20260520_204513`
  - `results/Pretrain/20260520_205230`
  - `results/directLyap/20260520_204510`

## Important Configuration Note

The prompt mentioned an older context with `200` episodes and `lyap_eps = 1e-3`. The latest local report and active scripts currently describe the analyzed run as:

- `n_episodes = 300`
- `set_points_len = 400`
- `rho_lyap = 0.99`
- `lyap_eps = 1e-2`
- `GAMMA = 0.99`
- `fallback_event_penalty = 10.0`

The deck uses the latest report as the primary evidence and treats `lyap_eps = 1e-3` as an already-tried earlier setting.

## Technical Conference Revision

The main result slides now prioritize compact CSChE-generated figures instead of broad report screenshots:

- `csche_phase_authority_summary.png`
- `csche_key_result_summary.png`
- `csche_final_episode_tracking_summary.png`
- `csche_target_selector_mechanism.png`

## Rebuild Figures

From the repository root:

```powershell
python csche\scripts\generate_csche_figures.py
```

## Compile

From the repository root:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=csche csche\csche_lyapunov_rl_presentation.tex
```

If `pdflatex` is not on the path, install a LaTeX distribution or compile from an IDE that provides Beamer support.
