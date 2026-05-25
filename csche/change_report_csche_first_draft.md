# CSChE First Draft Change Report

## Files Created

- `csche/csche_lyapunov_rl_presentation.tex`
- `csche/README.md`
- `csche/notes_for_speaker.md`
- `csche/change_report_csche_first_draft.md`
- `csche/scripts/generate_csche_figures.py`
- `csche/figures/` with copied and generated slide assets

## Reports, Code, And Results Inspected

- `report/rl_agent_authority_bc_latest_analysis_2026-05-19.md`
- `report/rl_agent_authority_bc_latest_analysis_2026-05-19.html`
- `report/chatgpt_repo_deep_research_prompt_2026-05-20.md`
- recent May 18, May 19, and May 20 change reports
- `DirectLyapunovMPC.py`
- `DirectLyapunovSafetyGateRL_ColdStart.py`
- `DirectLyapunovSafetyGateRL_Pretrained.py`
- `utils/direct_lyapunov_study.py`
- `Lyapunov/target_selector.py`
- `Lyapunov/frozen_output_disturbance_target.py`
- `Lyapunov/direct_lyapunov_mpc.py`
- `Simulation/run_rl_lyapunov.py`
- `TD3Agent/reward_functions.py`
- latest analyzed result folders:
  - `results/ColdStart/20260520_204513`
  - `results/Pretrain/20260520_205230`
  - `results/directLyap/20260520_204510`

## Figures Reused Or Generated

Copied from the latest report figure folder:

- `performance_runtime_summary.png`
- `rl_authority_diagnostics.png`
- `tail_offset_comparison.png`
- `last_episode_tracking_primary_methods.png`
- `mpc_only_would_be_activation.png`
- `target_diagnostics_summary.png`
- `activation_contraction_episode_counts.png`
- `reward_penalty_scale.png`

Copied from the Stats and Control deck assets:

- `logo.png`
- `CSTR.png`

Generated under `csche/figures/`:

- `csche_key_result_summary.png`
- `csche_phase_authority_summary.png`
- `csche_target_selector_mechanism.png`
- `csche_final_episode_tracking_summary.png`

The generated figures are reproducible with:

```powershell
python csche\scripts\generate_csche_figures.py
```

## Main Slide Story

The draft frames the work as a technical conference presentation on practical RL/MPC integration:

1. MPC is the deployment baseline for constrained process control.
2. RL is allowed to propose actions, but the Lyapunov safety gate keeps final authority.
3. Pretraining gives better full-horizon reward and RMSE.
4. Cold-start RL currently has better safety-gate authority.
5. Direct LMPC settles well near the final target, but has worse full-horizon raw-setpoint RMSE and slower runtime.
6. The central unresolved issue is target selection: a target can be admissible for Lyapunov contraction but still poor for raw setpoint tracking.

## Configuration Note

The user prompt mentioned an older `200` episode and `lyap_eps = 1e-3` context. The latest local report and active scripts currently use:

- `n_episodes = 300`
- `set_points_len = 400`
- `rho_lyap = 0.99`
- `lyap_eps = 1e-2`
- `GAMMA = 0.99`
- `fallback_event_penalty = 10.0`

The deck follows the latest report as primary evidence and treats `lyap_eps = 1e-3` as an already-tried earlier setting.

## Compile Status

Compilation was attempted with:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=csche csche\csche_lyapunov_rl_presentation.tex
```

The normal compile hung while MiKTeX appeared to be waiting for package installation. A second validation run used:

```powershell
pdflatex -disable-installer -interaction=nonstopmode -halt-on-error -output-directory=csche csche\csche_lyapunov_rl_presentation.tex
```

That run failed before processing the deck because the local LaTeX installation is missing `beamer.cls`. No PDF was produced. The TeX source and all referenced figures were still validated for local file existence.

## Revision After Initial Draft

The first revision made the main result section more technical and used the cleaner CSChE-generated figures:

- Slide 2 now starts from the closed-loop control objective and Lyapunov certification inequality.
- Slide 9 now uses `csche_phase_authority_summary.png` instead of the broader report authority panel.
- Slide 11 now uses `csche_final_episode_tracking_summary.png`, generated directly from result arrays, instead of two crowded copied report figures.
- Speaker notes were updated to flag Slides 9 through 11 as the second review priority after the target-selector slide.

## Unresolved Issues

- This is a first draft. Slide density and figure readability should be reviewed visually.
- The PDF was not rendered because the local MiKTeX installation does not currently have Beamer installed.
- No new literature was added beyond local BibTeX entries already present in `MACC2026/research_summary_2026_draft.bib`.
- The current slide evidence is based on a single latest analyzed run, not seed averages.
- The deck intentionally avoids presenting target-selector variants as solved methods.

## What To Review Manually Next

Review Slide 12 first. It is the scientific hinge of the talk and should clearly show why target quality, not just safety filtering or reward shaping, is the central bottleneck.
