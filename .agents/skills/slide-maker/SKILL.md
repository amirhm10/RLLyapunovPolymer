---
name: slide-maker
description: Use this skill when the user asks to create, revise, audit, polish, or scientifically improve slides, Beamer presentations, talks, conference presentations, thesis slides, defense slides, research update slides, or presentation figures. Trigger especially for tasks mentioning slides, Beamer, LaTeX slides, presentation, talk, figures, plots, flowcharts, results, journal comparison, paper figures, or PhD-level research storytelling.
---

# Slide Maker

## Overview

Create or improve academic research slides by reading the repository first, testing each claim against evidence, and then shaping the deck into a strong oral research story. In this repository, prioritize scientific judgment over decoration and push back on slide wording that overclaims what the results actually show.

## Start With Repository Evidence

Inspect the repository before writing or editing slides. In this project, check these files and folders first unless the user points elsewhere:

- `MACC2026/macc2026_poster_slides.tex`
- `MACC2026/macc2026_3m_three_column_slide.tex`
- `MACC2026/research_summary_2026_draft.tex`
- `main.tex`
- `revised.tex`
- `Response to Reviewers.tex`
- `ref_lib.bib`
- `acs-main.bib`
- `MACC2026/research_summary_2026_draft.bib`
- `Figures/`
- `MACC2026/Figures/`

Infer and reuse the local research context:

- The central theme is practical reinforcement learning for chemical process control with MPC kept in the loop.
- The main case studies are the styrene polymerization reactor and the Aspen Dynamics `C_2` splitter.
- The recurring method families are MPC-pretrained TD3, offset-aware reward shaping, mixed replay, RL-assisted MPC, and Lyapunov-filtered safe RL.
- The recurring baselines and labels include OF-MPC, `RL_1`, `RL_2`, residual correction, weight tuning, horizon tuning, and model multipliers.
- The existing slide style in `MACC2026/` uses `Copenhagen`, a maroon structure color, muted blue/green/red accents, compact blocks, and `\graphicspath{{./}{Figures/}{MACC2026/Figures/}}`.

Preserve local notation whenever it already exists, especially symbols like `\eta`, `T`, `x_{24,\mathrm{C_2H_6}}`, `T_{85}`, `Q`, `R`, `N_p`, and `N_c`.

If the task is ambiguous, make the best scientifically reasonable assumption, continue, and record the assumption in the final summary. In this repo, default to a research update or conference-style technical audience unless the artifacts clearly indicate a defense, class talk, or journal response.

## Follow This Workflow

### 1. Understand the scientific context first

- Map existing Beamer decks, TeX files, BibTeX files, reports, images, notebooks, result folders, and plotting scripts.
- Identify the main method being presented.
- Identify the case study or system being discussed.
- Infer the intended audience from filenames, nearby documents, slide density, and tone.
- Identify the current theme, color palette, figure conventions, and notation.
- Do not stop for clarification unless multiple interpretations would materially change the science or create overwrite risk.

### 2. Start from the scientific story

Build the deck around the argument, not the decoration.

- Give every slide one clear message.
- Prefer conclusion-style slide titles over topic-only titles.
- Use a logical flow: motivation, gap, method, formulation, implementation, results, interpretation, limitations, next steps.
- Avoid overcrowded frames.
- Prefer figures, schematics, and short comparisons over dense paragraphs.
- Use equations only when they help the audience understand the method or a result.
- Use simple academic English.
- Reuse the user's writing style when it is already clear and technically sound.
- Avoid semicolons in prose.
- Put mathematics in LaTeX math mode.

For PhD-style research storytelling in this repo, usually tell one of these arcs:

1. Why linear or fixed MPC becomes insufficient in nonlinear or drifting operation.
2. What practical RL modification is added without discarding MPC.
3. How the controller is implemented on the polymer reactor or Aspen Dynamics `C_2` splitter.
4. What evidence shows improvement over OF-MPC or an earlier RL baseline.
5. What still remains limited, noisy, or unfinished.

### 3. Reason scientifically on every technical slide

Do not merely make slides prettier. For each technical slide, identify:

- the claim
- the evidence
- what the figure, table, or equation is supposed to prove
- what limitation or uncertainty still remains

Check whether the result actually supports the stated conclusion. If results are weak, noisy, contradictory, or incomplete, present them honestly and add the most useful next experiment.

For RL and MPC slides, explicitly check:

- What is the baseline
- What is the proposed method
- What is held constant across comparisons
- What metric is optimized
- Whether reward improvement agrees with tracking improvement
- Whether gains come from RL itself, reward shaping, mixed replay, residual action, model update, or another change
- Whether constraints, offset-free behavior, disturbance handling, and input movement are treated fairly
- Whether the plots support the final claim

If the evidence is not strong enough, tighten the claim instead of overstating the result.

### 4. Create or improve figures when the evidence needs it

Actively look for opportunities to improve figures instead of only reusing old ones. Inspect notebooks, CSV files, NumPy files, pickle files, MATLAB files, PDFs, images, and existing plotting scripts when they are present.

Useful figure types include:

- tracking plots with outputs, setpoints, and tolerance bands
- manipulated-input trajectories with bounds
- reward curves and smoothed reward curves
- IAE, ISE, RMSE, final offset, settling time, and constraint-violation summaries
- OF-MPC vs `RL_1` vs `RL_2` comparisons
- ablation plots
- algorithm flowcharts
- closed-loop block diagrams
- timelines for pretraining, warm start, online fine-tuning, and evaluation
- conceptual diagrams for replay-buffer design, reward shaping, residual policies, model re-identification, and Lyapunov filtering

Follow these figure rules:

- Do not create misleading figures.
- Do not hide poor performance.
- Do not compare methods unless the setup is fair.
- Keep labels, legends, and units readable from a slide.
- Give every figure a short, proper caption that states what the audience should notice.
- When you create a figure yourself, do not label it as "Created figure" on the slide. Write a normal scientific caption instead.
- Use line styles and markers that still work in grayscale.
- Note deviation coordinates, scaling, or normalization when needed.
- Keep labels consistent across the deck.
- Save generated figures in a local, reproducible place such as `Figures/`, `MACC2026/Figures/`, `slides/figures/`, or a task-specific figure folder near the deck.
- Keep the source script or notebook reproducible.

In this repo, prefer to preserve local conventions such as red dashed setpoint lines, clear separation of reward plots from tracking plots, and naming that distinguishes nominal, fluctuation, ramp, and last-episode evaluations.

### 5. Add flowcharts or diagrams when the method is hard to parse from text

When the method is algorithmic, create a clean flowchart or block diagram if it improves comprehension. Common diagrams in this repo include:

- OF-MPC data generation -> behavior cloning -> online TD3 fine-tuning
- offset-free observer and MPC loop
- RL-assisted MPC architecture
- residual policy wrapped around MPC
- Lyapunov or safety filter with fallback logic
- mixed replay with prioritized, recent, and uniform sampling
- reward-shaping logic near the setpoint
- model re-identification + MPC + RL update loop

Use TikZ when the deck is already LaTeX-heavy or when text fidelity matters. Otherwise use Python to create clean SVG, PDF, or PNG figures with large fonts. If starting from scratch and no better theme exists, reuse `assets/repo_beamer_palette.tex` for a local color and panel baseline.

### 6. Connect the deck to papers carefully

Read the relevant paper sections before citing them. Prefer citations already present in the local `.bib` files. When comparing to papers:

- explain what is similar
- explain what is different
- explain what the contribution is
- distinguish background from direct comparison evidence

Never invent citations. Never claim that a paper supports a statement unless that support has been verified from the paper or from reliable metadata.

Do not copy copyrighted plots from papers unless permission is clear. Prefer one of these:

- cite the paper and summarize the point in your own words
- redraw a simplified conceptual version
- create a comparison table
- create a placeholder slide stating what external figure is still needed and why

### 7. Edit Beamer and LaTeX responsibly

When editing an existing deck:

- preserve the existing theme unless there is a clear reason to change it
- keep the existing color scheme, title style, block style, and bibliography style
- keep the source maintainable
- avoid manual spacing hacks when a cleaner structure exists
- use macros for repeated notation
- use small but readable captions
- keep figure widths consistent
- avoid overfull boxes when possible
- visually inspect the compiled slide pages and make sure text, figures, legends, and diagrams actually fit on the slide without crowding, clipping, or awkward whitespace

When creating a new deck in this repo:

- start from a professional academic structure
- include title, motivation, gap, method, results, interpretation, limitations, next steps, and backup slides
- include bibliography support using the local bibliography style already used nearby
- use placeholders only when evidence is unavailable
- put long derivations, hyperparameters, extra plots, and ablations in backup slides
- visually inspect the resulting pages and iterate until the layout looks presentation-ready, not only source-valid

Compile the deck when possible with the engine already used in the project. If changes introduce compile errors, fix them. If an issue remains unresolved, report it clearly.

### 8. Use audit mode for existing slides

When asked to review or improve existing slides, audit:

- story flow
- slide titles
- visual density
- figure readability
- mathematical consistency
- notation consistency
- claims versus evidence
- citation quality
- whether each slide has one clear message
- whether the overall deck looks like PhD-level research work

Lead with the most important scientific and communication issues first. Use `scripts/audit_beamer.py` for a quick local pass, then do a manual scientific review.

### 9. Use result-to-slide mode when the user gives results first

When asked to make slides from results:

- find the relevant result files
- identify the experiment setup
- identify baselines and variants
- compute or extract useful metrics when possible
- create clean plots where useful
- write slide conclusions from evidence
- add one or more interpretation slides
- add limitations and next-experiment slides

Avoid making the result look stronger than it is.

### 10. Use report-to-slide mode when the user gives a paper or report first

When asked to turn a report into slides:

- read the report first
- extract the main narrative, equations, figures, results, and claims
- compress paragraphs into slide-level messages
- keep detailed derivations in backup slides
- preserve citations
- produce a deck that can be spoken aloud, not a copied report

In this repo, `main.tex`, `revised.tex`, and `MACC2026/research_summary_2026_draft.tex` are the first places to mine for narrative and notation.

### 11. End each task with a clear summary

At the end of a slide task, summarize:

- what files were created or changed
- what figures were created or updated
- what assumptions were made
- what was compiled or tested
- any unresolved issues
- the most useful next improvements

### 12. Protect the repo and the science

- Do not delete existing slides, figures, notebooks, or reports unless explicitly asked.
- Do not overwrite important files when creating a clearly named new version is safer.
- Do not modify research code unless figure generation or analysis truly requires it.
- Keep generated scripts reproducible.
- Prefer adding new files over risky edits.
- Never fabricate results, citations, or paper claims.

## Continuation Rule For This Repo

When the user starts an ongoing slide task in this repository, continue using this skill implicitly across follow-up turns until the user clearly changes topic. Do not require the user to restate `slide-maker` on every slide-related message in the same working thread.

## Use the Bundled Helpers

Read the relevant reference file only when it helps the current task:

- `references/slide_quality_checklist.md`
- `references/phd_research_story_structure.md`
- `references/beamer_style_checklist.md`
- `references/figure_quality_checklist.md`
- `references/paper_comparison_checklist.md`
- `references/rl_mpc_slide_checklist.md`

Use the local scripts when they save time:

- `scripts/audit_beamer.py` checks for missing images, word-heavy frames, overfull-prone lines, and placeholder text.
- `scripts/collect_slide_assets.py` inventories likely slide inputs such as TeX, BibTeX, figures, PDFs, notebooks, and result files.
- `scripts/check_figures.py` checks figure existence, basic dimensions, and naming quirks.

Use these helpers to support scientific slide work, not to replace judgment.
