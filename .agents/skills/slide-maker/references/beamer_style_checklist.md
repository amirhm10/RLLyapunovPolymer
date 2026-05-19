# Beamer Style Checklist

Use this when editing or creating LaTeX slides.

## Local repo cues

- Existing MACC deck style uses `\usetheme{Copenhagen}`
- The structure color is maroon
- Support colors are muted blue, green, and red
- `\graphicspath{{./}{Figures/}{MACC2026/Figures/}}` is already common
- Compact blocks and panel-style figures fit the existing repo style

## Build rules

- Preserve theme and color choices unless the deck has a clear reason to change
- Keep captions short and readable
- Reuse macros for repeated notation
- Align figure widths within a slide
- Avoid empty filler bullets
- Avoid manual spacing commands unless they solve a specific layout problem
- Watch for overfull boxes after edits
- Compile after significant changes

## Readability rules

- Prefer one figure plus one conclusion over four mini-figures with no message
- Use short bullets
- Keep math to what the audience needs for the claim
- Reserve dense derivations for backup slides
