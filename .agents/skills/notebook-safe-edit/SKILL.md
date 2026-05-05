---
name: notebook-safe-edit
description: Use for editing Jupyter notebooks, debugging notebook JSON or execution issues, preserving cell structure and metadata, handling outputs and trust safely, using notebook-aware version-control workflows, and validating notebook integrity after changes.
---

# Notebook Safe Edit

Use this skill when the task involves `.ipynb` files.

## Rules

- Treat `.ipynb` as structured JSON, not plain prose.
- Preserve valid notebook JSON and schema at all times.
- Prefer `nbformat`-compatible or other notebook-aware tools over ad hoc text rewrites.
- Preserve metadata unless the task requires changing it.
- If adding custom metadata, use a unique namespace rather than generic top-level keys when practical.
- Avoid introducing invalid notebook schema.
- If the notebook has execution, format, or JSON issues, diagnose the root cause first.
- If a notebook is damaged, validate and repair structure before changing logic.
- Prefer minimal edits.
- Preserve cell order unless reordering is necessary and explained.
- Preserve cell structure unless the task requires changing it.
- Keep outputs cleared unless the task explicitly asks to save outputs.
- Do not delete markdown explanation cells unless they are clearly obsolete.
- When changing analysis logic, also mirror important logic in a `.py` helper when practical.
- Prefer reproducible scripts over hidden notebook state.
- For repeated or collaborative work, prefer notebook-aware tooling over raw git line diffs.
- Prefer text-paired or script-backed workflows when they make version control and review safer.
- Treat notebook trust deliberately: HTML, JavaScript, and widget-heavy outputs may require re-execution or explicit trust before they render fully.
- If Python dependencies are missing, install them in the project environment and document them.

## Preferred Tooling

- Use `nbformat` or another notebook-aware editor to read, validate, and write notebooks.
- Prefer `Jupytext` pairing for long-lived notebooks under version control.
  Use `py:percent` for code-heavy notebooks and Markdown-based text notebooks for documentation-heavy ones.
- Prefer `nbdime` for notebook diffs and merges instead of plain line-based diff tools.
- Prefer `nbclient` or `jupyter execute` for lightweight, reproducible execution checks.
- Prefer `papermill` when the notebook should run with explicit parameters instead of hidden state.

## Workflow

1. Inspect the notebook structure, metadata, kernel information, outputs, and failure mode before editing.
2. Validate the notebook JSON.
3. Repair structural issues first if validation fails.
4. Make the smallest notebook-safe change that satisfies the request.
5. If logic is important or reusable, move or mirror it into a Python module.
6. For notebooks that should be reviewable or reproducible over time, consider pairing them with Jupytext or backing key logic with scripts.
7. Clear outputs unless saved outputs are explicitly requested.
8. Before finishing, validate that the notebook can open and that key cells can execute.
9. If execution fails, preserve the failing traceback in a saved notebook or clearly report the blocking cell.
10. Report any remaining risks if validation could not be completed fully.

## Validation

- Check notebook JSON validity.
- Check required `kernelspec` and `language_info` fields if present.
- Validate notebook structure after edits.
- Prefer lightweight execution checks on safe key cells instead of full notebook reruns unless the user asks for a full run.
- For parameterized workflows, prefer a reproducible execution path such as `papermill` rather than manual reruns.
- If rich outputs matter, note whether the notebook may still need `jupyter trust` or re-execution for trusted rendering.
- For shared notebooks under version control, prefer notebook-aware diffs and paired text representations when appropriate.
- Report any remaining structural, dependency, or execution risks.
