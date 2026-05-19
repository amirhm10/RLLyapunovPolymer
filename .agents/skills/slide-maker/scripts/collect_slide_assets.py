#!/usr/bin/env python3
"""Collect likely slide assets from a repository."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


EXT_GROUPS = {
    "tex": {".tex"},
    "bib": {".bib"},
    "figures": {".png", ".jpg", ".jpeg", ".svg", ".pdf"},
    "data": {".csv", ".npy", ".npz", ".pkl", ".pickle", ".mat"},
    "notebooks": {".ipynb"},
}
SKIP_DIRS = {".agents", ".git", "__pycache__", ".venv", "node_modules", "_texstubs", "_tmpbuild"}


def category_for(path: Path) -> str | None:
    suffix = path.suffix.lower()
    for name, extensions in EXT_GROUPS.items():
        if suffix in extensions:
            return name
    return None


def score_path(path: Path, query_terms: list[str]) -> int:
    text = str(path).lower()
    score = 0
    for token in query_terms:
        if token in text:
            score += 5
    for hot_word in ("slide", "poster", "figure", "macc2026", "research_summary", "main.tex"):
        if hot_word in text:
            score += 2
    if "figures" in text:
        score += 1
    return score


def iter_files(root: Path):
    for current_root, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            path = Path(current_root) / filename
            category = category_for(path)
            if category:
                yield category, path


def main() -> int:
    parser = argparse.ArgumentParser(description="List likely assets for a slide task.")
    parser.add_argument("query", nargs="*", help="Optional search words such as c2, lyapunov, poster")
    parser.add_argument("--root", default=".", help="Repository root to scan")
    parser.add_argument("--max-per-group", type=int, default=12, help="Maximum entries to print per category")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    query_terms = [item.lower() for item in args.query]

    grouped: dict[str, list[tuple[int, Path]]] = {name: [] for name in EXT_GROUPS}
    for category, path in iter_files(root):
        grouped[category].append((score_path(path.relative_to(root), query_terms), path.relative_to(root)))

    print(f"Slide asset scan under {root}")
    if query_terms:
        print(f"Query terms: {', '.join(query_terms)}")
    print()

    for category in ("tex", "bib", "figures", "data", "notebooks"):
        ranked = sorted(grouped[category], key=lambda item: (-item[0], str(item[1]).lower()))
        if not ranked:
            continue
        print(f"[{category}]")
        for score, path in ranked[: args.max_per_group]:
            print(f"- score={score:02d}  {path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
