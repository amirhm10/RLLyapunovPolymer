#!/usr/bin/env python3
"""Quick checks for common Beamer slide issues."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PLACEHOLDER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bTODO\b",
        r"\bTBD\b",
        r"\bFIXME\b",
        r"\bWIP\b",
        r"\bplaceholder\b",
        r"\blorem ipsum\b",
        r"\bxxx\b",
    )
]
IMAGE_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".svg")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_graphicspaths(tex: str) -> list[str]:
    match = re.search(r"\\graphicspath\{((?:\{[^{}]+\})+)\}", tex)
    if not match:
        return []
    return re.findall(r"\{([^{}]+)\}", match.group(1))


def resolve_graphic(deck_path: Path, graphicspaths: list[str], raw_path: str) -> Path | None:
    given = Path(raw_path)
    search_roots = [deck_path.parent]
    for item in graphicspaths:
        search_roots.append((deck_path.parent / item).resolve())

    def candidates(base_path: Path) -> list[Path]:
        if given.suffix:
            return [base_path / given]
        return [(base_path / raw_path).with_suffix(ext) for ext in IMAGE_EXTENSIONS]

    for root in search_roots:
        for candidate in candidates(root):
            if candidate.exists():
                return candidate
    return None


def strip_latex(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\$.*?\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\[a-zA-Z@]+(\*?)", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def frame_blocks(tex: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    pattern = re.compile(
        r"\\begin\{frame\}(?:\[[^\]]*\])?(?:\{[^}]*\})?(.*?)\\end\{frame\}",
        re.DOTALL,
    )
    for match in pattern.finditer(tex):
        start_line = tex[: match.start()].count("\n") + 1
        blocks.append((start_line, match.group(1)))
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a Beamer deck for common issues.")
    parser.add_argument("deck", help="Path to the .tex Beamer file")
    parser.add_argument("--word-limit", type=int, default=120, help="Warn above this many words per frame")
    parser.add_argument("--line-limit", type=int, default=140, help="Warn above this many characters per line")
    parser.add_argument("--strict", action="store_true", help="Exit with code 1 when warnings are found")
    args = parser.parse_args()

    deck_path = Path(args.deck).resolve()
    if not deck_path.exists():
        print(f"[error] deck not found: {deck_path}")
        return 1

    tex = read_text(deck_path)
    graphicspaths = parse_graphicspaths(tex)

    warnings: list[str] = []

    for line_number, line in enumerate(tex.splitlines(), start=1):
        if len(line) > args.line_limit and not line.lstrip().startswith("%"):
            warnings.append(f"line {line_number}: long line ({len(line)} chars)")
        for pattern in PLACEHOLDER_PATTERNS:
            if pattern.search(line):
                warnings.append(f"line {line_number}: placeholder text matched `{pattern.pattern}`")
                break

    for line_number, raw_path in [
        (idx + 1, match.group(1))
        for idx, line in enumerate(tex.splitlines())
        for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", line)
    ]:
        if "#" in raw_path:
            continue
        if resolve_graphic(deck_path, graphicspaths, raw_path) is None:
            warnings.append(f"line {line_number}: missing image `{raw_path}`")

    for start_line, block in frame_blocks(tex):
        cleaned = strip_latex(block)
        word_count = len(re.findall(r"\b[\w-]+\b", cleaned))
        if word_count > args.word_limit:
            warnings.append(f"frame at line {start_line}: heavy text ({word_count} words)")

    if warnings:
        print(f"Audit warnings for {deck_path}:")
        for item in warnings:
            print(f"- {item}")
        print(f"\nTotal warnings: {len(warnings)}")
        return 1 if args.strict else 0

    print(f"No warnings found for {deck_path}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
