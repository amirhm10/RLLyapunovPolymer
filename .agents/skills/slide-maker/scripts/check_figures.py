#!/usr/bin/env python3
"""Check figure existence, dimensions, and simple naming quirks."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path
from xml.etree import ElementTree


SUPPORTED = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
DOUBLE_EXTENSIONS = (".png.png", ".jpg.jpg", ".jpeg.jpeg", ".svg.svg", ".pdf.pdf")


def collect_paths(items: list[str]) -> list[Path]:
    collected: list[Path] = []
    for item in items:
        path = Path(item)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED:
                    collected.append(child)
        else:
            collected.append(path)
    return collected


def png_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        header = handle.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def jpeg_size(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as handle:
        data = handle.read()
    if not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index < len(data):
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3):
            block = data[index + 5 : index + 9]
            height, width = struct.unpack(">HH", block)
            return width, height
        if index + 4 > len(data):
            return None
        block_len = struct.unpack(">H", data[index + 2 : index + 4])[0]
        index += 2 + block_len
    return None


def svg_size(path: Path) -> tuple[int, int] | None:
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError:
        return None
    width = root.attrib.get("width")
    height = root.attrib.get("height")
    view_box = root.attrib.get("viewBox")
    if width and height:
        try:
            return int(float(width.rstrip("px"))), int(float(height.rstrip("px")))
        except ValueError:
            pass
    if view_box:
        parts = view_box.split()
        if len(parts) == 4:
            try:
                return int(float(parts[2])), int(float(parts[3]))
            except ValueError:
                return None
    return None


def probe_size(path: Path) -> tuple[int, int] | None:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return png_size(path)
    if suffix in {".jpg", ".jpeg"}:
        return jpeg_size(path)
    if suffix == ".svg":
        return svg_size(path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check figures for slide use.")
    parser.add_argument("paths", nargs="+", help="Figure files or directories")
    parser.add_argument("--min-width", type=int, default=800, help="Warn below this raster width")
    parser.add_argument("--min-height", type=int, default=450, help="Warn below this raster height")
    args = parser.parse_args()

    for path in collect_paths(args.paths):
        notes: list[str] = []
        if not path.exists():
            print(f"[missing] {path}")
            continue

        size = probe_size(path)
        if size:
            width, height = size
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                if width < args.min_width or height < args.min_height:
                    notes.append(f"small raster {width}x{height}")
            size_text = f"{width}x{height}"
        else:
            size_text = "size n/a"

        lower_name = path.name.lower()
        for suffix in DOUBLE_EXTENSIONS:
            if lower_name.endswith(suffix):
                notes.append("double extension")
                break

        note_text = ", ".join(notes) if notes else "ok"
        print(f"[check] {path}  {size_text}  {note_text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
