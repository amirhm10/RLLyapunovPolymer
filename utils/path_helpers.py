from __future__ import annotations

from pathlib import Path
from typing import Optional, Union


PathLike = Union[str, Path]

_REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_root() -> Path:
    return _REPO_ROOT


def repo_path(*parts: PathLike) -> Path:
    path = _REPO_ROOT
    for part in parts:
        path = path / Path(part)
    return path


def resolve_repo_path(path: Optional[PathLike] = None, *default_parts: PathLike, create: bool = False) -> Path:
    if path is None:
        resolved = repo_path(*default_parts)
    else:
        candidate = Path(path).expanduser()
        resolved = candidate if candidate.is_absolute() else repo_path(candidate)

    if create:
        resolved.mkdir(parents=True, exist_ok=True)

    return resolved
