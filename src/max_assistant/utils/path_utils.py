# Copyright (c) 2026, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
Filesystem and project path resolution utilities.
"""
from functools import lru_cache
from pathlib import Path
from typing import Tuple


@lru_cache(maxsize=1)
def get_project_root(
        anchor_file: str = __file__,
        markers: Tuple[str, ...] = ("pyproject.toml", "src")
) -> Path:
    """
    Resolves and caches the project root directory by climbing up from `anchor_file`
    until one of the `markers` is found.
    """
    current = Path(anchor_file).resolve()
    start_dir = current if current.is_dir() else current.parent

    for parent in [start_dir, *start_dir.parents]:
        if any((parent / marker).exists() for marker in markers):
            return parent

    if Path("/app").exists():
        return Path("/app")

    return start_dir


def resolve_project_path(relative_or_abs_path: str | Path) -> Path:
    """
    Resolves a path relative to the project root, or returns it directly if already absolute.
    """
    path = Path(relative_or_abs_path)
    if path.is_absolute():
        return path
    return get_project_root() / path