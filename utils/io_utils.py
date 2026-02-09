"""I/O helpers for loading and validating artifacts."""

from pathlib import Path
from typing import Iterable


def ensure_dirs(paths: Iterable[Path]) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def exists_all(paths: Iterable[Path]) -> bool:
    """Return True only if all provided paths exist."""
    return all(path.exists() for path in paths)
