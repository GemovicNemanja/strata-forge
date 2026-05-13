"""Helpers for loading ``.env`` files into the process environment.

``python-dotenv`` does the heavy lifting; this module wraps it with a typed,
defensible surface that returns whether a file was actually loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from forge.core.types import PathLike

__all__ = ["load_env_file"]


def load_env_file(path: PathLike | None = None, *, override: bool = False) -> bool:
    """Load environment variables from a ``.env`` file.

    Args:
        path: Path to the ``.env`` file. When ``None``, looks for ``.env`` in
            the current working directory.
        override: When ``True``, existing environment variables are
            overwritten. When ``False`` (the default), already-set values are
            preserved — useful so a developer's shell session always wins
            against the committed-by-accident defaults in ``.env``.

    Returns:
        ``True`` if a file was found and loaded; ``False`` if no file existed
        at the resolved path.
    """
    target = Path(path) if path is not None else Path(".env")
    if not target.exists():
        return False
    return load_dotenv(target, override=override)
