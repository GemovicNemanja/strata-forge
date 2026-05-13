"""Shared type aliases used across `forge.core` and downstream modules.

These aliases exist to keep call signatures readable and to give a single place
to evolve a foundational type. Add to this module only when the alias is used
by at least two modules — module-local types stay local.
"""

from __future__ import annotations

from os import PathLike as _OsPathLike

__all__ = ["JSONValue", "PathLike"]


type JSONValue = None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
"""A JSON-encodable value — the standard recursive definition."""


type PathLike = str | _OsPathLike[str]
"""Anything ``pathlib.Path(...)`` accepts as a path-like argument."""
