"""Local file read tool — sandboxed to caller-supplied allowed directories.

``fs_read_tool`` is a **factory**: it takes an ``allowed_dirs``
list and returns a configured :class:`Tool`. The Tool's bound function
resolves the requested path and refuses any read outside the allow
list — including symlink-escape attempts, because the comparison is
against the resolved (symlink-followed) target.

Typical use::

    from pathlib import Path
    from strata_forge.agents import Agent
    from strata_forge.agents.tools import fs_read_tool

    docs_reader = fs_read_tool(allowed_dirs=[Path("docs/").resolve()])
    agent = Agent("doc-bot", client=client, tools=[docs_reader])
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from strata_forge.llm.tools import Tool

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "FSReadArgs",
    "fs_read_tool",
]


class FSReadArgs(BaseModel):
    """Arguments for the file-read tool."""

    path: str = Field(
        description="Path to the file to read. Must be inside one of the allowed directories.",
        min_length=1,
    )
    max_bytes: int = Field(
        default=100_000,
        ge=1,
        le=10_000_000,
        description="Cap on the returned content length in characters. Default 100 KB.",
    )


def _is_under(path: Path, directory: Path) -> bool:
    """True when ``path`` is inside ``directory`` (or equal to it)."""
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def fs_read_tool(
    *,
    allowed_dirs: Sequence[Path | str],
    name: str = "fs_read",
    description: str | None = None,
) -> Tool:
    """Build a file-reading :class:`Tool` sandboxed to ``allowed_dirs``.

    The resolved (symlink-followed) target path must live inside one
    of the resolved ``allowed_dirs`` — symlink escape attempts fail
    the check.

    Args:
        allowed_dirs: Directories the tool is permitted to read from.
            At least one is required. Each entry is resolved at tool
            construction time.
        name: Tool name surfaced to the LLM.
        description: Tool description surfaced to the LLM. When
            ``None``, includes the allowed directory list so the
            model knows where it can look.

    Raises:
        ValueError: When ``allowed_dirs`` is empty.
    """
    resolved_allowed = [Path(d).resolve() for d in allowed_dirs]
    if not resolved_allowed:
        err = "fs_read_tool requires at least one allowed directory"
        raise ValueError(err)

    if description is None:
        listing = ", ".join(str(d) for d in resolved_allowed)
        description = (
            "Read a file from disk and return its contents (truncated). "
            f"Allowed root directories: {listing}."
        )

    async def _read(args: FSReadArgs) -> str:
        # Path operations are CPU-bound and quick; we use sync pathlib
        # here rather than anyio.Path to keep the tool dependency-free.
        # Wrapped in asyncio.to_thread to avoid blocking the loop when
        # the file is large.
        import asyncio

        def _do_read() -> str:
            target = Path(args.path).resolve()
            if not any(_is_under(target, d) for d in resolved_allowed):
                err = f"path {target!s} is outside the allowed directories"
                raise PermissionError(err)
            if not target.is_file():
                err = f"file not found: {target!s}"
                raise FileNotFoundError(err)
            return target.read_text(encoding="utf-8", errors="replace")[: args.max_bytes]

        return await asyncio.to_thread(_do_read)

    return Tool(
        name=name,
        description=description,
        parameters_model=FSReadArgs,
        fn=_read,  # type: ignore[arg-type]
    )
