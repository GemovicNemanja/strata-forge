"""Unit tests for `strata_forge.agents.tools.fs_read`."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from strata_forge.agents.tools.fs_read import FSReadArgs, fs_read_tool

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """An empty directory the tests are allowed to read from."""
    return tmp_path


# ---------------------------------------------------------------------------
# Factory validation
# ---------------------------------------------------------------------------


class TestFactory:
    def test_empty_allowed_dirs_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            fs_read_tool(allowed_dirs=[])

    def test_returns_tool_instance(self, sandbox: Path) -> None:
        tool = fs_read_tool(allowed_dirs=[sandbox])
        assert tool.name == "fs_read"
        assert tool.parameters_model is FSReadArgs

    def test_custom_name(self, sandbox: Path) -> None:
        tool = fs_read_tool(allowed_dirs=[sandbox], name="read_docs")
        assert tool.name == "read_docs"

    def test_description_lists_allowed_dirs_by_default(self, sandbox: Path) -> None:
        tool = fs_read_tool(allowed_dirs=[sandbox])
        assert str(sandbox) in tool.description

    def test_custom_description(self, sandbox: Path) -> None:
        tool = fs_read_tool(allowed_dirs=[sandbox], description="custom desc")
        assert tool.description == "custom desc"

    def test_string_paths_accepted(self, sandbox: Path) -> None:
        # Callers can pass str or Path interchangeably.
        tool = fs_read_tool(allowed_dirs=[str(sandbox)])
        assert tool.name == "fs_read"


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestReadHappyPath:
    async def test_reads_file_in_allowed_dir(self, sandbox: Path) -> None:
        target = sandbox / "note.txt"
        target.write_text("hello world")
        tool = fs_read_tool(allowed_dirs=[sandbox])
        result = await tool.invoke({"path": str(target)})
        assert result == "hello world"

    async def test_reads_file_in_subdir(self, sandbox: Path) -> None:
        sub = sandbox / "sub"
        sub.mkdir()
        target = sub / "deep.txt"
        target.write_text("nested")
        tool = fs_read_tool(allowed_dirs=[sandbox])
        result = await tool.invoke({"path": str(target)})
        assert result == "nested"

    async def test_truncates_to_max_bytes(self, sandbox: Path) -> None:
        target = sandbox / "big.txt"
        target.write_text("x" * 1000)
        tool = fs_read_tool(allowed_dirs=[sandbox])
        result = await tool.invoke({"path": str(target), "max_bytes": 100})
        assert result == "x" * 100

    async def test_multiple_allowed_dirs(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "f1.txt").write_text("from a")
        (b / "f2.txt").write_text("from b")
        tool = fs_read_tool(allowed_dirs=[a, b])
        assert await tool.invoke({"path": str(a / "f1.txt")}) == "from a"
        assert await tool.invoke({"path": str(b / "f2.txt")}) == "from b"


# ---------------------------------------------------------------------------
# Sandbox enforcement
# ---------------------------------------------------------------------------


class TestSandbox:
    async def test_path_outside_allowed_rejected(self, sandbox: Path, tmp_path: Path) -> None:
        # Create a file in a sibling directory that's NOT allowed.
        sibling = tmp_path.parent / f"sibling-{os.getpid()}"
        sibling.mkdir(exist_ok=True)
        forbidden = sibling / "secret.txt"
        forbidden.write_text("shhh")
        try:
            tool = fs_read_tool(allowed_dirs=[sandbox])
            with pytest.raises(PermissionError, match="outside"):
                await tool.invoke({"path": str(forbidden)})
        finally:
            forbidden.unlink(missing_ok=True)
            sibling.rmdir()

    async def test_parent_traversal_rejected(self, sandbox: Path) -> None:
        # ../outside-the-sandbox shouldn't bypass the check after resolving.
        outside = sandbox.parent / "escaped.txt"
        outside.write_text("nope")
        try:
            tool = fs_read_tool(allowed_dirs=[sandbox])
            with pytest.raises(PermissionError, match="outside"):
                await tool.invoke({"path": str(sandbox / ".." / "escaped.txt")})
        finally:
            outside.unlink(missing_ok=True)

    async def test_symlink_escape_rejected(self, sandbox: Path) -> None:
        # A symlink inside the sandbox pointing outside resolves to outside,
        # which the check catches.
        outside = sandbox.parent / "outside_target.txt"
        outside.write_text("secret")
        try:
            link = sandbox / "escape_link"
            link.symlink_to(outside)
            tool = fs_read_tool(allowed_dirs=[sandbox])
            with pytest.raises(PermissionError, match="outside"):
                await tool.invoke({"path": str(link)})
        finally:
            outside.unlink(missing_ok=True)

    async def test_missing_file_raises(self, sandbox: Path) -> None:
        tool = fs_read_tool(allowed_dirs=[sandbox])
        with pytest.raises(FileNotFoundError):
            await tool.invoke({"path": str(sandbox / "nonexistent.txt")})

    async def test_directory_path_raises(self, sandbox: Path) -> None:
        sub = sandbox / "subdir"
        sub.mkdir()
        tool = fs_read_tool(allowed_dirs=[sandbox])
        with pytest.raises(FileNotFoundError):
            await tool.invoke({"path": str(sub)})
