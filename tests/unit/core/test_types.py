"""Unit tests for `strata_forge.core.types`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata_forge.core.types import JSONValue, PathLike


class TestAliasesUsable:
    """The aliases must be importable and usable in function signatures.

    These tests don't assert much beyond "this compiles" — type aliases mainly
    exist to make signatures readable; the real check is pyright strict.
    """

    def test_jsonvalue_accepts_primitive_kinds(self) -> None:
        def echo(x: JSONValue) -> JSONValue:
            return x

        assert echo(None) is None
        assert echo(True) is True
        assert echo(1) == 1
        assert echo(1.5) == 1.5
        assert echo("text") == "text"

    def test_jsonvalue_accepts_recursive_containers(self) -> None:
        def echo(x: JSONValue) -> JSONValue:
            return x

        payload: JSONValue = {
            "list": [1, 2, "three", None],
            "nested": {"deeper": [True, False]},
        }
        assert echo(payload) == payload

    def test_pathlike_accepts_str_and_path(self) -> None:
        def to_path(p: PathLike) -> Path:
            return Path(p)

        assert to_path("examples").name == "examples"
        assert to_path(Path("examples")).name == "examples"
