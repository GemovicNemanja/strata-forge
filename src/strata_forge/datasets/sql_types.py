"""Immutable closed values for an unbound local SQL operation, not source authority."""

from __future__ import annotations

import hashlib
from typing import ClassVar, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from strata_forge.core.errors import ForgeError
from strata_forge.datasets import _sql_wire as wire

__all__ = [
    "LocalParquetFiles",
    "LocalSQLInstallation",
    "SQLExecutionError",
    "SQLInputManifest",
    "SQLQuerySpec",
]


class SQLExecutionError(ForgeError):
    """Fixed body-free failure without SQL, source contents or native diagnostics."""

    def __init__(self, code: object = "unavailable") -> None:
        self.code = code if type(code) is str and code in wire.CODES else "unavailable"
        super().__init__("sql_execution_refused")


def _profile(cls: type[object]) -> tuple[str, int]:
    if cls is SQLQuerySpec:
        return "query", wire.MAX_REQUEST
    if cls is SQLInputManifest:
        return "manifest", wire.MAX_MANIFEST
    if cls is LocalSQLInstallation:
        return "installation", 16384
    raise SQLExecutionError("invalid_spec")


def _payload(kind: str, value: object) -> dict[str, wire.JSON]:
    result: dict[str, wire.JSON] | None = None
    code = "invalid_spec"
    try:
        if kind == "query":
            result = wire.query(value)
        elif kind == "manifest":
            result = wire.manifest(value)
        elif kind == "installation":
            result = wire.installation(value)
    except wire.WireError as refusal:
        code = refusal.code
    if result is None:
        raise SQLExecutionError(code)
    return result


class _Value(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)
    canonical: bytes = Field(repr=False)
    KIND: ClassVar[Literal["query", "manifest", "installation"]]
    LIMIT: ClassVar[int]

    def __init__(self, **data: object) -> None:
        failed = False
        try:
            super().__init__(**data)
        except Exception:
            failed = True
        if failed:
            raise SQLExecutionError("invalid_spec")

    @field_validator("canonical")
    @classmethod
    def _canonical(cls, value: bytes) -> bytes:
        # A complete shared wire validator is used in both the parent and image;
        # no Pydantic-only acceptance or mutable prepared maps escape this boundary.
        kind, limit = _profile(cls)
        decoded = wire.parse(value, limit=limit)
        checked = _payload(kind, decoded)
        if wire.canonical(checked, limit=limit) != value:
            raise SQLExecutionError("invalid_spec")
        return value

    @classmethod
    def from_wire(cls, value: object) -> _Value:
        kind, limit = _profile(cls)
        checked = _payload(kind, value)
        return cls(canonical=wire.canonical(checked, limit=limit))

    def wire_payload(self) -> dict[str, wire.JSON]:
        kind, limit = _profile(type(self))
        result: dict[str, wire.JSON] | None = None
        try:
            raw: object = self.canonical
            decoded = wire.parse(raw, limit=limit)
            result = _payload(kind, decoded)
            if wire.canonical(result, limit=limit) != raw:
                result = None
        except AttributeError, wire.WireError, SQLExecutionError:
            result = None
        if result is None:
            raise SQLExecutionError("invalid_spec")
        return result

    def digest(self) -> str:
        _, limit = _profile(type(self))
        checked = _Value.wire_payload(self)
        return hashlib.sha256(wire.canonical(checked, limit=limit)).hexdigest()


class SQLQuerySpec(_Value):
    """Exact SQL and resource request; no engine, source path, callback or grant."""

    KIND = "query"
    LIMIT = wire.MAX_REQUEST

    @classmethod
    def from_wire(cls, value: object) -> SQLQuerySpec:
        return cast("SQLQuerySpec", super().from_wire(value))

    @property
    def sql(self) -> str:
        return cast("str", _Value.wire_payload(self)["sql"])

    @property
    def timeout_ms(self) -> int:
        return cast("int", _Value.wire_payload(self)["timeout_ms"])

    @property
    def max_result_rows(self) -> int:
        return cast("int", _Value.wire_payload(self)["max_result_rows"])


class SQLInputManifest(_Value):
    """Complete provided byte/schema population and descriptive, unverified origin."""

    KIND = "manifest"
    LIMIT = wire.MAX_MANIFEST

    @classmethod
    def from_wire(cls, value: object) -> SQLInputManifest:
        return cast("SQLInputManifest", super().from_wire(value))


class LocalSQLInstallation(_Value):
    """Trusted operator description only; construction does not inspect or start Docker."""

    KIND = "installation"
    LIMIT = 16384

    @classmethod
    def from_wire(cls, value: object) -> LocalSQLInstallation:
        return cast("LocalSQLInstallation", super().from_wire(value))


class LocalParquetFiles(BaseModel):
    """Local-only caller file descriptions; query wire accepts no descriptors or grants."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)
    manifest: SQLInputManifest = Field(repr=False)
    fds: tuple[int, ...] = Field(repr=False)

    def __init__(self, **data: object) -> None:
        failed = False
        try:
            super().__init__(**data)
            self.checked()
        except Exception:
            failed = True
        if failed:
            raise SQLExecutionError("invalid_spec")

    @classmethod
    def from_open_fds(
        cls, *, manifest: SQLInputManifest, fds: tuple[int, ...]
    ) -> LocalParquetFiles:
        return cls(manifest=manifest, fds=fds)

    def checked(self) -> tuple[SQLInputManifest, tuple[int, ...]]:
        failed = False
        try:
            valid = type(self) is LocalParquetFiles and type(self.manifest) is SQLInputManifest
            if valid:
                parts = _Value.wire_payload(self.manifest)["parts"]
                valid = type(self.fds) is tuple and len(self.fds) == len(
                    cast("list[wire.JSON]", parts)
                )
                valid = valid and all(type(fd) is int and 0 <= fd <= 1_000_000 for fd in self.fds)
        except AttributeError, SQLExecutionError:
            failed = True
            valid = False
        if failed or not valid:
            raise SQLExecutionError("invalid_spec")
        return self.manifest, self.fds
