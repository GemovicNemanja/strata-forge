"""Immutable local dataset operation values; no source or platform authority."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

from strata_forge.core.errors import ForgeError
from strata_forge.datasets import _dataset_wire as wire
from strata_forge.datasets import _sql_wire as sql
from strata_forge.datasets.sql_types import LocalParquetFiles


class DatasetExecutionError(ForgeError):
    """Fixed refusal without source bytes, SQL, paths or native exception chains."""

    def __init__(self, code: object = "unavailable") -> None:
        self.code = code if type(code) is str and code in sql.CODES else "unavailable"
        super().__init__("dataset_execution_refused")


def _checked(kind: str, value: object) -> dict[str, sql.JSON]:
    if kind == "convert":
        return wire.convert_spec(value)
    if kind == "query":
        return wire.query_spec(value)
    if kind == "input":
        return wire.input_manifest(value)
    if kind == "installation":
        return wire.installation(value)
    raise sql.WireError("invalid_spec")


class _Value(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)
    canonical: bytes = Field(repr=False)

    def __init__(self, **data: object) -> None:
        failed = False
        try:
            super().__init__(**data)
        except Exception:
            failed = True
        if failed:
            raise DatasetExecutionError("invalid_spec")

    @field_validator("canonical")
    @classmethod
    def _canonical(cls, value: bytes) -> bytes:
        checked = _checked(_kind(cls), sql.parse(value, limit=sql.MAX_MANIFEST))
        wire.need(sql.canonical(checked, limit=sql.MAX_MANIFEST) == value)
        return value

    @classmethod
    def from_wire(cls, value: object) -> _Value:
        checked: _Value | None = None
        try:
            checked = cls(
                canonical=sql.canonical(_checked(_kind(cls), value), limit=sql.MAX_MANIFEST)
            )
        except Exception:
            checked = None
        if checked is None:
            raise DatasetExecutionError("invalid_spec")
        return checked

    def wire_payload(self) -> dict[str, sql.JSON]:
        result: dict[str, sql.JSON] | None = None
        try:
            checked = _checked(_kind(type(self)), sql.parse(self.canonical, limit=sql.MAX_MANIFEST))
            wire.need(sql.canonical(checked, limit=sql.MAX_MANIFEST) == self.canonical)
            result = checked
        except Exception:
            result = None
        if result is None:
            raise DatasetExecutionError("invalid_spec")
        return result

    def digest(self) -> str:
        return sql.digest(_Value.wire_payload(self))


class DatasetConvertSpec(_Value):
    """Explicit scalar conversion or schema-inspected provided Parquet."""

    @classmethod
    def from_wire(cls, value: object) -> DatasetConvertSpec:
        return cast("DatasetConvertSpec", super().from_wire(value))


class DatasetQuerySpec(_Value):
    """One exact SQL query against a complete ordered named input binding."""

    @classmethod
    def from_wire(cls, value: object) -> DatasetQuerySpec:
        return cast("DatasetQuerySpec", super().from_wire(value))


class DatasetInputManifest(_Value):
    """Exact supplied raw bytes; no upstream population or access claim."""

    @classmethod
    def from_wire(cls, value: object) -> DatasetInputManifest:
        return cast("DatasetInputManifest", super().from_wire(value))


class DatasetInstallation(_Value):
    """Explicit trusted operator installation; does not start or inspect Docker."""

    @classmethod
    def from_wire(cls, value: object) -> DatasetInstallation:
        return cast("DatasetInstallation", super().from_wire(value))


class _Files(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)

    def __init__(self, **data: object) -> None:
        failed = False
        try:
            super().__init__(**data)
            if type(self) is ProvidedDatasetFiles:
                ProvidedDatasetFiles.checked(self)
            elif type(self) is NamedParquetFiles:
                NamedParquetFiles.checked(self)
            else:
                raise DatasetExecutionError("invalid_spec")
        except Exception:
            failed = True
        if failed:
            raise DatasetExecutionError("invalid_spec")


class ProvidedDatasetFiles(_Files):
    """Already-held supplied byte descriptors, not caller paths or URLs."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)
    manifest: DatasetInputManifest = Field(repr=False)
    fds: tuple[int, ...] = Field(repr=False)

    def checked(self) -> tuple[dict[str, sql.JSON], tuple[int, ...]]:
        result: tuple[dict[str, sql.JSON], tuple[int, ...]] | None = None
        try:
            wire.need(
                type(self) is ProvidedDatasetFiles and type(self.manifest) is DatasetInputManifest
            )
            manifest = DatasetInputManifest.wire_payload(self.manifest)
            wire.need(
                type(self.fds) is tuple
                and len(self.fds) == len(cast("list[sql.JSON]", manifest["parts"]))
            )
            wire.need(all(type(fd) is int and 0 <= fd <= 1000000 for fd in self.fds))
            result = manifest, self.fds
        except Exception:
            result = None
        if result is None:
            raise DatasetExecutionError("invalid_spec")
        return result


class NamedParquetFiles(_Files):
    """An exact table alias paired with already-held pinned Parquet descriptors."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)
    alias: str
    source: LocalParquetFiles = Field(repr=False)

    def checked(self) -> tuple[dict[str, sql.JSON], tuple[int, ...]]:
        result: tuple[dict[str, sql.JSON], tuple[int, ...]] | None = None
        try:
            wire.need(type(self) is NamedParquetFiles and type(self.source) is LocalParquetFiles)
            name = wire.alias(self.alias)
            manifest, fds = LocalParquetFiles.checked(self.source)
            result = {"alias": name, "manifest": manifest.wire_payload()}, fds
        except Exception:
            result = None
        if result is None:
            raise DatasetExecutionError("invalid_spec")
        return result


def _kind(cls: object) -> str:
    if cls is DatasetConvertSpec:
        return "convert"
    if cls is DatasetQuerySpec:
        return "query"
    if cls is DatasetInputManifest:
        return "input"
    if cls is DatasetInstallation:
        return "installation"
    raise sql.WireError("invalid_spec")
