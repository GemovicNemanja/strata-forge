"""Bounded pinned native-Parquet reads after a trusted evaluation access decision."""

from __future__ import annotations

import asyncio
import fnmatch
import itertools
import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, Protocol, cast

from strata_forge.pipelines._evaluation_access import EvaluationError
from strata_forge.pipelines._hub_ids import validate_hub_ref
from strata_forge.storage import HFHubClient

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from strata_forge.pipelines.evaluation_spec import Selection

type Row = dict[str, object]


class _ArrowSchema(Protocol):
    names: list[str]


class _ArrowBatch(Protocol):
    def to_pylist(self) -> list[Row]: ...


class _ParquetFile(Protocol):
    schema_arrow: _ArrowSchema

    def iter_batches(self, *, batch_size: int, columns: list[str]) -> Iterator[_ArrowBatch]: ...


class EvaluationReader(Protocol):
    """Read already-authorized pins without following implicit external dependencies."""

    async def prepare(self, ref: str, *, model: bool = False) -> None: ...

    async def rows(
        self,
        ref: str,
        config: str,
        split: str,
        selection: Selection | None,
        columns: tuple[str, ...],
    ) -> tuple[Row, ...]: ...
    async def json(
        self, ref: str, path: str, *, repo_type: Literal["model", "dataset"] = "dataset"
    ) -> dict[str, object]: ...


def _read_text(path: Path, maximum: int) -> str:
    if path.stat().st_size > maximum:
        raise EvaluationError("input_metadata_too_large")
    return path.read_text(encoding="utf-8")


def _safe_path(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value or ":" in value:
        raise EvaluationError("dataset_dependency_unsupported")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("."):
        raise EvaluationError("dataset_dependency_unsupported")
    return value


def _files(card: str, inventory: Sequence[str], config: str, split: str) -> tuple[str, ...]:
    import yaml

    if not card.startswith("---\n") or "\n---" not in card[4:]:
        raise EvaluationError("dataset_card_configs_required")
    try:
        metadata: object = yaml.safe_load(card[4:].split("\n---", 1)[0])
    except yaml.YAMLError:
        raise EvaluationError("dataset_card_invalid") from None
    if not isinstance(metadata, dict):
        raise EvaluationError("dataset_card_configs_required")
    info = cast("Row", metadata)
    config_entries = info.get("configs")
    if not isinstance(config_entries, list):
        raise EvaluationError("dataset_card_configs_required")
    configs = [
        cast("Row", c)
        for c in cast("list[object]", config_entries)
        if isinstance(c, dict) and cast("Row", c).get("config_name") == config
    ]
    if len(configs) != 1 or not isinstance(configs[0].get("data_files"), list):
        raise EvaluationError("dataset_config_invalid")
    patterns: list[str] = []
    for part in cast("list[object]", configs[0]["data_files"]):
        if isinstance(part, dict) and cast("Row", part).get("split") == split:
            value = cast("Row", part).get("path")
            for path in cast("list[object]", value) if isinstance(value, list) else [value]:
                patterns.append(_safe_path(path))
    selected = tuple(
        sorted(
            {
                name
                for pattern in patterns
                for name in inventory
                if fnmatch.fnmatchcase(name, pattern)
            }
        )
    )
    if (
        not selected
        or len(selected) > 32
        or any(not name.endswith(".parquet") or _safe_path(name) != name for name in selected)
    ):
        raise EvaluationError("dataset_files_unsupported")
    return selected


def _parquet_rows(
    paths: tuple[Path, ...], selection: Selection | None, columns: tuple[str, ...]
) -> tuple[Row, ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("install 'strata-forge[hf]' for evaluation Parquet reads") from exc
    limit = selection.row_limit if selection is not None else 1_000_001
    start = selection.row_start if selection is not None else 0
    end = selection.row_end if selection is not None else None

    def source() -> Iterator[Row]:
        for path in paths:
            if path.stat().st_size > 256 * 1024 * 1024:
                raise EvaluationError("dataset_file_too_large")
            file = cast("_ParquetFile", pq.ParquetFile(path))
            if not set(columns) <= set(file.schema_arrow.names):
                raise EvaluationError("dataset_column_missing")
            for batch in file.iter_batches(batch_size=1024, columns=list(columns)):
                yield from batch.to_pylist()

    rows = tuple(itertools.islice(itertools.islice(source(), start, end), limit))
    if selection is None and len(rows) > 1_000_000:
        raise EvaluationError("prediction_limit_exceeded")
    return rows


class HubEvaluationReader:
    """Read declared repository Parquet files at exact commits, never dataset scripts."""

    def __init__(self, token: str | Literal[False]) -> None:
        self.hub = HFHubClient(token=token)

    async def prepare(self, ref: str, *, model: bool = False) -> None:
        validate_hub_ref(ref)
        repo, sha = ref.rsplit("@", 1)
        kind = "model" if model else "dataset"
        actual = await self.hub.resolve_revision(repo, revision=sha, repo_type=kind)
        if actual != sha:
            raise EvaluationError("input_ref_mismatch")
        if model:
            files = await self.hub.list_repo_files(repo, repo_type="model", revision=sha)
            if (
                "adapter_config.json" in files
                or "config.json" not in files
                or not any(name.endswith(".safetensors") for name in files)
            ):
                raise EvaluationError("model_checkpoint_unsupported")

    async def _file(
        self, ref: str, path: str, repo_type: Literal["model", "dataset"] = "dataset"
    ) -> Path:
        repo, sha = ref.rsplit("@", 1)
        return await self.hub.download_file(
            repo, _safe_path(path), repo_type=repo_type, revision=sha
        )

    async def json(
        self, ref: str, path: str, *, repo_type: Literal["model", "dataset"] = "dataset"
    ) -> dict[str, object]:
        file = await self._file(ref, path, repo_type)
        try:
            cap = 65_536 if path == "strata_verifier.json" else 32 * 1024 * 1024
            value: object = json.loads(await asyncio.to_thread(_read_text, file, cap))
        except ValueError, UnicodeError:
            raise EvaluationError("input_metadata_invalid") from None
        if not isinstance(value, dict):
            raise EvaluationError("input_metadata_invalid")
        return cast("dict[str, object]", value)

    async def rows(
        self,
        ref: str,
        config: str,
        split: str,
        selection: Selection | None,
        columns: tuple[str, ...],
    ) -> tuple[Row, ...]:
        repo, sha = ref.rsplit("@", 1)
        card = await self._file(ref, "README.md")
        inventory = await self.hub.list_repo_files(repo, repo_type="dataset", revision=sha)
        paths = _files(await asyncio.to_thread(_read_text, card, 65_536), inventory, config, split)
        files = tuple([await self._file(ref, path) for path in paths])
        return await asyncio.to_thread(_parquet_rows, files, selection, columns)
