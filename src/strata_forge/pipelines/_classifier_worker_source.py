"""Pinned Hub classifier inputs with bounded metadata, private staging and streamed rows."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import stat
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

# Reuse the installed checkpoint and inert dataset selectors without widening their API.
from strata_forge.pipelines._classifier_checkpoint import (
    _SAFE_NAMES,  # pyright: ignore[reportPrivateUsage]
)
from strata_forge.pipelines._evaluation_reader import _files  # pyright: ignore[reportPrivateUsage]
from strata_forge.pipelines.classifier_runner import (
    ResolvedClassifierCheckpoint,
    TrainingReadAuthority,
    TrainingReadDecision,
    TrainingReadRequest,
)
from strata_forge.pipelines.classifier_spec import (
    MAX_INPUT_BYTES,
    MAX_ROW_BYTES,
    ClassifierDataset,
    ClassifierTrainSpec,
    ClassifierWorkerError,
)
from strata_forge.training._classifier_native import owned_native

# The optional Hub and Arrow APIs are dynamic native-library boundaries; they remain lazy.
type _Native = Any
if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Generator
_MODEL_BYTES = 64 * 1024**3
_MODEL_ENTRIES = 128
_DATASET_ENTRIES = 4096
_DATASET_BYTES = 256 * 1024**2
_CARD_BYTES = 65_536
_PARQUET_BATCH_ROWS = 32
_PARQUET_BATCH_BYTES = 8 * 1024**2
_HUB_ORIGIN = "https://huggingface.co"


@dataclass(frozen=True)
class _Entry:
    size: int
    digest: str
    lfs: bool


async def settled_io[T](function: Callable[[], T]) -> T:
    with suppress(Exception):
        return await owned_native(function)
    raise ClassifierWorkerError("classifier_source_invalid")


def _path(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ClassifierWorkerError("classifier_source_invalid")
    return value


def _regular(path: Path, expected: int, maximum: int) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size != expected or not 0 <= expected <= maximum:
        raise ClassifierWorkerError("classifier_source_invalid")


def _read_card(path: Path, expected: int) -> str:
    _regular(path, expected, _CARD_BYTES)
    with path.open("rb") as stream:
        raw = stream.read(_CARD_BYTES + 1)
    if len(raw) != expected:
        raise ClassifierWorkerError("classifier_source_invalid")
    return raw.decode("utf-8")


def _verified_file(
    source: Path, entry: _Entry, maximum: int, destination: Path | None = None
) -> None:
    _regular(source, entry.size, maximum)
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size != entry.size:
            raise ClassifierWorkerError("classifier_source_invalid")
        digest = hashlib.sha256() if entry.lfs else hashlib.sha1(usedforsecurity=False)
        if not entry.lfs:
            digest.update(f"blob {entry.size}\0".encode("ascii"))
        target = None
        try:
            if destination is not None:
                output = os.open(
                    destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
                )
                target = os.fdopen(output, "wb")
            remaining = entry.size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise ClassifierWorkerError("classifier_source_invalid")
                digest.update(chunk)
                if target is not None:
                    target.write(chunk)
                remaining -= len(chunk)
            after = os.fstat(descriptor)
            if (
                digest.hexdigest() != entry.digest
                or os.read(descriptor, 1)
                or (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            ):
                raise ClassifierWorkerError("classifier_source_invalid")
        finally:
            if target is not None:
                target.close()
    finally:
        os.close(descriptor)


class HubClassifierSource:
    """Use only explicitly delivered Hub credentials and fresh training-purpose decisions."""

    def __init__(
        self,
        spec: ClassifierTrainSpec,
        *,
        authority: TrainingReadAuthority,
        token: str | Literal[False],
        parent: Path,
    ) -> None:
        self.spec = ClassifierTrainSpec.model_validate(spec.model_dump(mode="json"))
        if token is not False and (type(token) is not str or not token.strip()):
            raise ClassifierWorkerError("classifier_source_invalid")
        self.authority, self._token, self.parent = authority, token, parent
        self._temporary: TemporaryDirectory[str] | None = None
        self._closed = False

    async def _check(self, ref: str, kind: Literal["model", "dataset"]) -> None:
        if self._closed:
            raise ClassifierWorkerError("classifier_read_denied")
        request = TrainingReadRequest(
            run_id=self.spec.run_id, spec_sha256=self.spec.digest(), ref=ref, repo_type=kind
        )
        decision = TrainingReadDecision.model_validate(await self.authority.decide(request))
        if (
            decision.authorized is not True
            or decision.custody != "unsealed"
            or any(getattr(decision, key) != value for key, value in request.model_dump().items())
        ):
            raise ClassifierWorkerError("classifier_read_denied")

    def _hub(self) -> _Native:
        try:
            import huggingface_hub
        except ImportError:
            raise ImportError("Install 'strata-forge[hf]' for classifier Hub inputs.") from None
        return huggingface_hub

    def _root(self) -> Path:
        if self._temporary is None:
            if (
                not self.parent.is_absolute()
                or ".." in self.parent.parts
                or any(path.is_symlink() for path in (self.parent, *self.parent.parents))
                or not self.parent.is_dir()
            ):
                raise ClassifierWorkerError("classifier_attempt_invalid")
            self._temporary = TemporaryDirectory(prefix=".classifier-inputs-", dir=self.parent)
        return Path(self._temporary.name)

    def _inventory(self, ref: str, kind: Literal["model", "dataset"]) -> dict[str, _Entry]:
        hub = self._hub()
        repo, revision = ref.rsplit("@", 1)
        api = hub.HfApi(token=self._token, endpoint=_HUB_ORIGIN)
        if api.repo_info(repo_id=repo, repo_type=kind, revision=revision).sha != revision:
            raise ClassifierWorkerError("classifier_ref_mismatch")
        entries = iter(api.list_repo_tree(repo, repo_type=kind, revision=revision, recursive=True))
        result: dict[str, _Entry] = {}
        seen: set[str] = set()
        maximum = _MODEL_ENTRIES if kind == "model" else _DATASET_ENTRIES
        try:
            for count, entry in enumerate(entries, 1):
                if count > maximum or not isinstance(entry, (hub.RepoFile, hub.RepoFolder)):
                    raise ClassifierWorkerError("classifier_input_limit")
                name = _path(entry.path)
                if name in seen:
                    raise ClassifierWorkerError("classifier_source_invalid")
                seen.add(name)
                if isinstance(entry, hub.RepoFile):
                    if type(entry.size) is not int or entry.size < 0:
                        raise ClassifierWorkerError("classifier_source_invalid")
                    is_lfs = entry.lfs is not None
                    digest = entry.lfs.sha256 if is_lfs else entry.blob_id
                    if (
                        type(digest) is not str
                        or re.fullmatch(r"[a-f0-9]{64}" if is_lfs else r"[a-f0-9]{40}", digest)
                        is None
                    ):
                        raise ClassifierWorkerError("classifier_source_invalid")
                    result[name] = _Entry(entry.size, digest, is_lfs)
        finally:
            close = getattr(entries, "close", None)
            if close is not None:
                close()
        return result

    def _download(
        self, ref: str, name: str, kind: Literal["model", "dataset"], entry: _Entry, maximum: int
    ) -> Path:
        hub = self._hub()
        repo, revision = ref.rsplit("@", 1)
        result = hub.hf_hub_download(
            repo,
            _path(name),
            repo_type=kind,
            revision=revision,
            token=self._token,
            endpoint=_HUB_ORIGIN,
            local_dir=self._root() / ("download-" + kind),
        )
        path = Path(result)
        root = (self._root() / ("download-" + kind)).resolve()
        if not path.is_relative_to(root) or any(p.is_symlink() for p in (path, *path.parents)):
            raise ClassifierWorkerError("classifier_source_invalid")
        _regular(path, entry.size, maximum)
        return path

    async def resolve_model(self, ref: str) -> ResolvedClassifierCheckpoint:
        if ref != self.spec.base.ref:
            raise ClassifierWorkerError("classifier_ref_mismatch")
        await self._check(ref, "model")
        inventory = await settled_io(lambda: self._inventory(ref, "model"))
        files = {
            name: entry
            for name, entry in inventory.items()
            if name in _SAFE_NAMES or re.fullmatch(r"[A-Za-z0-9_.-]+\.safetensors", name)
        }
        if (
            "adapter_config.json" in inventory
            or "config.json" not in files
            or not any(name.endswith(".safetensors") for name in files)
            or sum(entry.size for entry in files.values()) > _MODEL_BYTES
            or any(
                entry.size > 32 * 1024**2
                for name, entry in files.items()
                if not name.endswith(".safetensors")
            )
        ):
            raise ClassifierWorkerError("classifier_checkpoint_invalid")
        root = await settled_io(self._root)
        if (await settled_io(lambda: shutil.disk_usage(root))).free < 2 * sum(
            entry.size for entry in files.values()
        ):
            raise ClassifierWorkerError("classifier_input_limit")
        directory = root / "model"
        await settled_io(lambda: directory.mkdir(mode=0o700))
        for name, entry in files.items():
            await self._check(ref, "model")
            source = await settled_io(
                partial(self._download, ref, name, "model", entry, _MODEL_BYTES)
            )
            await self._check(ref, "model")
            await settled_io(partial(_verified_file, source, entry, _MODEL_BYTES, directory / name))
        await self._check(ref, "model")
        return ResolvedClassifierCheckpoint(ref, directory)

    async def rows(self, source: ClassifierDataset) -> AsyncIterator[bytes]:
        source = ClassifierDataset.model_validate(source.model_dump(mode="json"))
        if source != self.spec.dataset:
            raise ClassifierWorkerError("classifier_ref_mismatch")
        await self._check(source.ref, "dataset")
        inventory = await settled_io(lambda: self._inventory(source.ref, "dataset"))
        card_entry = inventory.get("README.md")
        if card_entry is None or card_entry.size > _CARD_BYTES:
            raise ClassifierWorkerError("classifier_source_invalid")
        await self._check(source.ref, "dataset")
        card = await settled_io(
            lambda: self._download(source.ref, "README.md", "dataset", card_entry, _CARD_BYTES)
        )
        await settled_io(lambda: _verified_file(card, card_entry, _CARD_BYTES))
        text = await settled_io(lambda: _read_card(card, card_entry.size))
        await settled_io(lambda: _verified_file(card, card_entry, _CARD_BYTES))
        names = _files(text, tuple(inventory), source.config, source.split)
        if sum(inventory[name].size for name in names) > _DATASET_BYTES:
            raise ClassifierWorkerError("classifier_input_limit")
        used = 0
        for name in names:
            await self._check(source.ref, "dataset")
            path = await settled_io(
                partial(
                    self._download, source.ref, name, "dataset", inventory[name], _DATASET_BYTES
                )
            )
            await settled_io(partial(_verified_file, path, inventory[name], _DATASET_BYTES))
            iterator = self._parquet(path, tuple(source.columns.values()))
            try:
                while True:
                    await self._check(source.ref, "dataset")
                    batch = await settled_io(partial(next, iterator, None))
                    if batch is None:
                        break
                    for raw in batch:
                        if used + len(raw) > MAX_INPUT_BYTES:
                            raise ClassifierWorkerError("classifier_input_limit")
                        used += len(raw)
                        yield raw
            finally:
                await settled_io(iterator.close)
            await settled_io(partial(_verified_file, path, inventory[name], _DATASET_BYTES))
        await self._check(source.ref, "dataset")

    def _parquet(self, path: Path, columns: tuple[str, ...]) -> Generator[tuple[bytes, ...]]:
        try:
            parquet: _Native = importlib.import_module("pyarrow.parquet")
        except ImportError:
            raise ImportError("Install 'strata-forge[hf]' for classifier Parquet inputs.") from None
        with parquet.ParquetFile(path) as file:
            if not set(columns) <= set(file.schema_arrow.names):
                raise ClassifierWorkerError("classifier_row_invalid")
            for batch in file.iter_batches(batch_size=_PARQUET_BATCH_ROWS, columns=list(columns)):
                values: list[bytes] = []
                used = 0
                for row in batch.to_pylist():
                    raw = json.dumps(
                        row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    ).encode("utf-8")
                    if len(raw) > MAX_ROW_BYTES or used + len(raw) > _PARQUET_BATCH_BYTES:
                        raise ClassifierWorkerError("classifier_input_limit")
                    used += len(raw)
                    values.append(raw)
                yield tuple(values)

    async def aclose(self) -> None:
        self._closed = True
        try:
            if self._temporary is not None:
                await settled_io(self._temporary.cleanup)
        finally:
            self._temporary = None
            self._token = False
