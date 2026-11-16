"""Pinned source adapters use fictional Hub metadata and real private files only."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from strata_forge.pipelines import _classifier_worker_source as source
from strata_forge.pipelines.classifier_runner import TrainingReadDecision
from strata_forge.pipelines.classifier_spec import ClassifierTrainSpec, ClassifierWorkerError
from tests.unit.pipelines.test_classifier_spec import payload

# Deliberately dynamic optional-Hub/Arrow fixtures are confined to this synthetic boundary.


class Authority:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed, self.calls = allowed, []

    async def decide(self, request: Any) -> TrainingReadDecision:
        self.calls.append(request)
        return TrainingReadDecision(
            **request.model_dump(),
            authorized=self.allowed,
            custody="unsealed",
            policy_version="fictional-training-v1",
        )


@dataclass
class File:
    path: str
    size: int
    blob_id: str
    lfs: Any = None


@dataclass
class Folder:
    path: str


class Hub:
    RepoFile, RepoFolder = File, Folder

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files, self.downloads, self.api_calls = files, [], []
        self.tree: list[File | Folder] = [
            File(
                name,
                len(raw),
                hashlib.sha1(
                    f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False
                ).hexdigest(),
            )
            for name, raw in files.items()
        ]

    def HfApi(self, **kwargs: Any) -> Hub:  # noqa: N802 — upstream optional API name
        self.api_calls.append(kwargs)
        return self

    def repo_info(self, **kwargs: Any) -> Any:
        return SimpleNamespace(sha=kwargs["revision"])

    def list_repo_tree(self, *_: Any, **__: Any) -> Any:
        return iter(self.tree)

    def hf_hub_download(self, repo: str, name: str, **kwargs: Any) -> str:
        self.downloads.append((repo, name, kwargs))
        path = Path(kwargs["local_dir"]) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[name])
        return str(path)


def instance(
    tmp_path: Path, hub: Hub, monkeypatch: pytest.MonkeyPatch, *, allowed: bool = True
) -> source.HubClassifierSource:
    spec = ClassifierTrainSpec.model_validate(payload())
    value = source.HubClassifierSource(
        spec, authority=Authority(allowed=allowed), token=False, parent=tmp_path
    )
    monkeypatch.setattr(value, "_hub", lambda: hub)
    return value


@pytest.mark.parametrize(
    "path", ["/abs", "../parent", "a/../b", "a//b", "a\\b", "https://host/a", "a\x00b"]
)
def test_paths_refuse_ambiguous_or_external_names(path: str) -> None:
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_source_invalid$"):
        source._path(path)


@pytest.mark.parametrize("lfs", [True, False])
def test_verified_copy_uses_actual_pinned_blob_and_private_destination(
    tmp_path: Path, lfs: bool
) -> None:
    raw = b"fictional bytes\n"
    original, target = tmp_path / "original", tmp_path / "copy"
    original.write_bytes(raw)
    expected = (
        hashlib.sha256(raw).hexdigest()
        if lfs
        else hashlib.sha1(f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False).hexdigest()
    )
    source._verified_file(original, source._Entry(len(raw), expected, lfs), len(raw), target)
    assert target.read_bytes() == raw
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.stat().st_ino != original.stat().st_ino


def test_actual_wrong_blob_or_symlink_never_becomes_verified(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"changed")
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_source_invalid$"):
        source._verified_file(original, source._Entry(7, "0" * 64, True), 100)
    alias = tmp_path / "alias"
    alias.symlink_to(original)
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_source_invalid$"):
        source._verified_file(alias, source._Entry(7, "0" * 64, True), 100)


async def test_denied_source_refuses_before_hub_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = Hub({})
    value = instance(tmp_path, hub, monkeypatch, allowed=False)
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_read_denied$"):
        await value.resolve_model(value.spec.base.ref)
    assert not hub.api_calls
    assert not await asyncio.to_thread(lambda: list(tmp_path.iterdir()))


async def test_safe_model_staging_ignores_docs_pickle_and_hub_hidden_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {
        ".gitattributes": b"metadata",
        "config.json": b"{}",
        "model.safetensors": b"fictional weights",
        "pytorch_model.bin": b"must not read",
        "README.md": b"not input",
    }
    hub = Hub(files)
    value = instance(tmp_path, hub, monkeypatch)
    result = await value.resolve_model(value.spec.base.ref)
    assert sorted(p.name for p in result.directory.iterdir()) == [
        "config.json",
        "model.safetensors",
    ]
    assert {name for _, name, _ in hub.downloads} == {"config.json", "model.safetensors"}
    assert all(
        kwargs["token"] is False and kwargs["endpoint"] == "https://huggingface.co"
        for _, _, kwargs in hub.downloads
    )
    directory = result.directory
    await value.aclose()
    assert not directory.exists()


async def test_tree_limit_counts_folders_and_stops_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = Hub({})
    hub.tree = [Folder(f"folder-{i}") for i in range(source._MODEL_ENTRIES + 1)]
    value = instance(tmp_path, hub, monkeypatch)
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_source_invalid$"):
        await value.resolve_model(value.spec.base.ref)
    assert not hub.downloads
    await value.aclose()


async def test_adapter_or_oversize_model_refuses_without_weight_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = Hub({"config.json": b"{}", "model.safetensors": b"bytes", "adapter_config.json": b"{}"})
    value = instance(tmp_path, hub, monkeypatch)
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_checkpoint_invalid$"):
        await value.resolve_model(value.spec.base.ref)
    assert not hub.downloads


async def test_parquet_rows_are_column_mapped_bounded_and_current(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b"],
                "text": ["one", "two"],
                "label": ["E", "C"],
                "unused": ["private canary", "private canary"],
            }
        ),
        parquet,
    )
    card = b"---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train.parquet\n---\n"
    hub = Hub({"README.md": card, "data/train.parquet": parquet.read_bytes()})
    value = instance(tmp_path, hub, monkeypatch)
    rows = [json.loads(raw) async for raw in value.rows(value.spec.dataset)]
    assert rows == [
        {"id": "a", "text": "one", "label": "E"},
        {"id": "b", "text": "two", "label": "C"},
    ]
    assert len(value.authority.calls) >= 4
    await value.aclose()


async def test_row_cap_refuses_before_second_batch_is_delivered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    card = b"---\nconfigs:\n- config_name: default\n  data_files:\n  - split: train\n    path: data/train.parquet\n---\n"
    hub = Hub({"README.md": card, "data/train.parquet": b"synthetic file"})
    value = instance(tmp_path, hub, monkeypatch)
    closed = []

    def rows(*_: Any) -> Any:
        try:
            yield (b"1234",)
            yield (b"5678",)
        finally:
            closed.append(True)

    monkeypatch.setattr(value, "_parquet", rows)
    monkeypatch.setattr(source, "MAX_INPUT_BYTES", 7)
    stream = value.rows(value.spec.dataset)
    assert await anext(stream) == b"1234"
    with pytest.raises(ClassifierWorkerError, match=r"^classifier_input_limit$"):
        await anext(stream)
    assert closed == [True]
    await value.aclose()


async def test_repeated_cancel_owns_native_thread_until_actual_return() -> None:
    entered, release = threading.Event(), threading.Event()

    def held() -> int:
        entered.set()
        release.wait(2)
        return 7

    task = asyncio.create_task(source.settled_io(held))
    try:
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert entered.is_set()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_cancelled_cleanup_drains_private_files_and_forgets_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    value = instance(tmp_path, Hub({}), monkeypatch)
    value._token = "hf_fictional_private_canary"
    root = value._root()
    assert value._temporary is not None
    original_cleanup = value._temporary.cleanup
    entered, release = threading.Event(), threading.Event()

    def held_cleanup() -> None:
        entered.set()
        release.wait(2)
        original_cleanup()

    monkeypatch.setattr(value._temporary, "cleanup", held_cleanup)
    task = asyncio.create_task(value.aclose())
    try:
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.001)
        assert entered.is_set()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not root.exists()
    assert value._temporary is None
    assert value._token is False


@pytest.mark.parametrize("control", [SystemExit, KeyboardInterrupt, asyncio.CancelledError])
async def test_native_controls_are_contained_inside_the_thread(
    control: type[BaseException],
) -> None:
    def refuse() -> None:
        raise control

    with pytest.raises(ClassifierWorkerError, match=r"^classifier_source_invalid$"):
        await source.settled_io(refuse)
