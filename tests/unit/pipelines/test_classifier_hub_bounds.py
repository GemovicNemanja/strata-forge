"""Actual installed Hub code uses only synthetic HTTP, never external models or requests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from strata_forge.pipelines._classifier_checkpoint import probe_forge_classifier
from strata_forge.pipelines.classifier_spec import ClassifierWorkerError
from strata_forge.training._classifier_heads import HeadInitializationReceipt
from strata_forge.training.classifier import ClassifierConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

_REVISION = "1" * 40
_MODEL = "fixture/model"
_RESOLVE = f"/{_MODEL}/resolve/{_REVISION}/"
_TREE = f"/api/models/{_MODEL}/tree/{_REVISION}"


def sidecar() -> bytes:
    initialization = HeadInitializationReceipt(
        initializer_id="encoder_random_head",
        family="bert",
        labels=("E", "C"),
        created_keys=("classifier.weight",),
        removed_keys=(),
        body_key_sha256="0" * 64,
        body_key_count=1,
        pooling="native_encoder",
        seed=0,
    )
    return json.dumps(
        {
            "version": 1,
            "implementation": "forge_primary_classifier_v1",
            "config": ClassifierConfig(labels=("E", "C")).model_dump(mode="json"),
            "initialization": initialization.model_dump(mode="json"),
            "checkpoint_selection": "final",
            "steps": 1,
            "trained_rows": 1,
            "overflow_ids": [],
            "overflow_count": 0,
            "overflow_omitted": 0,
            "environment": {},
        }
    ).encode()


@pytest.fixture
def mock_hub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[Path, Callable[[Callable[[httpx.Request], httpx.Response]], None]]]:
    import huggingface_hub as hub
    from huggingface_hub import constants
    from huggingface_hub.utils import (
        _http,  # pyright: ignore[reportPrivateUsage] — snapshot and restore only the SDK's test transport state
    )

    original_factory = _http._GLOBAL_CLIENT_FACTORY  # pyright: ignore[reportPrivateUsage] — restore the SDK's exact process-local test state

    cache = tmp_path / "cache"
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_TELEMETRY", True)
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_PROGRESS_BARS", True)

    def install(handler: Callable[[httpx.Request], httpx.Response]) -> None:
        hub.set_client_factory(lambda: httpx.Client(transport=httpx.MockTransport(handler)))

    try:
        yield cache, install
    finally:
        hub.set_client_factory(original_factory)


def headers(size: int, *, commit: str = _REVISION) -> dict[str, str]:
    return {"X-Repo-Commit": commit, "ETag": '"' + "2" * 40 + '"', "Content-Length": str(size)}


def test_oversized_sidecar_never_gets_or_caches_body(
    mock_hub: tuple[Path, Callable[[Callable[[httpx.Request], httpx.Response]], None]],
) -> None:
    cache, install = mock_hub
    calls: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "huggingface.co"
        assert request.url.path == _RESOLVE + "strata_classifier.json"
        assert "authorization" not in request.headers
        calls.append(request.method)
        return httpx.Response(200, headers=headers(262_144))

    install(transport)
    with pytest.raises(ClassifierWorkerError, match="classifier_artifact_unavailable"):
        probe_forge_classifier(_MODEL, revision=_REVISION, token=False)
    assert calls == ["HEAD"]
    assert not list(cache.rglob("blobs/*"))


def test_actual_entry_not_found_is_native_fallback_without_get(
    mock_hub: tuple[Path, Callable[[Callable[[httpx.Request], httpx.Response]], None]],
) -> None:
    cache, install = mock_hub
    calls: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        assert request.url.path == _RESOLVE + "strata_classifier.json"
        return httpx.Response(404, headers={"X-Error-Code": "EntryNotFound"})

    install(transport)
    assert probe_forge_classifier(_MODEL, revision=_REVISION, token=False) is None
    assert calls == ["HEAD"]
    assert not list(cache.rglob("blobs/*"))


def test_unsupported_preflight_never_uses_unbounded_legacy_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub as hub

    called: list[str] = []

    def legacy(repo: str, name: str, *, revision: str, token: object) -> str:
        called.append(name)
        raise AssertionError("unsupported dry_run must not use a second unbounded call")

    monkeypatch.setattr(hub, "hf_hub_download", legacy)
    with pytest.raises(ClassifierWorkerError, match="classifier_artifact_unavailable"):
        probe_forge_classifier(_MODEL, revision=_REVISION, token=False)
    assert called == []


@pytest.mark.parametrize("failure", ["missing_size", "different_commit"])
def test_unknown_or_wrong_pin_preflight_never_gets_body(
    mock_hub: tuple[Path, Callable[[Callable[[httpx.Request], httpx.Response]], None]],
    failure: str,
) -> None:
    cache, install = mock_hub
    calls: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        values = headers(4, commit="3" * 40 if failure == "different_commit" else _REVISION)
        if failure == "missing_size":
            del values["Content-Length"]
        return httpx.Response(200, headers=values, stream=httpx.ByteStream(b""))

    install(transport)
    with pytest.raises(ClassifierWorkerError, match="classifier_artifact_unavailable"):
        probe_forge_classifier(_MODEL, revision=_REVISION, token=False)
    assert calls == ["HEAD"]
    assert not list(cache.rglob("blobs/*"))


def test_oversized_metadata_never_gets_asset_or_weights(
    mock_hub: tuple[Path, Callable[[Callable[[httpx.Request], httpx.Response]], None]],
) -> None:
    cache, install = mock_hub
    body = sidecar()
    calls: list[tuple[str, str]] = []

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "huggingface.co"
        assert "authorization" not in request.headers
        calls.append((request.method, request.url.path))
        if request.url.path == _TREE:
            return httpx.Response(
                200,
                json=[
                    {"type": "file", "path": name, "size": size, "oid": "2" * 40}
                    for name, size in (
                        ("strata_classifier.json", len(body)),
                        ("model.safetensors", 1),
                        ("config.json", 32 * 1024 * 1024 + 1),
                    )
                ],
            )
        if request.url.path == _RESOLVE + "strata_classifier.json":
            if request.method == "HEAD":
                return httpx.Response(200, headers=headers(len(body)))
            return httpx.Response(200, headers=headers(len(body)), content=body)
        assert request.url.path == _RESOLVE + "config.json"
        return httpx.Response(200, headers=headers(32 * 1024 * 1024 + 1))

    install(transport)
    with pytest.raises(ClassifierWorkerError, match="classifier_artifact_unavailable"):
        probe_forge_classifier(_MODEL, revision=_REVISION, token=False)
    assert ("HEAD", _RESOLVE + "config.json") in calls
    assert [(method, path) for method, path in calls if method == "GET" and path != _TREE] == [
        ("GET", _RESOLVE + "strata_classifier.json")
    ]
    assert [file.stat().st_size for file in cache.rglob("blobs/*") if file.is_file()] == [len(body)]


def test_inventory_counts_folders_and_stops_before_later_pages_or_assets(
    mock_hub: tuple[Path, Callable[[Callable[[httpx.Request], httpx.Response]], None]],
) -> None:
    _, install = mock_hub
    body = sidecar()
    pages: list[int] = []
    downloads: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "huggingface.co"
        assert "authorization" not in request.headers
        if request.url.path == _TREE:
            page = int(request.url.params.get("cursor", "0"))
            pages.append(page)
            assert page < 3, "iteration must stop at entry 129 before requesting page four"
            if page == 0:
                assert request.url.params.get("recursive") == "true"
                assert request.url.params.get("expand") == "true"
            entries: list[dict[str, str | int]] = [
                {"type": "directory", "path": f"folder{i}", "oid": "2" * 40}
                for i in range(page * 50, min((page + 1) * 50, 129))
            ]
            if page == 0:
                entries[0] = {
                    "type": "file",
                    "path": "strata_classifier.json",
                    "size": len(body),
                    "oid": "2" * 40,
                }
            return httpx.Response(
                200,
                json=entries,
                headers={"Link": f'<https://huggingface.co{_TREE}?cursor={page + 1}>; rel="next"'},
            )
        assert request.url.path == _RESOLVE + "strata_classifier.json"
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers(len(body)))
        downloads.append(request.url.path)
        return httpx.Response(200, headers=headers(len(body)), content=body)

    install(transport)
    with pytest.raises(ClassifierWorkerError, match="classifier_artifact_unavailable"):
        probe_forge_classifier(_MODEL, revision=_REVISION, token=False)
    assert pages == [0, 1, 2]
    assert downloads == [_RESOLVE + "strata_classifier.json"]
