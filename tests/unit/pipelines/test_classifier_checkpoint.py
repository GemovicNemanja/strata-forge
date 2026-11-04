"""Pinned sidecar resolution distinguishes definite absence from unavailable authority."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import httpx
import pytest

from strata_forge.pipelines._classifier_checkpoint import (
    load_classifier_base,
    probe_forge_classifier,
)
from strata_forge.pipelines.classifier_spec import ClassifierWorkerError

if TYPE_CHECKING:
    from pathlib import Path

# Malformed JSON fixtures and installed native ML/mock objects are intentionally dynamic.


def test_only_definite_remote_file_404_is_native_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub as hub
    from huggingface_hub.errors import RemoteEntryNotFoundError

    def absent(*args: Any, **kwargs: Any) -> Any:
        raise RemoteEntryNotFoundError(
            "fictional-private-message",
            response=httpx.Response(
                404, request=httpx.Request("GET", "https://fictional.invalid/")
            ),
        )

    monkeypatch.setattr(hub, "hf_hub_download", absent)
    assert probe_forge_classifier("fixture/model", revision="1" * 40, token=False) is None

    def unavailable(*args: Any, **kwargs: Any) -> Any:
        raise TimeoutError("fictional-private-message")

    monkeypatch.setattr(hub, "hf_hub_download", unavailable)
    with pytest.raises(ClassifierWorkerError) as error:
        probe_forge_classifier("fixture/model", revision="1" * 40, token=False)
    assert error.value.__context__ is None
    assert "private-message" not in str(error.value)


def test_probe_refuses_unpinned_before_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub as hub

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("no probe before exact pin")

    monkeypatch.setattr(hub, "hf_hub_download", forbidden)
    with pytest.raises(ClassifierWorkerError):
        probe_forge_classifier("fixture/model", revision="main", token=False)


def test_malformed_sidecar_refuses_before_weight_inventory_or_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub as hub

    sidecar = tmp_path / "strata_classifier.json"
    sidecar.write_text("{}")
    calls = []

    def download(model_id: Any, filename: str, **kwargs: Any) -> Any:
        calls.append(filename)
        assert filename == "strata_classifier.json"
        return str(sidecar)

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("malformed metadata must refuse before inventory")

    monkeypatch.setattr(hub, "hf_hub_download", download)
    monkeypatch.setattr(hub, "HfApi", forbidden)
    with pytest.raises(ClassifierWorkerError, match="classifier_artifact_unavailable") as error:
        probe_forge_classifier("fixture/model", revision="1" * 40, token=False)
    assert calls == ["strata_classifier.json"]
    assert error.value.__context__ is None


@pytest.mark.parametrize(
    ("file", "body"),
    [
        ("model.safetensors.index.json", '{"weight_map":{"body":"../outside.safetensors"}}'),
        ("tokenizer_config.json", '{"tokenizer_file":"../outside.json"}'),
    ],
)
async def test_checkpoint_asset_paths_cannot_escape_prepared_inventory(
    tmp_path: Path, file: str, body: str
) -> None:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fictional not actual weights")
    (tmp_path / file).write_text(body)
    with pytest.raises(ClassifierWorkerError, match="classifier_base_invalid"):
        await load_classifier_base(tmp_path)
