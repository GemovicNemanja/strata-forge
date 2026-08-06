"""Unit tests for `strata_forge.storage.hf_hub.HFHubClient`."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from strata_forge.storage import HFHubClient


class _FakeApi:
    """Fake HfApi capturing every call."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create_repo(self, **kwargs: Any) -> str:
        self.calls.append(("create_repo", kwargs))
        return f"https://huggingface.co/{kwargs['repo_id']}"

    def delete_repo(self, **kwargs: Any) -> None:
        self.calls.append(("delete_repo", kwargs))

    def list_repo_files(self, **kwargs: Any) -> list[str]:
        self.calls.append(("list_repo_files", kwargs))
        return ["config.json", "model.safetensors", "tokenizer.json"]

    def upload_file(self, **kwargs: Any) -> str:
        self.calls.append(("upload_file", kwargs))
        return f"https://huggingface.co/{kwargs['repo_id']}/commit/abc"

    def upload_folder(self, **kwargs: Any) -> str:
        self.calls.append(("upload_folder", kwargs))
        return f"https://huggingface.co/{kwargs['repo_id']}/commit/def"


@pytest.fixture
def fake_hf(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {
        "snapshot_calls": [],
        "download_file_calls": [],
        "HfApi_kwargs": None,
        "last_api": None,
    }

    def _hf_hub_download(**kwargs: Any) -> str:
        state["download_file_calls"].append(kwargs)
        return f"/cache/{kwargs['filename']}"

    def _snapshot_download(**kwargs: Any) -> str:
        state["snapshot_calls"].append(kwargs)
        local = kwargs.get("local_dir", "/cache/" + kwargs["repo_id"])
        return str(local)

    def _make_api(**kwargs: Any) -> _FakeApi:
        state["HfApi_kwargs"] = kwargs
        api = _FakeApi(**kwargs)
        state["last_api"] = api
        return api

    fake = types.ModuleType("huggingface_hub")
    fake.hf_hub_download = _hf_hub_download  # type: ignore[attr-defined]
    fake.snapshot_download = _snapshot_download  # type: ignore[attr-defined]
    fake.HfApi = _make_api  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return state


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    async def test_lazy_api_construction(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient(token="hf_xyz", endpoint="https://hub.example.com")
        # No HfApi until first use.
        assert fake_hf["HfApi_kwargs"] is None
        await client.create_repo("me/repo")
        assert fake_hf["HfApi_kwargs"] == {
            "token": "hf_xyz",
            "endpoint": "https://hub.example.com",
        }

    async def test_explicit_api_short_circuits_construction(self, fake_hf: dict[str, Any]) -> None:
        api = _FakeApi()
        client = HFHubClient(api=api)
        await client.delete_repo("me/repo")
        # HfApi was never invoked because we passed an explicit api=.
        assert fake_hf["HfApi_kwargs"] is None
        assert api.calls == [("delete_repo", {"repo_id": "me/repo", "repo_type": "model"})]

    async def test_api_cached_across_calls(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient(token="t")
        await client.create_repo("a/b")
        first = fake_hf["last_api"]
        await client.delete_repo("a/b")
        second = fake_hf["last_api"]
        assert first is second

    def test_missing_extra_at_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import asyncio

        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        client = HFHubClient(token="t")
        with pytest.raises(ImportError, match=r"\[storage\] extra"):
            asyncio.run(client.create_repo("me/repo"))


# ---------------------------------------------------------------------------
# Repo lifecycle
# ---------------------------------------------------------------------------


class TestRepoLifecycle:
    async def test_create_repo_defaults(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        url = await client.create_repo("me/llama-finetune")
        assert url == "https://huggingface.co/me/llama-finetune"
        kwargs = fake_hf["last_api"].calls[0][1]
        assert kwargs["repo_type"] == "model"
        assert kwargs["private"] is False
        assert kwargs["exist_ok"] is True

    async def test_create_repo_private_dataset(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        await client.create_repo("me/private-ds", repo_type="dataset", private=True, exist_ok=False)
        kwargs = fake_hf["last_api"].calls[0][1]
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["private"] is True
        assert kwargs["exist_ok"] is False

    async def test_create_repo_extras_forwarded(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        await client.create_repo("me/space", repo_type="space", extras={"space_sdk": "gradio"})
        kwargs = fake_hf["last_api"].calls[0][1]
        assert kwargs["space_sdk"] == "gradio"

    async def test_delete_repo(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        await client.delete_repo("me/old-thing", repo_type="dataset")
        assert fake_hf["last_api"].calls == [
            ("delete_repo", {"repo_id": "me/old-thing", "repo_type": "dataset"})
        ]

    async def test_list_repo_files(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        files = await client.list_repo_files("me/llama", revision="v1")
        assert files == ("config.json", "model.safetensors", "tokenizer.json")
        kwargs = fake_hf["last_api"].calls[0][1]
        assert kwargs["revision"] == "v1"


# ---------------------------------------------------------------------------
# Single-file ops
# ---------------------------------------------------------------------------


class TestSingleFile:
    async def test_download_file_defaults(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient(token="hf_xyz")
        path = await client.download_file("me/llama", "config.json")
        assert isinstance(path, Path)
        assert str(path).endswith("config.json")
        kwargs = fake_hf["download_file_calls"][0]
        assert kwargs["repo_id"] == "me/llama"
        assert kwargs["filename"] == "config.json"
        assert kwargs["repo_type"] == "model"
        assert kwargs["token"] == "hf_xyz"

    async def test_download_file_with_local_dir_and_revision(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        await client.download_file(
            "me/llama",
            "model.safetensors",
            local_dir="./weights",
            revision="abc123",
            extras={"force_download": True},
        )
        kwargs = fake_hf["download_file_calls"][0]
        assert kwargs["local_dir"] == "./weights"
        assert kwargs["revision"] == "abc123"
        assert kwargs["force_download"] is True

    async def test_upload_file(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        commit = await client.upload_file(
            "./local/model.safetensors",
            "model.safetensors",
            "me/llama",
            commit_message="initial upload",
        )
        assert commit.endswith("/commit/abc")
        kwargs = fake_hf["last_api"].calls[0][1]
        assert kwargs["path_or_fileobj"] == "./local/model.safetensors"
        assert kwargs["path_in_repo"] == "model.safetensors"
        assert kwargs["repo_id"] == "me/llama"
        assert kwargs["commit_message"] == "initial upload"

    async def test_upload_file_extras_forwarded(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        await client.upload_file(
            "./x",
            "x",
            "me/r",
            extras={"create_pr": True},
        )
        kwargs = fake_hf["last_api"].calls[0][1]
        assert kwargs["create_pr"] is True


# ---------------------------------------------------------------------------
# Snapshot ops
# ---------------------------------------------------------------------------


class TestSnapshot:
    async def test_download_snapshot(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient(token="hf_x")
        path = await client.download_snapshot(
            "me/llama",
            local_dir="./snap",
            revision="main",
            allow_patterns=("*.safetensors", "*.json"),
            ignore_patterns=("*.bin",),
        )
        assert path == Path("./snap")
        kwargs = fake_hf["snapshot_calls"][0]
        assert kwargs["repo_id"] == "me/llama"
        assert kwargs["local_dir"] == "./snap"
        assert kwargs["revision"] == "main"
        assert kwargs["allow_patterns"] == ["*.safetensors", "*.json"]
        assert kwargs["ignore_patterns"] == ["*.bin"]
        assert kwargs["token"] == "hf_x"

    async def test_upload_folder(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        commit = await client.upload_folder(
            "./checkpoints/llama",
            "me/llama",
            commit_message="snapshot 42",
            allow_patterns=("*.safetensors",),
        )
        assert commit.endswith("/commit/def")
        kwargs = fake_hf["last_api"].calls[0][1]
        assert kwargs["folder_path"] == "./checkpoints/llama"
        assert kwargs["commit_message"] == "snapshot 42"
        assert kwargs["allow_patterns"] == ["*.safetensors"]


# ---------------------------------------------------------------------------
# Convenience push/pull
# ---------------------------------------------------------------------------


class TestPushPull:
    async def test_push_model_creates_then_uploads(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        commit = await client.push_model(
            "./checkpoints/llama",
            "me/llama-sft",
            commit_message="first run",
            private=True,
        )
        assert "/commit/def" in commit
        # First call is create_repo, then upload_folder.
        calls = fake_hf["last_api"].calls
        names = [c[0] for c in calls]
        assert names == ["create_repo", "upload_folder"]
        assert calls[0][1]["private"] is True
        assert calls[1][1]["repo_type"] == "model"

    async def test_pull_model_delegates_to_snapshot(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        path = await client.pull_model(
            "me/llama", "./weights", revision="r2", allow_patterns=("*.bin",)
        )
        assert path == Path("./weights")
        kwargs = fake_hf["snapshot_calls"][0]
        assert kwargs["repo_type"] == "model"
        assert kwargs["revision"] == "r2"
        assert kwargs["allow_patterns"] == ["*.bin"]

    async def test_push_dataset(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        await client.push_dataset("./data", "me/eval-set")
        calls = fake_hf["last_api"].calls
        assert calls[0][1]["repo_type"] == "dataset"
        assert calls[1][1]["repo_type"] == "dataset"

    async def test_pull_dataset(self, fake_hf: dict[str, Any]) -> None:
        client = HFHubClient()
        await client.pull_dataset("me/eval-set", "./data")
        kwargs = fake_hf["snapshot_calls"][0]
        assert kwargs["repo_type"] == "dataset"
