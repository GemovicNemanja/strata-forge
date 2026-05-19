"""Unit tests for `forge.prompts.stores.langfuse`."""

from __future__ import annotations

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock

import pytest

from forge.core.errors import ForgeError
from forge.prompts.registry import PromptNotFoundError, PromptStore
from forge.prompts.stores.langfuse import LangfusePromptStore
from forge.prompts.template import PromptTemplate


def _fake_prompt(
    *,
    name: str,
    body: str | None = None,
    config: dict[str, Any] | None = None,
    version: int = 1,
) -> MagicMock:
    """Build an object that quacks like a Langfuse `Prompt`."""
    if body is None:
        body = json.dumps({"stable_section": "hi", "dynamic_section": ""})
    prompt = MagicMock()
    prompt.name = name
    prompt.prompt = body
    prompt.config = config or {}
    prompt.version = version
    return prompt


# ---------------------------------------------------------------------------
# Conformance + construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_is_promptstore(self) -> None:
        assert isinstance(LangfusePromptStore(client=MagicMock()), PromptStore)

    async def test_client_injection_skips_import(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # If the constructor tried to import langfuse despite the `client=`
        # arg, this monkeypatch would force the import to fail. Verifying
        # via behavior (a successful get call) rather than poking at the
        # private `_client` attribute.
        monkeypatch.setitem(sys.modules, "langfuse", None)
        client = MagicMock()
        client.get_prompt.return_value = _fake_prompt(name="x")
        store = LangfusePromptStore(client=client)

        result = await store.get("x")
        assert result.name == "x"

    def test_import_error_when_langfuse_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Defensive: even on a system where someone managed to install
        # langfuse, force the import to fail and verify the error message.
        monkeypatch.setitem(sys.modules, "langfuse", None)
        with pytest.raises(ImportError, match=r"\[langfuse\] extra"):
            LangfusePromptStore()

    def test_construction_with_explicit_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Inject a fake `langfuse` module so the constructor's lazy import
        # resolves; verify the kwargs are forwarded to its Langfuse class.
        constructed_with: dict[str, Any] = {}
        fake_module = types.ModuleType("langfuse")

        class _FakeLangfuse:
            def __init__(self, **kwargs: Any) -> None:
                constructed_with.update(kwargs)

        fake_module.Langfuse = _FakeLangfuse  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse", fake_module)

        LangfusePromptStore(host="https://test", public_key="pk", secret_key="sk")
        assert constructed_with == {
            "host": "https://test",
            "public_key": "pk",
            "secret_key": "sk",
        }

    def test_construction_omits_none_kwargs(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # When the caller passes nothing, the constructor builds the
        # client with no kwargs so Langfuse falls back to its own env reading.
        constructed_with: dict[str, Any] = {}
        fake_module = types.ModuleType("langfuse")

        class _FakeLangfuse:
            def __init__(self, **kwargs: Any) -> None:
                constructed_with.update(kwargs)

        fake_module.Langfuse = _FakeLangfuse  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "langfuse", fake_module)

        LangfusePromptStore()
        assert constructed_with == {}


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    async def test_get_latest_serializes_correctly(self) -> None:
        client = MagicMock()
        body = json.dumps(
            {"stable_section": "You are helpful.", "dynamic_section": "Hi {{ name }}"}
        )
        config = {
            "forge_template_v1": True,
            "stable_variables": [],
            "dynamic_variables": ["name"],
            "description": "greeter",
            "metadata": {"owner": "team"},
        }
        client.get_prompt.return_value = _fake_prompt(
            name="greet", body=body, config=config, version=3
        )
        store = LangfusePromptStore(client=client)

        result = await store.get("greet")

        # Defaults to label="latest" so the store round-trips its own
        # writes (new prompts only carry "latest" until promoted).
        client.get_prompt.assert_called_once_with("greet", label="latest")
        assert isinstance(result, PromptTemplate)
        assert result.name == "greet"
        assert result.stable_section == "You are helpful."
        assert result.dynamic_section == "Hi {{ name }}"
        assert result.dynamic_variables == ("name",)
        assert result.description == "greeter"
        assert result.metadata == {"owner": "team"}

    async def test_get_specific_version_passes_int(self) -> None:
        client = MagicMock()
        client.get_prompt.return_value = _fake_prompt(name="x", version=2)
        store = LangfusePromptStore(client=client)

        await store.get("x", "2")
        client.get_prompt.assert_called_once_with("x", version=2)

    async def test_get_failure_raises_promptnotfound(self) -> None:
        client = MagicMock()
        client.get_prompt.side_effect = RuntimeError("404")
        store = LangfusePromptStore(client=client)

        with pytest.raises(PromptNotFoundError) as info:
            await store.get("missing")
        assert info.value.name == "missing"
        assert info.value.__cause__ is not None

    async def test_get_specific_version_failure_records_version(self) -> None:
        client = MagicMock()
        client.get_prompt.side_effect = RuntimeError("404")
        store = LangfusePromptStore(client=client)

        with pytest.raises(PromptNotFoundError) as info:
            await store.get("x", "99")
        assert info.value.version == "99"

    async def test_get_malformed_body_raises_forge_error(self) -> None:
        client = MagicMock()
        client.get_prompt.return_value = _fake_prompt(name="x", body="this is not JSON")
        store = LangfusePromptStore(client=client)

        with pytest.raises(ForgeError, match="not the JSON format"):
            await store.get("x")

    async def test_get_handles_empty_config(self) -> None:
        # Older or externally-created prompts may have no config dict.
        client = MagicMock()
        body = json.dumps({"stable_section": "hi", "dynamic_section": ""})
        prompt = _fake_prompt(name="x", body=body)
        prompt.config = None
        client.get_prompt.return_value = prompt
        store = LangfusePromptStore(client=client)

        result = await store.get("x")
        assert result.stable_variables == ()
        assert result.dynamic_variables == ()
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# put
# ---------------------------------------------------------------------------


class TestPut:
    async def test_put_serializes_template(self) -> None:
        client = MagicMock()
        client.create_prompt.return_value = MagicMock(version=5)
        store = LangfusePromptStore(client=client)

        template = PromptTemplate(
            name="greet",
            stable_section="You are helpful.",
            dynamic_section="Hi {{ name }}",
            dynamic_variables=("name",),
            description="greeter",
            metadata={"owner": "team"},
        )
        version = await store.put(template)

        assert version == "5"
        call = client.create_prompt.call_args
        kwargs = call.kwargs
        assert kwargs["name"] == "greet"
        # Body is JSON with both sections.
        body = json.loads(kwargs["prompt"])
        assert body == {
            "stable_section": "You are helpful.",
            "dynamic_section": "Hi {{ name }}",
        }
        # Config carries the typed fields.
        config = kwargs["config"]
        assert config["forge_template_v1"] is True
        assert config["dynamic_variables"] == ["name"]
        assert config["description"] == "greeter"
        assert config["metadata"] == {"owner": "team"}

    async def test_put_returns_stringified_version(self) -> None:
        # Langfuse returns integer versions; the store stringifies them.
        client = MagicMock()
        client.create_prompt.return_value = MagicMock(version=42)
        store = LangfusePromptStore(client=client)

        version = await store.put(PromptTemplate(name="x", stable_section="hi"))
        assert version == "42"
        assert isinstance(version, str)


# ---------------------------------------------------------------------------
# versions
# ---------------------------------------------------------------------------


class TestVersions:
    async def test_versions_walks_from_latest_to_one(self) -> None:
        client = MagicMock()
        client.get_prompt.return_value = _fake_prompt(name="x", version=4)
        store = LangfusePromptStore(client=client)

        result = await store.versions("x")
        assert result == ["4", "3", "2", "1"]

    async def test_versions_for_single_version(self) -> None:
        client = MagicMock()
        client.get_prompt.return_value = _fake_prompt(name="x", version=1)
        store = LangfusePromptStore(client=client)

        assert await store.versions("x") == ["1"]

    async def test_versions_for_unknown_name_raises(self) -> None:
        client = MagicMock()
        client.get_prompt.side_effect = RuntimeError("404")
        store = LangfusePromptStore(client=client)

        with pytest.raises(PromptNotFoundError) as info:
            await store.versions("missing")
        assert info.value.name == "missing"


# ---------------------------------------------------------------------------
# list_names
# ---------------------------------------------------------------------------


class _FakeAsyncClient:
    """Drop-in for httpx.AsyncClient that returns scripted responses."""

    def __init__(self, responses: list[dict[str, Any]] | Exception) -> None:
        self._responses = responses

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        return None

    async def get(self, *_args: Any, **_kwargs: Any) -> MagicMock:
        if isinstance(self._responses, Exception):
            raise self._responses
        payload = self._responses.pop(0)
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = payload
        return resp


class TestListNames:
    # The v4 SDK dropped client.api.prompts.list, so list_names hits the
    # REST endpoint directly. Tests mock httpx.AsyncClient; the store
    # needs explicit host/public_key/secret_key to skip the env-var
    # fallback.

    @staticmethod
    def _store_with_creds() -> LangfusePromptStore:
        return LangfusePromptStore(
            client=MagicMock(),
            host="http://test",
            public_key="pk-test",
            secret_key="sk-test",
        )

    @staticmethod
    def _patch_httpx(
        monkeypatch: pytest.MonkeyPatch,
        responses: list[dict[str, Any]] | Exception,
    ) -> None:
        import httpx

        def _make_client(*_args: Any, **_kwargs: Any) -> _FakeAsyncClient:
            return _FakeAsyncClient(responses)

        monkeypatch.setattr(httpx, "AsyncClient", _make_client)

    async def test_returns_sorted_unique_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        page = {
            "data": [
                {"name": "zeta"},
                {"name": "alpha"},
                {"name": "mu"},
                {"name": "alpha"},  # duplicate
            ],
            "meta": {"totalPages": 1},
        }
        self._patch_httpx(monkeypatch, [page])
        store = self._store_with_creds()
        assert await store.list_names() == ["alpha", "mu", "zeta"]

    async def test_empty_listing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_httpx(monkeypatch, [{"data": [], "meta": {"totalPages": 1}}])
        store = self._store_with_creds()
        assert await store.list_names() == []

    async def test_listing_failure_raises_forge_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_httpx(monkeypatch, RuntimeError("server unhappy"))
        store = self._store_with_creds()
        with pytest.raises(ForgeError, match="list_names failed"):
            await store.list_names()

    async def test_missing_credentials_raises_forge_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No env credentials, no constructor args → ForgeError surfaces
        # the misconfiguration instead of dispatching an unauthenticated
        # REST call.
        for var in ("LANGFUSE_HOST", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"):
            monkeypatch.delenv(var, raising=False)
        store = LangfusePromptStore(client=MagicMock())
        with pytest.raises(ForgeError, match="LANGFUSE_HOST"):
            await store.list_names()

    async def test_paginates_through_multiple_pages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        responses = [
            {
                "data": [{"name": "alpha"}, {"name": "beta"}],
                "meta": {"totalPages": 2},
            },
            {
                "data": [{"name": "gamma"}],
                "meta": {"totalPages": 2},
            },
        ]
        self._patch_httpx(monkeypatch, responses)
        store = self._store_with_creds()
        assert await store.list_names() == ["alpha", "beta", "gamma"]


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_raises_not_implemented(self) -> None:
        client = MagicMock()
        store = LangfusePromptStore(client=client)

        with pytest.raises(NotImplementedError, match="does not implement delete"):
            await store.delete("x")

    async def test_delete_with_version_raises_not_implemented(self) -> None:
        client = MagicMock()
        store = LangfusePromptStore(client=client)

        with pytest.raises(NotImplementedError):
            await store.delete("x", "1")


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


class TestRoundTrip:
    async def test_put_then_get_recovers_template(self) -> None:
        # The store serializes to and deserializes from the same shape.
        original = PromptTemplate(
            name="greet",
            stable_section="You are helpful.",
            dynamic_section="Hi {{ name }}",
            stable_variables=(),
            dynamic_variables=("name",),
            description="A greeter",
            metadata={"owner": "team-llm", "tag": "v1"},
        )

        captured: dict[str, Any] = {}

        def _record_create(**kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            return MagicMock(version=1)

        client = MagicMock()
        client.create_prompt.side_effect = _record_create

        def _record_get(name: str, version: int | None = None) -> MagicMock:
            del version
            return _fake_prompt(
                name=name,
                body=captured["prompt"],
                config=captured["config"],
                version=1,
            )

        client.get_prompt.side_effect = _record_get

        store = LangfusePromptStore(client=client)
        version = await store.put(original)
        recovered = await store.get(original.name, version)
        assert recovered == original
