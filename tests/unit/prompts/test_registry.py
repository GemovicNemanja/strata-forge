"""Unit tests for `forge.prompts.registry`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from structlog.testing import capture_logs

from forge.core.errors import ForgeError
from forge.prompts.registry import (
    PromptNotFoundError,
    PromptRegistry,
    PromptStore,
)
from forge.prompts.template import PromptTemplate, PromptValidationError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


# Long enough to clear the cacheable threshold for the lint signal.
_LONG_STABLE = (
    "You are a helpful and concise assistant. Always think step by step "
    "before answering. When asked a factual question, prefer accuracy over "
    "speculation. When given an example, infer the user's intended pattern "
    "and follow it exactly. Use the provided tools when they would produce "
    "more accurate results than answering from training data alone. Avoid "
    "filler phrases. Match the user's level of formality. Cite sources "
    "where relevant. When confident in a numerical answer, give the number "
    "first and the reasoning afterward. When uncertain, say so explicitly "
    "and quantify the uncertainty when possible. Never fabricate output."
)


def _mock_store(**overrides: object) -> AsyncMock:
    """Build an AsyncMock conforming to the PromptStore interface."""
    store = AsyncMock(spec=PromptStore)
    for key, value in overrides.items():
        getattr(store, key).return_value = value
    return store


# ---------------------------------------------------------------------------
# PromptNotFoundError
# ---------------------------------------------------------------------------


class TestPromptNotFoundError:
    def test_is_forge_error(self) -> None:
        assert issubclass(PromptNotFoundError, ForgeError)

    def test_carries_name(self) -> None:
        err = PromptNotFoundError("missing", name="greet")
        assert err.name == "greet"
        assert err.version is None

    def test_carries_name_and_version(self) -> None:
        err = PromptNotFoundError("missing v3", name="greet", version="3")
        assert err.name == "greet"
        assert err.version == "3"


# ---------------------------------------------------------------------------
# PromptStore — abstract base
# ---------------------------------------------------------------------------


class TestPromptStoreAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            PromptStore()  # pyright: ignore[reportAbstractUsage]


# ---------------------------------------------------------------------------
# PromptRegistry — delegation
# ---------------------------------------------------------------------------


class TestRegistryDelegation:
    async def test_get_delegates(self) -> None:
        target = PromptTemplate(name="x", stable_section="hi")
        store = _mock_store(get=target)
        registry = PromptRegistry(store)

        result = await registry.get("x")

        store.get.assert_awaited_once_with("x", None)
        assert result is target

    async def test_get_with_version(self) -> None:
        store = _mock_store(get=PromptTemplate(name="x", stable_section="hi"))
        registry = PromptRegistry(store)

        await registry.get("x", "5")
        store.get.assert_awaited_once_with("x", "5")

    async def test_versions_delegates(self) -> None:
        store = _mock_store(versions=["3", "2", "1"])
        registry = PromptRegistry(store)

        result = await registry.versions("x")

        store.versions.assert_awaited_once_with("x")
        assert result == ["3", "2", "1"]

    async def test_list_names_delegates(self) -> None:
        store = _mock_store(list_names=["a", "b", "c"])
        registry = PromptRegistry(store)

        result = await registry.list_names()

        store.list_names.assert_awaited_once_with()
        assert result == ["a", "b", "c"]

    async def test_delete_delegates(self) -> None:
        store = _mock_store()
        registry = PromptRegistry(store)

        await registry.delete("x", "3")
        store.delete.assert_awaited_once_with("x", "3")

    async def test_delete_all_versions(self) -> None:
        store = _mock_store()
        registry = PromptRegistry(store)

        await registry.delete("x")
        store.delete.assert_awaited_once_with("x", None)


# ---------------------------------------------------------------------------
# PromptRegistry.put — validation + lint
# ---------------------------------------------------------------------------


class TestRegistryPut:
    async def test_valid_template_stored(self) -> None:
        store = _mock_store(put="1")
        registry = PromptRegistry(store)

        template = PromptTemplate(
            name="ok",
            stable_section=_LONG_STABLE,
            dynamic_section="Q: {{ q }}",
            dynamic_variables=("q",),
        )
        version = await registry.put(template)

        assert version == "1"
        store.put.assert_awaited_once_with(template)

    async def test_returns_store_assigned_version(self) -> None:
        store = _mock_store(put="42")
        registry = PromptRegistry(store)
        version = await registry.put(
            PromptTemplate(name="x", stable_section=_LONG_STABLE)
        )
        assert version == "42"

    async def test_invalid_template_rejected_before_store_call(self) -> None:
        # `query` referenced but not declared.
        store = _mock_store()
        registry = PromptRegistry(store)

        bad = PromptTemplate(
            name="bad",
            stable_section="",
            dynamic_section="{{ query }}",
        )
        with pytest.raises(PromptValidationError):
            await registry.put(bad)
        # Store.put was never reached — validation guards it.
        store.put.assert_not_awaited()

    async def test_overlap_rejected_before_store_call(self) -> None:
        store = _mock_store()
        registry = PromptRegistry(store)

        bad = PromptTemplate(
            name="bad",
            stable_section="{{ x }}",
            dynamic_section="{{ x }}",
            stable_variables=("x",),
            dynamic_variables=("x",),
        )
        with pytest.raises(PromptValidationError):
            await registry.put(bad)
        store.put.assert_not_awaited()


# ---------------------------------------------------------------------------
# PromptRegistry.put — lint signal for short stable section
# ---------------------------------------------------------------------------


def _warning_logs(
    records: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        r
        for r in records
        if r.get("log_level") == "warning"
        and r.get("event") == "prompt.stable_section_too_short"
    ]


class TestRegistryLint:
    async def test_warns_when_stable_section_too_short(self) -> None:
        store = _mock_store(put="1")
        registry = PromptRegistry(store)

        with capture_logs() as records:
            await registry.put(
                PromptTemplate(
                    name="short",
                    stable_section="Be helpful.",
                    dynamic_section="",
                )
            )
        warnings = _warning_logs(records)
        assert len(warnings) == 1
        assert warnings[0]["template_name"] == "short"
        # The lint signal is informational — the template was still stored.
        store.put.assert_awaited_once()

    async def test_no_warning_when_stable_section_long_enough(self) -> None:
        store = _mock_store(put="1")
        registry = PromptRegistry(store)

        with capture_logs() as records:
            await registry.put(
                PromptTemplate(name="long", stable_section=_LONG_STABLE)
            )
        assert _warning_logs(records) == []

    async def test_no_warning_for_simple_shorthand(self) -> None:
        # `PromptTemplate.simple` produces an empty stable section by design.
        # The lint should NOT fire — the author opted out of caching explicitly.
        store = _mock_store(put="1")
        registry = PromptRegistry(store)

        with capture_logs() as records:
            await registry.put(
                PromptTemplate.simple(
                    "q", "What is {{ topic }}?", variables=("topic",)
                )
            )
        assert _warning_logs(records) == []
