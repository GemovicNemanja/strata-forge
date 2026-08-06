"""Cross-module integration tests for `strata_forge.prompts`.

The per-file tests cover each piece in isolation; these wire the whole
module together — registry + store + validation + rendering — to catch
gluing bugs that don't show up in any single file.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from strata_forge.core.errors import ValidationError
from strata_forge.llm.messages import SystemMessage, UserMessage
from strata_forge.prompts.registry import PromptRegistry
from strata_forge.prompts.rendering import render
from strata_forge.prompts.stores.langfuse import LangfusePromptStore
from strata_forge.prompts.stores.memory import InMemoryPromptStore
from strata_forge.prompts.template import PromptTemplate, PromptValidationError

_PERSONA_TEMPLATE = PromptTemplate(
    name="greet",
    stable_section="You are {{ persona }}. Always be concise.",
    dynamic_section="Hi {{ name }}",
    stable_variables=("persona",),
    dynamic_variables=("name",),
    description="Greeter — a stable persona plus a per-user greeting.",
)


# ---------------------------------------------------------------------------
# In-memory store + registry + render
# ---------------------------------------------------------------------------


class TestInMemoryRoundTrip:
    async def test_put_get_render_round_trip(self) -> None:
        registry = PromptRegistry(InMemoryPromptStore())
        version = await registry.put(_PERSONA_TEMPLATE)
        retrieved = await registry.get("greet", version)

        rendered = render(retrieved, {"persona": "helpful", "name": "Alice"})

        assert len(rendered.messages) == 2
        assert isinstance(rendered.messages[0], SystemMessage)
        assert rendered.messages[0].content == "You are helpful. Always be concise."
        assert isinstance(rendered.messages[1], UserMessage)
        assert rendered.messages[1].content == "Hi Alice"

    async def test_latest_returned_when_version_omitted(self) -> None:
        registry = PromptRegistry(InMemoryPromptStore())
        await registry.put(PromptTemplate(name="x", stable_section="first", dynamic_section=""))
        await registry.put(PromptTemplate(name="x", stable_section="second", dynamic_section=""))
        retrieved = await registry.get("x")
        assert retrieved.stable_section == "second"

    async def test_specific_version_returned(self) -> None:
        registry = PromptRegistry(InMemoryPromptStore())
        v1 = await registry.put(
            PromptTemplate(name="x", stable_section="first", dynamic_section="")
        )
        v2 = await registry.put(
            PromptTemplate(name="x", stable_section="second", dynamic_section="")
        )
        first = await registry.get("x", v1)
        second = await registry.get("x", v2)
        assert first.stable_section == "first"
        assert second.stable_section == "second"

    async def test_invalid_template_never_reaches_store(self) -> None:
        store = InMemoryPromptStore()
        registry = PromptRegistry(store)

        bad = PromptTemplate(name="bad", dynamic_section="{{ name }}")
        with pytest.raises(PromptValidationError):
            await registry.put(bad)

        assert await store.list_names() == []

    async def test_render_fails_when_variable_missing(self) -> None:
        registry = PromptRegistry(InMemoryPromptStore())
        version = await registry.put(_PERSONA_TEMPLATE)
        retrieved = await registry.get("greet", version)
        with pytest.raises(ValidationError, match="missing variables"):
            render(retrieved, {"persona": "x"})  # missing 'name'


# ---------------------------------------------------------------------------
# Langfuse store + registry + render
# ---------------------------------------------------------------------------


class TestLangfuseRoundTrip:
    async def test_round_trip_through_mock_langfuse(self) -> None:
        # End-to-end through the Langfuse store: put -> serialize ->
        # roundtrip back -> render. The mock simulates a Langfuse server.
        captured: dict[str, Any] = {}

        def _create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock(version=1)

        def _get(name: str, version: int | None = None) -> Any:
            del version
            prompt = MagicMock()
            prompt.name = name
            prompt.prompt = captured["prompt"]
            prompt.config = captured["config"]
            prompt.version = 1
            return prompt

        mock = MagicMock()
        mock.create_prompt.side_effect = _create
        mock.get_prompt.side_effect = _get

        registry = PromptRegistry(LangfusePromptStore(client=mock))
        version = await registry.put(_PERSONA_TEMPLATE)
        retrieved = await registry.get("greet", version)

        # The deserialized template behaves identically to the original.
        rendered = render(retrieved, {"persona": "helpful", "name": "Alice"})
        assert rendered.messages[0].content == "You are helpful. Always be concise."
        assert rendered.messages[1].content == "Hi Alice"

    async def test_serialized_prompt_carries_forge_flag(self) -> None:
        # Defensive: an external reader of the Langfuse store should be
        # able to tell whether a prompt was written by Forge.
        captured: dict[str, Any] = {}

        def _create(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return MagicMock(version=1)

        mock = MagicMock()
        mock.create_prompt.side_effect = _create

        registry = PromptRegistry(LangfusePromptStore(client=mock))
        await registry.put(_PERSONA_TEMPLATE)

        config = captured["config"]
        assert config["forge_template_v1"] is True
        # The body is JSON — parsing yields the two sections.
        body = json.loads(captured["prompt"])
        assert "stable_section" in body
        assert "dynamic_section" in body


# ---------------------------------------------------------------------------
# Sweep: same template, multiple render variations, stable prefix preserved
# ---------------------------------------------------------------------------


class TestCacheableSweep:
    async def test_many_dynamic_renders_share_stable_digest(self) -> None:
        registry = PromptRegistry(InMemoryPromptStore())
        version = await registry.put(_PERSONA_TEMPLATE)
        retrieved = await registry.get("greet", version)

        # Render with the same persona but many different names. Cache
        # hits depend on the stable prefix being byte-identical, so all
        # five renders must report the same stable digest.
        renders = [
            render(retrieved, {"persona": "helper", "name": name})
            for name in ("Alice", "Bob", "Carol", "Dave", "Eve")
        ]
        digests = {r.cache_hints.stable_digest for r in renders}
        assert len(digests) == 1
