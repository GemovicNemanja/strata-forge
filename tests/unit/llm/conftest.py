"""Fixtures shared by the `strata_forge.llm` unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _model_map() -> dict[str, dict[str, Any]]:
    """LiteLLM's model map, typed."""
    import litellm

    return cast("dict[str, dict[str, Any]]", litellm.model_cost)  # pyright: ignore[reportUnknownMemberType]


@pytest.fixture
def litellm_map_without() -> Iterator[Callable[..., None]]:
    """Hide model ids from LiteLLM's model map, as its bundled map hides unreleased models.

    Calling the fixture with ids drops every map key whose last path segment is one of them.
    The map, LiteLLM's OpenAI model set and its lookup caches are restored afterwards.
    """
    import litellm
    from litellm.utils import (
        _invalidate_model_cost_lowercase_map,  # pyright: ignore[reportPrivateUsage]
    )

    saved_map = _model_map()
    openai_models = cast("set[str]", litellm.open_ai_chat_completion_models)  # pyright: ignore[reportUnknownMemberType]
    saved_openai = set(openai_models)
    litellm.model_cost = dict(saved_map)

    def _without(*model_ids: str) -> None:
        current = _model_map()
        for key in [k for k in current if k.rsplit("/", 1)[-1] in model_ids]:
            del current[key]
        _invalidate_model_cost_lowercase_map()

    yield _without
    litellm.model_cost = saved_map
    openai_models.clear()
    openai_models.update(saved_openai)
    _invalidate_model_cost_lowercase_map()
