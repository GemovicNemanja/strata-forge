"""Provider x scenario VCR replay tests.

Each test plays back a previously-recorded HTTP exchange against the
real LiteLLM dispatch path. When the matching cassette is missing,
the test skips with a clear reason; CI passes on a fresh checkout
with no cassettes committed. See ``tests/vcr/README.md`` for recording.

Cassettes live in ``tests/vcr/cassettes/<provider>/<scenario>.yaml``
(or ``_cross/<scenario>.yaml`` for cross-provider scenarios). Each test
specifies the exact cassette path via :func:`_cassette_for`, so the
naming stays explicit and discoverable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from strata_forge.llm.client import LLMClient
from strata_forge.llm.fallback import ModelFallback
from strata_forge.llm.messages import Message
from strata_forge.llm.tools import tool
from vcr import VCR  # pyright: ignore[reportMissingTypeStubs]

if TYPE_CHECKING:
    from strata_forge.llm.registry import ProviderName


_CASSETTE_ROOT = Path(__file__).parent / "cassettes"


# Mirror the scrubbing list from conftest.py so the standalone vcrpy
# instance below redacts the same headers/params on record.
_HEADERS_TO_SCRUB: tuple[str, ...] = (
    "authorization",
    "x-api-key",
    "x-goog-api-key",
    "openai-organization",
    "openai-project",
    "anthropic-api-key",
    "anthropic-version",
    "azure-openai-api-key",
    "api-key",
    "x-amz-security-token",
    "x-amz-date",
    "x-amz-content-sha256",
)
_QUERY_PARAMS_TO_SCRUB: tuple[str, ...] = ("api_key", "key", "access_token")
_REDACTED = "[REDACTED]"


def _vcr() -> Any:
    """Build a VCR instance with the same scrubbing policy as the conftest fixture.

    Using vcrpy directly (rather than pytest-recording's marker) lets us
    parametrize the cassette path per (provider, scenario) without
    fighting the marker's decoration-time path resolution. Returns ``Any``
    because vcrpy doesn't ship type stubs; callers treat the result
    duck-typed.
    """
    record_mode: Any = "once" if os.environ.get("RECORD") else "none"
    return VCR(
        record_mode=record_mode,
        filter_headers=[(h, _REDACTED) for h in _HEADERS_TO_SCRUB],
        filter_query_parameters=[(p, _REDACTED) for p in _QUERY_PARAMS_TO_SCRUB],
        match_on=("method", "scheme", "host", "path", "body"),
        decode_compressed_response=True,
    )


def _cassette_for(provider: str, scenario: str) -> Path:
    """Return the conventional cassette path for ``(provider, scenario)``."""
    return _CASSETTE_ROOT / provider / f"{scenario}.yaml"


def _requires_cassette(path: Path) -> None:
    """Skip the test cleanly when ``path`` doesn't exist and we're not recording.

    Recording (``RECORD=1``) bypasses the check — the cassette will be
    created by VCR on the first successful live request.
    """
    if os.environ.get("RECORD"):
        return
    if not path.exists():
        pytest.skip(f"No cassette: {path.relative_to(_CASSETTE_ROOT.parent.parent)}")


# Each registry-resident model paired with the route to drive through it for
# the cassette suite. Update when models or default models change.
_PER_PROVIDER_MODEL: dict[ProviderName, str] = {
    "openai": "gpt-5.5",
    "anthropic": "claude-opus-4-7",
    "vertex": "gemini-3.1-pro",
    "bedrock": "claude-opus-4-7",
    "azure": "gpt-5.5",
    "openai_compat": "gpt-5.5",
}

_ALL_PROVIDERS: tuple[ProviderName, ...] = tuple(_PER_PROVIDER_MODEL.keys())

# Providers + models with native image support.
_VISION_PROVIDERS: tuple[ProviderName, ...] = ("openai", "anthropic", "vertex", "bedrock")


# ---------------------------------------------------------------------------
# Sample types for the tool / structured scenarios
# ---------------------------------------------------------------------------


class _WeatherArgs(BaseModel):
    location: str


@tool
async def _get_weather(args: _WeatherArgs) -> str:
    """Get the current weather for a city."""
    return f"sunny in {args.location}"


class _Summary(BaseModel):
    title: str
    bullets: list[str]


# ---------------------------------------------------------------------------
# Per-provider scenario tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("provider", _ALL_PROVIDERS)
async def test_basic_completion(provider: ProviderName) -> None:
    cassette = _cassette_for(provider, "basic_completion")
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(_PER_PROVIDER_MODEL[provider], provider=provider)
        resp = await client.complete([Message.user("Say hello in one word.")])
        assert resp.text
        assert resp.usage.output_tokens > 0
        assert resp.cost_usd >= 0


@pytest.mark.parametrize("provider", _ALL_PROVIDERS)
async def test_streaming(provider: ProviderName) -> None:
    cassette = _cassette_for(provider, "streaming")
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(_PER_PROVIDER_MODEL[provider], provider=provider)
        chunks_text: list[str] = []
        async for chunk in await client.stream([Message.user("Count to three.")]):
            chunks_text.append(chunk.delta_text)
        assert "".join(chunks_text).strip()


@pytest.mark.parametrize("provider", _ALL_PROVIDERS)
async def test_structured_output(provider: ProviderName) -> None:
    cassette = _cassette_for(provider, "structured_output")
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(_PER_PROVIDER_MODEL[provider], provider=provider)
        resp = await client.complete_structured(
            [Message.user("Summarize: water is wet.")],
            schema=_Summary,
        )
        assert resp.parsed.title


@pytest.mark.parametrize("provider", _VISION_PROVIDERS)
async def test_multimodal(provider: ProviderName) -> None:
    from strata_forge.llm.messages import TextPart, UserMessage
    from strata_forge.llm.multimodal import ImageContent

    cassette = _cassette_for(provider, "multimodal")
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(_PER_PROVIDER_MODEL[provider], provider=provider)
        resp = await client.complete(
            [
                UserMessage(
                    content=[
                        TextPart(text="What color is the sky in this image?"),
                        ImageContent.from_url(
                            "https://upload.wikimedia.org/wikipedia/commons/0/0a/Sky_pictures.jpg"
                        ),
                    ]
                )
            ]
        )
        assert resp.text


@pytest.mark.parametrize("provider", _ALL_PROVIDERS)
async def test_single_tool_call(provider: ProviderName) -> None:
    cassette = _cassette_for(provider, "single_tool_call")
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(_PER_PROVIDER_MODEL[provider], provider=provider)
        resp = await client.complete(
            [Message.user("What's the weather in Tokyo right now?")],
            tools=[_get_weather],
        )
        # Either the model emitted a tool call or it answered directly — we
        # just assert the call didn't crash and capability gating worked.
        assert resp.finish_reason in {"tool_use", "stop", "length"}


@pytest.mark.parametrize("provider", _ALL_PROVIDERS)
async def test_multi_turn_tool_loop(provider: ProviderName) -> None:
    cassette = _cassette_for(provider, "multi_turn_tool_loop")
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(_PER_PROVIDER_MODEL[provider], provider=provider)
        resp = await client.run_tool_loop(
            [Message.user("Look up the weather in Tokyo then explain.")],
            tools=[_get_weather],
            max_iterations=4,
        )
        assert resp.finish_reason != "tool_use"


@pytest.mark.parametrize("provider", _ALL_PROVIDERS)
async def test_rate_limit_retry(provider: ProviderName) -> None:
    """The recorded cassette has a 429 followed by a 200 — `@retry` succeeds."""
    cassette = _cassette_for(provider, "rate_limit_retry")
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(
            _PER_PROVIDER_MODEL[provider],
            provider=provider,
            retry_max_attempts=3,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        resp = await client.complete([Message.user("hi")])
        assert resp.text


@pytest.mark.parametrize("provider", _ALL_PROVIDERS)
async def test_content_filter(provider: ProviderName) -> None:
    """The recorded request triggers a content-filter response."""
    from strata_forge.core.errors import ProviderContentFilterError

    cassette = _cassette_for(provider, "content_filter")
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(_PER_PROVIDER_MODEL[provider], provider=provider)
        with pytest.raises(ProviderContentFilterError):
            await client.complete([Message.user("[recorded request that hits the content filter]")])


# ---------------------------------------------------------------------------
# Cross-provider scenarios
# ---------------------------------------------------------------------------


async def test_provider_level_fallthrough() -> None:
    """Anthropic 429s -> Bedrock serves Claude. Same logical model both routes."""
    cassette = _CASSETTE_ROOT / "_cross" / "provider_fallthrough.yaml"
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient(
            chain=[
                ModelFallback(
                    model="claude-opus-4-7",
                    providers=("anthropic", "bedrock"),
                ),
            ],
            retry_max_attempts=1,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        resp = await client.complete([Message.user("hi")])
        assert resp.route.provider in {"anthropic", "bedrock"}


async def test_model_level_fallthrough() -> None:
    """Claude exhausted -> GPT-5.5 picks up."""
    cassette = _CASSETTE_ROOT / "_cross" / "model_fallthrough.yaml"
    _requires_cassette(cassette)
    with _vcr().use_cassette(str(cassette)):
        client = LLMClient.with_fallbacks(
            ["claude-opus-4-7", "gpt-5.5"],
            retry_max_attempts=1,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        resp = await client.complete([Message.user("hi")])
        assert resp.text
