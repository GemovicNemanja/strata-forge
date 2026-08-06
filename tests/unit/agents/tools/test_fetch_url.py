"""Unit tests for `strata_forge.agents.tools.fetch_url`."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from strata_forge.agents.tools.fetch_url import FetchURLArgs, fetch_url
from strata_forge.core.errors import ValidationError


class TestFetchURLArgs:
    def test_valid_https_url_accepted(self) -> None:
        args = FetchURLArgs(url="https://example.com")  # type: ignore[arg-type]
        assert str(args.url) == "https://example.com/"

    def test_invalid_url_rejected(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            FetchURLArgs(url="not a url")  # type: ignore[arg-type]

    def test_defaults(self) -> None:
        args = FetchURLArgs(url="https://example.com")  # type: ignore[arg-type]
        assert args.max_bytes == 100_000
        assert args.timeout_seconds == 10.0

    def test_max_bytes_bounds(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        with pytest.raises(PydanticValidationError):
            FetchURLArgs(url="https://example.com", max_bytes=0)  # type: ignore[arg-type]
        with pytest.raises(PydanticValidationError):
            FetchURLArgs(url="https://example.com", max_bytes=20_000_000)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


class TestFetchURLTool:
    def test_tool_name(self) -> None:
        assert fetch_url.name == "fetch_url"

    def test_tool_parameters_model(self) -> None:
        assert fetch_url.parameters_model is FetchURLArgs


# ---------------------------------------------------------------------------
# HTTP behaviour — mocked via respx
# ---------------------------------------------------------------------------


class TestFetchURLBehaviour:
    @respx.mock
    async def test_basic_fetch_returns_body(self) -> None:
        respx.get("https://example.com/").mock(return_value=Response(200, text="hello world"))
        result = await fetch_url.invoke({"url": "https://example.com"})
        assert result == "hello world"

    @respx.mock
    async def test_body_truncated_to_max_bytes(self) -> None:
        body = "x" * 1000
        respx.get("https://example.com/").mock(return_value=Response(200, text=body))
        result = await fetch_url.invoke({"url": "https://example.com", "max_bytes": 100})
        assert result == "x" * 100

    @respx.mock
    async def test_4xx_raises(self) -> None:
        respx.get("https://example.com/").mock(return_value=Response(404))
        from httpx import HTTPStatusError

        with pytest.raises(HTTPStatusError):
            await fetch_url.invoke({"url": "https://example.com"})

    @respx.mock
    async def test_5xx_raises(self) -> None:
        from httpx import HTTPStatusError

        respx.get("https://example.com/").mock(return_value=Response(500))
        with pytest.raises(HTTPStatusError):
            await fetch_url.invoke({"url": "https://example.com"})

    @respx.mock
    async def test_follows_redirects(self) -> None:
        respx.get("https://example.com/").mock(
            return_value=Response(301, headers={"Location": "https://final.example.com/"})
        )
        respx.get("https://final.example.com/").mock(return_value=Response(200, text="redirected"))
        result = await fetch_url.invoke({"url": "https://example.com"})
        assert result == "redirected"

    async def test_invoke_validates_url(self) -> None:
        with pytest.raises(ValidationError):
            await fetch_url.invoke({"url": "not a url"})
