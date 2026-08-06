"""HTTP fetch tool — GET a URL and return the body text.

Uses ``httpx.AsyncClient`` for the request. ``httpx`` is a transitive
dependency of ``litellm`` (already in core deps) so this tool is
available without any optional extra.

The tool truncates the response body to ``max_bytes`` to keep model
context bounded; callers that need full payloads should bypass the
agent layer and call ``httpx`` directly.
"""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field, HttpUrl

from strata_forge.llm.tools import tool

__all__ = [
    "FetchURLArgs",
    "fetch_url",
]


class FetchURLArgs(BaseModel):
    """Arguments for the :data:`fetch_url` tool."""

    url: HttpUrl = Field(description="The HTTP/HTTPS URL to fetch.")
    max_bytes: int = Field(
        default=100_000,
        ge=1,
        le=10_000_000,
        description=(
            "Cap on the returned body length in characters. Excess is "
            "truncated. Default 100 KB; max 10 MB."
        ),
    )
    timeout_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=120.0,
        description="Per-request timeout in seconds. Default 10, max 120.",
    )


@tool
async def fetch_url(args: FetchURLArgs) -> str:
    """Fetch a URL via HTTP GET and return the response body text (truncated)."""
    async with httpx.AsyncClient(
        timeout=args.timeout_seconds,
        follow_redirects=True,
    ) as client:
        response = await client.get(str(args.url))
        response.raise_for_status()
        return response.text[: args.max_bytes]
