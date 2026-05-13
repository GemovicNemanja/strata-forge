"""VCR cassette configuration for provider-touching tests.

``pytest-recording`` reads the ``vcr_config`` fixture below. The default policy
is ``record_mode="none"`` — CI replays committed cassettes and never reaches a
live provider. To record fresh cassettes, set ``RECORD=1`` in the environment
(make target: ``make vcr-record``).

API keys, AWS signatures, and GCP tokens are scrubbed from every cassette via
``before_record_request``. If you ever record without scrubbing, treat the
exposed credentials as leaked and rotate them.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from vcr.request import Request  # pyright: ignore[reportMissingTypeStubs]


# Request headers redacted from cassettes — anything provider auth-bearing.
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
    # AWS SigV4
    "x-amz-security-token",
    "x-amz-date",
    "x-amz-content-sha256",
)

_QUERY_PARAMS_TO_SCRUB: tuple[str, ...] = (
    "api_key",
    "key",
    "access_token",
)

_REDACTED = "[REDACTED]"


def _scrub_request(request: Request) -> Request:
    """vcrpy hook: redact sensitive headers and Authorization-bearing values."""
    for header in _HEADERS_TO_SCRUB:
        if header in request.headers:
            request.headers[header] = _REDACTED
    return request


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, Any]:
    """Default VCR config for cassette-based provider tests.

    - ``record_mode="none"`` unless ``RECORD=1`` (fail rather than hit network)
    - Sensitive headers / query params scrubbed pre-record and ignored on match
    - Cassettes are matched on method + scheme + host + path + body
    """
    return {
        "record_mode": "once" if os.environ.get("RECORD") else "none",
        "filter_headers": [(h, _REDACTED) for h in _HEADERS_TO_SCRUB],
        "filter_query_parameters": [(p, _REDACTED) for p in _QUERY_PARAMS_TO_SCRUB],
        "before_record_request": _scrub_request,
        "match_on": ("method", "scheme", "host", "path", "body"),
        "decode_compressed_response": True,
    }
