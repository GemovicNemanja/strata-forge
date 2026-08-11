"""LiteLLM exception → ``ProviderError`` mapping.

Every provider call inside ``strata_forge.llm`` funnels through this seam so callers
above the module never see raw LiteLLM exceptions. The mapping is in
**most-specific-first** order: classes that inherit from ``BadRequestError``
(content-policy, context-window-exceeded, image-fetch, etc.) are handled
before the ``BadRequestError`` catch-all, otherwise the more specific
semantics would be lost.

Per-provider quirks worth keeping in mind:

- **OpenAI** raises rich subclasses (``AuthenticationError``,
  ``RateLimitError``, ``PermissionDeniedError``) with HTTP-style status codes.
- **Anthropic** uses similar subclasses; ``Overloaded`` surfaces as
  ``InternalServerError`` (HTTP 529 → maps to ``ProviderServerError``).
- **AWS Bedrock** often surfaces auth as ``AuthenticationError`` even when the
  underlying problem is an IAM/SigV4 mismatch; the ``ProviderAuthError``
  message preserves the LiteLLM string so operators can trace back.
- **GCP Vertex** typically uses ``APIConnectionError`` for credential issues
  (because the SDK fails to refresh OAuth tokens before HTTP); we map that to
  ``ProviderTimeoutError`` — the retry policy is identical regardless.
- **Azure OpenAI** is wire-compatible with OpenAI, so exceptions look identical.
- **OpenAI-compatible servers** (vLLM/TGI/SGLang) inherit OpenAI shapes; some
  return raw HTTP errors that LiteLLM surfaces as bare ``APIError``.
"""

from __future__ import annotations

from typing import NoReturn

import litellm.exceptions as litellm_exc
import openai

from strata_forge.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)

__all__ = ["map_litellm_exception", "raise_as_provider_error"]


def _classes(*names: str) -> tuple[type[BaseException], ...]:
    """Resolve LiteLLM exception classes by NAME, skipping any this LiteLLM does not define.

    ``litellm>=1.55`` is a floor with no ceiling, so this module runs against versions that
    predate classes it would like to match. Naming one directly costs an ``AttributeError`` —
    raised from inside the mapper, which only runs when something has ALREADY failed. The
    original diagnosis is then replaced by an unrelated one about a missing attribute, and every
    call looks like the same bug regardless of what actually went wrong. That is far worse than
    declining to classify: a batch run reported 2098 identical
    ``AttributeError: module 'litellm.exceptions' has no attribute ...`` and the real errors
    behind them were never recorded anywhere.

    Resolving by name at import turns a version difference into a class this build simply cannot
    match — the fallback then applies, and the caller still learns what went wrong.
    """
    found: list[type[BaseException]] = []
    for name in names:
        candidate = getattr(litellm_exc, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            found.append(candidate)
    return tuple(found)


# Matched FIRST: several of these inherit from BadRequestError, so the broader group below would
# otherwise swallow the more specific semantics.
_CONTENT_FILTER = _classes(
    "ContentPolicyViolationError",
    "BlockedPiiEntityError",
    "RejectedRequestError",
    "GuardrailRaisedException",
    "GuardrailInterventionNormalStringError",
)
_AUTH = _classes("AuthenticationError", "PermissionDeniedError")
_RATE_LIMIT = _classes("RateLimitError")
_TIMEOUT = _classes("Timeout", "APIConnectionError")
_BAD_REQUEST = _classes(
    "BadRequestError",
    "UnprocessableEntityError",
    "NotFoundError",
    "APIResponseValidationError",
)
_SERVER = _classes(
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "APIError",
    "OpenAIError",
)
_BUDGET = _classes("BudgetExceededError")


def map_litellm_exception(
    exc: BaseException,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> ProviderError:
    """Translate a LiteLLM exception into a ``ProviderError`` subclass.

    Args:
        exc: The exception raised from a LiteLLM call.
        model: Logical Forge model name to annotate the resulting error with.
        provider: Provider route name to annotate the resulting error with.

    Returns:
        A ``ProviderError`` subclass instance. Never raises — for any input
        that isn't a recognized LiteLLM class, a generic ``ProviderError`` is
        returned with a descriptive message so callers can still re-raise it
        through their normal control flow.
    """
    # Some LiteLLM exception __str__ methods assume specific attributes set by
    # __init__; if a caller hands us a partially-constructed or unusual subclass,
    # fall back to the class name rather than letting str() raise.
    try:
        rendered = str(exc)
    except Exception:
        rendered = ""
    message = rendered or type(exc).__name__
    raw_status = getattr(exc, "status_code", None)
    status_code: int | None = raw_status if isinstance(raw_status, int) else None

    # --- Content policy / guardrails ---------------------------------------
    # Several of these inherit from BadRequestError, so they must be matched
    # first to preserve the more specific semantics.
    if isinstance(exc, _CONTENT_FILTER):
        return ProviderContentFilterError(
            message, model=model, provider=provider, status_code=status_code
        )

    # --- Auth / permission --------------------------------------------------
    if isinstance(exc, _AUTH):
        return ProviderAuthError(message, model=model, provider=provider, status_code=status_code)

    # --- Rate limit ---------------------------------------------------------
    if isinstance(exc, _RATE_LIMIT):
        return ProviderRateLimitError(
            message, model=model, provider=provider, status_code=status_code
        )

    # --- Timeout / network --------------------------------------------------
    if isinstance(exc, _TIMEOUT):
        return ProviderTimeoutError(
            message, model=model, provider=provider, status_code=status_code
        )

    # --- Bad request and its remaining specializations ----------------------
    # (Content / auth subclasses of BadRequestError were already handled above.)
    # Note: some LiteLLM classes (e.g. InvalidRequestError) inherit directly
    # from openai.BadRequestError without going through litellm.BadRequestError,
    # so we check against both bases.
    if isinstance(exc, (*_BAD_REQUEST, openai.BadRequestError)):
        return ProviderBadRequestError(
            message, model=model, provider=provider, status_code=status_code
        )

    # --- Server errors (5xx) ------------------------------------------------
    # MidStreamFallbackError inherits from ServiceUnavailableError; it's
    # caught here for free.
    if isinstance(exc, _SERVER):
        return ProviderServerError(message, model=model, provider=provider, status_code=status_code)

    # --- LiteLLM's own budget tracker --------------------------------------
    # Name-collides with strata_forge.core.errors.BudgetExceededError but is a
    # different concept (LiteLLM proxy budget vs. Forge BudgetContext).
    # Treat as a generic provider rejection.
    if isinstance(exc, _BUDGET):
        return ProviderError(
            f"LiteLLM budget tracker rejected the call: {message}",
            model=model,
            provider=provider,
            status_code=status_code,
        )

    # --- Fallback for unrecognized classes ---------------------------------
    return ProviderError(
        f"Unmapped LiteLLM exception {type(exc).__name__}: {message}",
        model=model,
        provider=provider,
        status_code=status_code,
    )


def raise_as_provider_error(
    exc: BaseException,
    *,
    model: str | None = None,
    provider: str | None = None,
) -> NoReturn:
    """Map ``exc`` to a ``ProviderError`` and re-raise it, preserving ``__cause__``.

    Convenience helper for the common ``try / except`` block at the seam:

    .. code-block:: python

        try:
            return await litellm.acompletion(...)
        except Exception as exc:
            raise_as_provider_error(exc, model=route.model, provider=route.provider)
    """
    raise map_litellm_exception(exc, model=model, provider=provider) from exc
