"""LiteLLM exception → ``ProviderError`` mapping.

Every provider call inside ``forge.llm`` funnels through this seam so callers
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

from forge.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)

__all__ = ["map_litellm_exception", "raise_as_provider_error"]


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
    if isinstance(
        exc,
        litellm_exc.ContentPolicyViolationError
        | litellm_exc.BlockedPiiEntityError
        | litellm_exc.RejectedRequestError
        | litellm_exc.GuardrailRaisedException
        | litellm_exc.GuardrailInterventionNormalStringError,
    ):
        return ProviderContentFilterError(
            message, model=model, provider=provider, status_code=status_code
        )

    # --- Auth / permission --------------------------------------------------
    if isinstance(exc, litellm_exc.AuthenticationError | litellm_exc.PermissionDeniedError):
        return ProviderAuthError(message, model=model, provider=provider, status_code=status_code)

    # --- Rate limit ---------------------------------------------------------
    if isinstance(exc, litellm_exc.RateLimitError):
        return ProviderRateLimitError(
            message, model=model, provider=provider, status_code=status_code
        )

    # --- Timeout / network --------------------------------------------------
    if isinstance(exc, litellm_exc.Timeout | litellm_exc.APIConnectionError):
        return ProviderTimeoutError(
            message, model=model, provider=provider, status_code=status_code
        )

    # --- Bad request and its remaining specializations ----------------------
    # (Content / auth subclasses of BadRequestError were already handled above.)
    # Note: some LiteLLM classes (e.g. InvalidRequestError) inherit directly
    # from openai.BadRequestError without going through litellm.BadRequestError,
    # so we check against both bases.
    if isinstance(
        exc,
        litellm_exc.BadRequestError
        | openai.BadRequestError
        | litellm_exc.UnprocessableEntityError
        | litellm_exc.NotFoundError
        | litellm_exc.APIResponseValidationError,
    ):
        return ProviderBadRequestError(
            message, model=model, provider=provider, status_code=status_code
        )

    # --- Server errors (5xx) ------------------------------------------------
    # MidStreamFallbackError inherits from ServiceUnavailableError; it's
    # caught here for free.
    if isinstance(
        exc,
        litellm_exc.InternalServerError
        | litellm_exc.ServiceUnavailableError
        | litellm_exc.BadGatewayError
        | litellm_exc.APIError
        | litellm_exc.OpenAIError,
    ):
        return ProviderServerError(message, model=model, provider=provider, status_code=status_code)

    # --- LiteLLM's own budget tracker --------------------------------------
    # Name-collides with forge.core.errors.BudgetExceededError but is a
    # different concept (LiteLLM proxy budget vs. Forge BudgetContext).
    # Treat as a generic provider rejection.
    if isinstance(exc, litellm_exc.BudgetExceededError):
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
