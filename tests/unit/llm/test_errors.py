"""Unit tests for `strata_forge.llm.errors` — LiteLLM exception → ProviderError mapping."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    BlockedPiiEntityError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    GuardrailInterventionNormalStringError,
    GuardrailRaisedException,
    ImageFetchError,
    InternalServerError,
    InvalidRequestError,
    JSONSchemaValidationError,
    LiteLLMUnknownProvider,
    MidStreamFallbackError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RejectedRequestError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
    UnsupportedParamsError,
)
from litellm.exceptions import BudgetExceededError as LiteLLMBudgetExceededError

from strata_forge.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from strata_forge.llm.errors import map_litellm_exception, raise_as_provider_error


def _make_response(status: int = 500) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        request=httpx.Request("POST", "https://example.com"),
    )


def _make(cls: type[BaseException], **extras: Any) -> BaseException:
    """Construct a LiteLLM exception with sensible test defaults.

    Different LiteLLM classes take different positional orders for the same
    fields (some are ``(message, model, llm_provider)``, others
    ``(message, llm_provider, model)``); using kwargs side-steps that.
    A handful require a ``response`` argument and fall through to a second try.
    The last resort bypasses ``__init__`` entirely.
    """
    base: dict[str, Any] = {
        "message": "test error",
        "model": "claude-opus-4-7",
        "llm_provider": "anthropic",
    }
    base.update(extras)
    try:
        return cls(**base)
    except TypeError:
        pass
    try:
        return cls(**base, response=_make_response())  # pyright: ignore[reportCallIssue]
    except TypeError:
        pass
    instance = cls.__new__(cls)
    instance.args = (base["message"],)
    return instance


# Each row: (LiteLLM exception class, expected ProviderError subclass).
# Order in the file is by Forge category, but pytest runs them all
# independently via parametrize.
MAPPING_TABLE: tuple[tuple[type[BaseException], type[ProviderError]], ...] = (
    # Content / guardrails — must precede BadRequest in the impl because
    # several inherit from it.
    (ContentPolicyViolationError, ProviderContentFilterError),
    (BlockedPiiEntityError, ProviderContentFilterError),
    (RejectedRequestError, ProviderContentFilterError),
    (GuardrailRaisedException, ProviderContentFilterError),
    (GuardrailInterventionNormalStringError, ProviderContentFilterError),
    # Auth
    (AuthenticationError, ProviderAuthError),
    (PermissionDeniedError, ProviderAuthError),
    # Rate limit
    (RateLimitError, ProviderRateLimitError),
    # Timeout / network
    (Timeout, ProviderTimeoutError),
    (APIConnectionError, ProviderTimeoutError),
    # Bad request and its remaining specializations
    (BadRequestError, ProviderBadRequestError),
    (ContextWindowExceededError, ProviderBadRequestError),
    (ImageFetchError, ProviderBadRequestError),
    (LiteLLMUnknownProvider, ProviderBadRequestError),
    (UnsupportedParamsError, ProviderBadRequestError),
    (InvalidRequestError, ProviderBadRequestError),
    (UnprocessableEntityError, ProviderBadRequestError),
    (NotFoundError, ProviderBadRequestError),
    (JSONSchemaValidationError, ProviderBadRequestError),
    # Server errors (MidStream inherits from ServiceUnavailable)
    (InternalServerError, ProviderServerError),
    (ServiceUnavailableError, ProviderServerError),
    (BadGatewayError, ProviderServerError),
    (MidStreamFallbackError, ProviderServerError),
)


class TestMapping:
    @pytest.mark.parametrize(("litellm_cls", "expected_cls"), MAPPING_TABLE)
    def test_maps_to_expected_class(
        self,
        litellm_cls: type[BaseException],
        expected_cls: type[ProviderError],
    ) -> None:
        exc = _make(litellm_cls)
        result = map_litellm_exception(exc)
        assert isinstance(result, expected_cls), (
            f"{litellm_cls.__name__} → expected {expected_cls.__name__}, "
            f"got {type(result).__name__}"
        )

    def test_api_error_maps_to_server(self) -> None:
        # APIError has a different __init__: (status_code, message, ...).
        exc = APIError(
            status_code=500,
            message="server kaboom",
            llm_provider="anthropic",
            model="claude-opus-4-7",
        )
        result = map_litellm_exception(exc)
        assert isinstance(result, ProviderServerError)

    def test_litellm_budget_maps_to_generic_provider_error(self) -> None:
        # LiteLLM's BudgetExceededError name-collides with Forge's
        # core.errors.BudgetExceededError but represents a different concept
        # (LiteLLM proxy budget). Must surface as a generic ProviderError.
        exc = LiteLLMBudgetExceededError(current_cost=10.0, max_budget=5.0)
        result = map_litellm_exception(exc)
        assert type(result) is ProviderError
        assert "LiteLLM budget" in str(result)

    def test_unrecognized_falls_back_to_generic_provider_error(self) -> None:
        class CustomError(Exception):
            pass

        result = map_litellm_exception(CustomError("strange shape"))
        assert type(result) is ProviderError
        assert "Unmapped" in str(result)
        assert "CustomError" in str(result)
        assert "strange shape" in str(result)

    def test_empty_message_falls_back_to_class_name(self) -> None:
        class CustomError(Exception):
            pass

        result = map_litellm_exception(CustomError())
        # When str(exc) is empty, the message defaults to the class name.
        assert "CustomError" in str(result)


class TestRouteAnnotation:
    def test_model_and_provider_preserved(self) -> None:
        exc = _make(RateLimitError)
        result = map_litellm_exception(exc, model="claude-opus-4-7", provider="anthropic")
        assert result.model == "claude-opus-4-7"
        assert result.provider == "anthropic"

    def test_model_and_provider_default_to_none(self) -> None:
        exc = _make(RateLimitError)
        result = map_litellm_exception(exc)
        assert result.model is None
        assert result.provider is None

    def test_status_code_preserved_when_present(self) -> None:
        exc = APIError(
            status_code=503,
            message="boom",
            llm_provider="anthropic",
            model="claude-opus-4-7",
        )
        result = map_litellm_exception(exc)
        assert result.status_code == 503

    def test_status_code_none_when_absent(self) -> None:
        class BareError(Exception):
            pass

        result = map_litellm_exception(BareError("plain"))
        assert result.status_code is None

    def test_non_int_status_code_treated_as_none(self) -> None:
        class WeirdError(Exception):
            status_code = "not-an-int"

        result = map_litellm_exception(WeirdError("plain"))
        assert result.status_code is None


class TestRaiseAsProviderError:
    def test_chains_via_from(self) -> None:
        original = _make(RateLimitError)
        with pytest.raises(ProviderRateLimitError) as excinfo:
            raise_as_provider_error(original, model="m", provider="p")
        assert excinfo.value.__cause__ is original
        assert excinfo.value.model == "m"
        assert excinfo.value.provider == "p"

    def test_unmapped_raises_provider_error(self) -> None:
        class CustomError(Exception):
            pass

        original = CustomError("oops")
        with pytest.raises(ProviderError) as excinfo:
            raise_as_provider_error(original)
        # NOT a ProviderError subclass — the exact base type.
        assert type(excinfo.value) is ProviderError
        assert excinfo.value.__cause__ is original
