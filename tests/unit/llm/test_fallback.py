"""Unit tests for `strata_forge.llm.fallback`."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import pytest

from strata_forge.core.errors import (
    FallbackExhaustedError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    RegistryError,
)
from strata_forge.llm.fallback import (
    ModelFallback,
    normalize_fallback_chain,
    run_with_fallback,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from strata_forge.llm.routing import ModelRoute


# ---------------------------------------------------------------------------
# Helpers — a fake `call` that consults a script of per-route reactions.
# ---------------------------------------------------------------------------


def _make_call(
    script: dict[tuple[str, str], BaseException | str],
    *,
    record: list[ModelRoute] | None = None,
) -> Callable[[ModelRoute], Awaitable[str]]:
    """Build a fake `call` that looks up `(model, provider)` in the script.

    A string value is returned as success; an exception value is raised.
    Routes outside the script raise ``KeyError`` so test scripts have to be
    exhaustive (avoids silent mis-routing during tests).
    """

    async def _call(route: ModelRoute) -> str:
        if record is not None:
            record.append(route)
        key = (route.model, route.provider)
        reaction = script[key]
        if isinstance(reaction, BaseException):
            raise reaction
        return reaction

    return _call


class _RetryKwargs(TypedDict, total=False):
    retry_max_attempts: int
    retry_initial_wait: float
    retry_max_wait: float


# Retry kwargs to make tests fast (no real backoff).
_FAST_RETRY: _RetryKwargs = {
    "retry_max_attempts": 1,
    "retry_initial_wait": 0.0,
    "retry_max_wait": 0.0,
}


# ---------------------------------------------------------------------------
# ModelFallback / normalize_fallback_chain
# ---------------------------------------------------------------------------


class TestModelFallback:
    def test_construct(self) -> None:
        f = ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))
        assert f.model == "claude-opus-4-7"
        assert f.providers == ("anthropic", "bedrock")

    def test_default_providers_none(self) -> None:
        f = ModelFallback(model="gpt-5.5")
        assert f.providers is None

    def test_is_frozen(self) -> None:
        f = ModelFallback(model="m")
        with pytest.raises((AttributeError, TypeError)):
            f.model = "other"  # type: ignore[misc]


class TestNormalizeFallbackChain:
    def test_string_becomes_default_route_entry(self) -> None:
        result = normalize_fallback_chain(["gpt-5.5"])
        assert result == [ModelFallback(model="gpt-5.5", providers=None)]

    def test_passthrough_modelfallback(self) -> None:
        original = ModelFallback(model="claude-opus-4-7", providers=("anthropic",))
        result = normalize_fallback_chain([original])
        assert result == [original]

    def test_mixed_entries(self) -> None:
        result = normalize_fallback_chain(
            [
                "gpt-5.5",
                ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock")),
            ]
        )
        assert len(result) == 2
        assert result[0] == ModelFallback(model="gpt-5.5", providers=None)
        assert result[1].providers == ("anthropic", "bedrock")

    def test_empty(self) -> None:
        assert normalize_fallback_chain([]) == []


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestSuccessFirstTry:
    async def test_single_entry_default_route_succeeds(self) -> None:
        # Default route for claude-opus-4-7 is "anthropic" per the registry.
        recorded: list[ModelRoute] = []
        call = _make_call({("claude-opus-4-7", "anthropic"): "ok"}, record=recorded)
        result = await run_with_fallback(
            ["claude-opus-4-7"],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"
        assert len(recorded) == 1
        assert recorded[0].provider == "anthropic"

    async def test_first_provider_succeeds_no_failover(self) -> None:
        recorded: list[ModelRoute] = []
        call = _make_call({("claude-opus-4-7", "anthropic"): "ok"}, record=recorded)
        result = await run_with_fallback(
            [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"
        # Only the first provider was tried.
        assert len(recorded) == 1


# ---------------------------------------------------------------------------
# Provider-level failover (inner loop)
# ---------------------------------------------------------------------------


class TestProviderLevelFailover:
    async def test_rate_limit_advances_to_next_provider(self) -> None:
        recorded: list[ModelRoute] = []
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderRateLimitError(
                    "429", model="claude-opus-4-7", provider="anthropic"
                ),
                ("claude-opus-4-7", "bedrock"): "ok",
            },
            record=recorded,
        )
        result = await run_with_fallback(
            [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"
        assert [r.provider for r in recorded] == ["anthropic", "bedrock"]

    async def test_timeout_advances(self) -> None:
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderTimeoutError("timeout"),
                ("claude-opus-4-7", "bedrock"): "ok",
            }
        )
        result = await run_with_fallback(
            [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"

    async def test_server_error_advances(self) -> None:
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderServerError("500"),
                ("claude-opus-4-7", "bedrock"): "ok",
            }
        )
        result = await run_with_fallback(
            [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"

    async def test_auth_error_advances_to_other_provider(self) -> None:
        # Auth is config-level — different providers may have valid creds.
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderAuthError("no key"),
                ("claude-opus-4-7", "bedrock"): "ok",
            }
        )
        result = await run_with_fallback(
            [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"

    async def test_all_providers_exhausted_within_one_entry(self) -> None:
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderRateLimitError("a"),
                ("claude-opus-4-7", "bedrock"): ProviderServerError("b"),
                ("claude-opus-4-7", "vertex"): ProviderTimeoutError("c"),
            }
        )
        with pytest.raises(FallbackExhaustedError) as info:
            await run_with_fallback(
                [
                    ModelFallback(
                        model="claude-opus-4-7",
                        providers=("anthropic", "bedrock", "vertex"),
                    )
                ],
                call,
                **_FAST_RETRY,
            )
        assert len(info.value.causes) == 3
        providers_attempted = [c[1] for c in info.value.causes]
        assert providers_attempted == ["anthropic", "bedrock", "vertex"]


# ---------------------------------------------------------------------------
# Model-level failover (outer loop)
# ---------------------------------------------------------------------------


class TestModelLevelFailover:
    async def test_first_model_exhausted_drops_to_second(self) -> None:
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderRateLimitError("429"),
                ("gpt-5.5", "openai"): "from gpt",
            }
        )
        result = await run_with_fallback(
            ["claude-opus-4-7", "gpt-5.5"],
            call,
            **_FAST_RETRY,
        )
        assert result == "from gpt"

    async def test_mixed_provider_then_model_failover(self) -> None:
        # Try Anthropic+Bedrock+Vertex for Claude — all fail — then GPT.
        recorded: list[ModelRoute] = []
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderRateLimitError("1"),
                ("claude-opus-4-7", "bedrock"): ProviderServerError("2"),
                ("claude-opus-4-7", "vertex"): ProviderTimeoutError("3"),
                ("gpt-5.5", "openai"): "ok from gpt",
            },
            record=recorded,
        )
        result = await run_with_fallback(
            [
                ModelFallback(
                    model="claude-opus-4-7",
                    providers=("anthropic", "bedrock", "vertex"),
                ),
                ModelFallback(model="gpt-5.5", providers=("openai",)),
            ],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok from gpt"
        # 4 attempts total — 3 Claude provider-level + 1 GPT.
        assert len(recorded) == 4
        assert recorded[-1].model == "gpt-5.5"

    async def test_every_model_exhausted_raises(self) -> None:
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderRateLimitError("1"),
                ("gpt-5.5", "openai"): ProviderServerError("2"),
                ("gemini-3.1-pro", "vertex"): ProviderTimeoutError("3"),
            }
        )
        with pytest.raises(FallbackExhaustedError) as info:
            await run_with_fallback(
                ["claude-opus-4-7", "gpt-5.5", "gemini-3.1-pro"],
                call,
                **_FAST_RETRY,
            )
        causes = info.value.causes
        assert len(causes) == 3
        # Causes are in chain order.
        assert causes[0][0] == "claude-opus-4-7"
        assert causes[1][0] == "gpt-5.5"
        assert causes[2][0] == "gemini-3.1-pro"


# ---------------------------------------------------------------------------
# ContentFilter short-circuit
# ---------------------------------------------------------------------------


class TestContentFilterShortCircuit:
    async def test_content_filter_propagates_immediately(self) -> None:
        recorded: list[ModelRoute] = []
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderContentFilterError(
                    "blocked", model="claude-opus-4-7", provider="anthropic"
                ),
                # These would succeed if we ever reached them — but we shouldn't.
                ("claude-opus-4-7", "bedrock"): "would succeed",
                ("gpt-5.5", "openai"): "would also succeed",
            },
            record=recorded,
        )
        with pytest.raises(ProviderContentFilterError, match="blocked"):
            await run_with_fallback(
                [
                    ModelFallback(
                        model="claude-opus-4-7",
                        providers=("anthropic", "bedrock"),
                    ),
                    "gpt-5.5",
                ],
                call,
                **_FAST_RETRY,
            )
        # Only the first provider was attempted; everything else was skipped.
        assert len(recorded) == 1
        assert recorded[0].provider == "anthropic"


# ---------------------------------------------------------------------------
# BadRequest handling (default vs strict)
# ---------------------------------------------------------------------------


class TestBadRequestHandling:
    async def test_default_advances_on_bad_request(self) -> None:
        # Some providers reject params others accept — default = advance.
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderBadRequestError("400"),
                ("claude-opus-4-7", "bedrock"): "ok",
            }
        )
        result = await run_with_fallback(
            [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"

    async def test_strict_mode_aborts_on_bad_request(self) -> None:
        recorded: list[ModelRoute] = []
        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): ProviderBadRequestError("400"),
                ("claude-opus-4-7", "bedrock"): "would succeed",
            },
            record=recorded,
        )
        with pytest.raises(FallbackExhaustedError) as info:
            await run_with_fallback(
                [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
                call,
                strict_bad_request=True,
                **_FAST_RETRY,
            )
        # Aborted after first failure; only one attempt.
        assert len(recorded) == 1
        assert len(info.value.causes) == 1
        assert isinstance(info.value.causes[0][2], ProviderBadRequestError)

    async def test_strict_mode_carries_bad_request_as_cause(self) -> None:
        bad_req = ProviderBadRequestError(
            "bad params", model="claude-opus-4-7", provider="anthropic"
        )
        call = _make_call({("claude-opus-4-7", "anthropic"): bad_req})
        with pytest.raises(FallbackExhaustedError) as info:
            await run_with_fallback(
                ["claude-opus-4-7"],
                call,
                strict_bad_request=True,
                **_FAST_RETRY,
            )
        # The __cause__ of the FallbackExhaustedError should be the bad_req.
        assert info.value.__cause__ is bad_req


# ---------------------------------------------------------------------------
# RegistryError handling
# ---------------------------------------------------------------------------


class TestRegistryErrorHandling:
    async def test_unknown_model_recorded_and_advances(self) -> None:
        # "unknown-model" doesn't exist; resolve() raises RegistryError. The
        # runner records the cause and moves on to the next chain entry.
        call = _make_call({("claude-opus-4-7", "anthropic"): "ok"})
        result = await run_with_fallback(
            ["unknown-model", "claude-opus-4-7"],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"

    async def test_unsupported_route_recorded_and_advances(self) -> None:
        # Gemini doesn't have a Bedrock route; resolve() raises
        # RegistryError(reason="unsupported_route").
        call = _make_call({("gemini-3.1-pro", "vertex"): "ok from vertex"})
        result = await run_with_fallback(
            [
                ModelFallback(model="gemini-3.1-pro", providers=("bedrock", "vertex")),
            ],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok from vertex"

    async def test_only_unknown_models_raises_exhausted(self) -> None:
        call = _make_call({})  # never called
        with pytest.raises(FallbackExhaustedError) as info:
            await run_with_fallback(
                ["unknown-1", "unknown-2"],
                call,
                **_FAST_RETRY,
            )
        assert len(info.value.causes) == 2
        for _, _, err in info.value.causes:
            assert isinstance(err, RegistryError)


# ---------------------------------------------------------------------------
# Retry integration
# ---------------------------------------------------------------------------


class TestRetryIntegration:
    async def test_transient_error_retries_before_advancing(self) -> None:
        # First two calls fail with a retryable error; third succeeds. With
        # retry_max_attempts=3, the same provider should succeed on attempt
        # 3 without ever advancing to the next provider.
        attempts: list[ModelRoute] = []
        sequence = [
            ProviderRateLimitError("429"),
            ProviderRateLimitError("429"),
            "finally ok",
        ]

        async def _call(route: ModelRoute) -> str:
            attempts.append(route)
            reaction = sequence.pop(0)
            if isinstance(reaction, BaseException):
                raise reaction
            return reaction

        result = await run_with_fallback(
            [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            _call,
            retry_max_attempts=3,
            retry_initial_wait=0.0,
            retry_max_wait=0.0,
        )
        assert result == "finally ok"
        # All three attempts on Anthropic — never crossed over to Bedrock.
        assert [r.provider for r in attempts] == ["anthropic", "anthropic", "anthropic"]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_empty_chain_raises_immediately(self) -> None:
        call = _make_call({})
        with pytest.raises(FallbackExhaustedError) as info:
            await run_with_fallback([], call, **_FAST_RETRY)
        assert info.value.causes == []

    async def test_non_provider_exception_propagates(self) -> None:
        # ValueError isn't a ProviderError — it should bubble out unchanged.
        call = _make_call({("claude-opus-4-7", "anthropic"): ValueError("programmer error")})
        with pytest.raises(ValueError, match="programmer error"):
            await run_with_fallback(["claude-opus-4-7"], call, **_FAST_RETRY)

    async def test_future_provider_error_subclass_advances(self) -> None:
        # An unknown ProviderError subclass should be treated conservatively
        # as "advance" (catch-all in the runner).
        class _NewProviderError(ProviderError):
            pass

        call = _make_call(
            {
                ("claude-opus-4-7", "anthropic"): _NewProviderError("new"),
                ("claude-opus-4-7", "bedrock"): "ok",
            }
        )
        result = await run_with_fallback(
            [ModelFallback(model="claude-opus-4-7", providers=("anthropic", "bedrock"))],
            call,
            **_FAST_RETRY,
        )
        assert result == "ok"

    async def test_causes_carry_route_annotation(self) -> None:
        call = _make_call({("claude-opus-4-7", "anthropic"): ProviderRateLimitError("429")})
        with pytest.raises(FallbackExhaustedError) as info:
            await run_with_fallback(["claude-opus-4-7"], call, **_FAST_RETRY)
        assert info.value.causes[0][0] == "claude-opus-4-7"
        assert info.value.causes[0][1] == "anthropic"
        assert isinstance(info.value.causes[0][2], ProviderRateLimitError)
