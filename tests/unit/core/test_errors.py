"""Unit tests for `forge.core.errors`."""

from __future__ import annotations

import pytest

from forge.core.errors import (
    BudgetExceededError,
    CacheError,
    ConfigError,
    FallbackExhaustedError,
    ForgeError,
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
    RegistryError,
    ValidationError,
)

# All concrete error classes Forge code paths may raise.
ALL_ERRORS: tuple[type[ForgeError], ...] = (
    ConfigError,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderBadRequestError,
    ProviderServerError,
    ProviderContentFilterError,
    BudgetExceededError,
    ValidationError,
    CacheError,
    RegistryError,
    FallbackExhaustedError,
)

PROVIDER_SUBCLASSES: tuple[type[ProviderError], ...] = (
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderBadRequestError,
    ProviderServerError,
    ProviderContentFilterError,
)


class TestHierarchy:
    """All Forge errors must inherit from `ForgeError`; provider errors from `ProviderError`."""

    @pytest.mark.parametrize("cls", ALL_ERRORS)
    def test_subclass_of_forge_error(self, cls: type[ForgeError]) -> None:
        assert issubclass(cls, ForgeError)

    @pytest.mark.parametrize("cls", PROVIDER_SUBCLASSES)
    def test_subclass_of_provider_error(self, cls: type[ProviderError]) -> None:
        assert issubclass(cls, ProviderError)

    def test_forge_error_is_exception(self) -> None:
        assert issubclass(ForgeError, Exception)

    @pytest.mark.parametrize("cls", ALL_ERRORS)
    def test_can_be_caught_as_forge_error(self, cls: type[ForgeError]) -> None:
        with pytest.raises(ForgeError):
            raise cls("boom")


class TestForgeErrorBase:
    def test_plain_construction(self) -> None:
        err = ForgeError("something went wrong")
        assert str(err) == "something went wrong"


class TestConfigError:
    def test_without_source(self) -> None:
        err = ConfigError("missing setting")
        assert str(err) == "missing setting"
        assert err.source is None

    def test_with_source(self) -> None:
        err = ConfigError("missing env var", source="OPENAI_API_KEY")
        assert err.source == "OPENAI_API_KEY"
        assert "OPENAI_API_KEY" in str(err)
        assert "source:" in str(err)

    def test_source_is_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            ConfigError("oops", "OPENAI_API_KEY")  # type: ignore[misc]


class TestProviderError:
    def test_plain_construction(self) -> None:
        err = ProviderError("boom")
        assert str(err) == "boom"
        assert err.model is None
        assert err.provider is None
        assert err.status_code is None

    def test_with_full_route(self) -> None:
        err = ProviderError(
            "rate limited",
            model="claude-opus-4-7",
            provider="anthropic",
            status_code=429,
        )
        rendered = str(err)
        assert "rate limited" in rendered
        assert "model=claude-opus-4-7" in rendered
        assert "provider=anthropic" in rendered
        assert "status=429" in rendered

    def test_partial_annotations(self) -> None:
        err = ProviderError("oops", model="gpt-5.5")
        rendered = str(err)
        assert "model=gpt-5.5" in rendered
        assert "provider=" not in rendered
        assert "status=" not in rendered

    def test_kwargs_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            ProviderError("oops", "claude-opus-4-7")  # type: ignore[misc]

    @pytest.mark.parametrize("cls", PROVIDER_SUBCLASSES)
    def test_subclass_carries_route(self, cls: type[ProviderError]) -> None:
        err = cls("boom", model="gpt-5.5", provider="azure", status_code=500)
        assert err.model == "gpt-5.5"
        assert err.provider == "azure"
        assert err.status_code == 500


class TestBudgetExceededError:
    def test_plain_construction(self) -> None:
        err = BudgetExceededError("over budget")
        assert err.limit_usd is None
        assert err.limit_tokens is None
        assert err.spent_usd is None
        assert err.spent_tokens is None

    def test_full_construction(self) -> None:
        err = BudgetExceededError(
            "would exceed budget",
            limit_usd=1.0,
            limit_tokens=10_000,
            spent_usd=1.05,
            spent_tokens=10_500,
        )
        assert err.limit_usd == 1.0
        assert err.limit_tokens == 10_000
        assert err.spent_usd == 1.05
        assert err.spent_tokens == 10_500

    def test_kwargs_are_keyword_only(self) -> None:
        with pytest.raises(TypeError):
            BudgetExceededError("oops", 1.0)  # type: ignore[misc]


class TestValidationError:
    def test_chained_from_pydantic(self) -> None:
        original = ValueError("not an int")
        with pytest.raises(ValidationError) as excinfo:
            raise ValidationError("bad input") from original
        assert excinfo.value.__cause__ is original


class TestCacheError:
    def test_with_backend(self) -> None:
        err = CacheError("redis unreachable", backend="redis")
        assert err.backend == "redis"
        assert "redis" in str(err)
        assert "backend:" in str(err)

    def test_without_backend(self) -> None:
        err = CacheError("eviction failed")
        assert err.backend is None
        assert str(err) == "eviction failed"


class TestRegistryError:
    def test_full_annotations(self) -> None:
        err = RegistryError(
            "unsupported route",
            model="claude-opus-4-7",
            provider="openai",
            reason="unsupported_route",
        )
        rendered = str(err)
        assert "unsupported route" in rendered
        assert "reason=unsupported_route" in rendered
        assert "model=claude-opus-4-7" in rendered
        assert "provider=openai" in rendered

    def test_without_annotations(self) -> None:
        err = RegistryError("generic")
        assert str(err) == "generic"


class TestFallbackExhaustedError:
    def test_empty_causes(self) -> None:
        err = FallbackExhaustedError("nothing tried")
        assert err.causes == []
        assert str(err) == "nothing tried"

    def test_aggregates_causes(self) -> None:
        causes: list[tuple[str, str | None, BaseException]] = [
            ("claude-opus-4-7", "anthropic", ProviderRateLimitError("429")),
            ("claude-opus-4-7", "bedrock", ProviderTimeoutError("timeout")),
            ("gpt-5.5", "openai", ProviderServerError("500")),
        ]
        err = FallbackExhaustedError("all routes failed", causes=causes)
        rendered = str(err)
        assert "all routes failed" in rendered
        assert "claude-opus-4-7@anthropic" in rendered
        assert "claude-opus-4-7@bedrock" in rendered
        assert "gpt-5.5@openai" in rendered
        assert "ProviderRateLimitError" in rendered
        assert "ProviderTimeoutError" in rendered
        assert "ProviderServerError" in rendered
        assert err.causes == causes

    def test_cause_provider_can_be_none(self) -> None:
        # If route resolution itself failed, there's no provider yet — `None` is valid.
        causes: list[tuple[str, str | None, BaseException]] = [
            ("claude-opus-99", None, RegistryError("unknown model", reason="unknown_model")),
        ]
        err = FallbackExhaustedError("registry rejected first entry", causes=causes)
        assert "claude-opus-99@default" in str(err)

    def test_causes_list_is_copied(self) -> None:
        causes: list[tuple[str, str | None, BaseException]] = [
            ("claude-opus-4-7", "anthropic", ProviderRateLimitError("429")),
        ]
        err = FallbackExhaustedError("failed", causes=causes)
        causes.clear()
        # The error keeps its own copy — mutating the caller's list doesn't strip evidence.
        assert len(err.causes) == 1


class TestRaiseFromChaining:
    """`raise ProviderError(...) from underlying` should set `__cause__` cleanly."""

    def test_chain_with_from(self) -> None:
        original = ConnectionError("network down")
        with pytest.raises(ProviderTimeoutError) as excinfo:
            raise ProviderTimeoutError("request timed out", provider="anthropic") from original
        assert excinfo.value.__cause__ is original
        assert excinfo.value.provider == "anthropic"


class TestPublicReExports:
    """All concrete error classes are re-exported from `forge.core`."""

    def test_reexports(self) -> None:
        import forge.core as core

        for cls in ALL_ERRORS:
            assert getattr(core, cls.__name__) is cls
        assert core.ForgeError is ForgeError
