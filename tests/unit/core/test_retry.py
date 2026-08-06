"""Unit tests for `strata_forge.core.retry`."""

from __future__ import annotations

import pytest

from strata_forge.core.errors import (
    ProviderAuthError,
    ProviderBadRequestError,
    ProviderContentFilterError,
    ProviderRateLimitError,
    ProviderServerError,
    ProviderTimeoutError,
)
from strata_forge.core.retry import DEFAULT_RETRY_ON, retry


class TestDefaults:
    def test_default_retry_on_includes_transient(self) -> None:
        assert ProviderRateLimitError in DEFAULT_RETRY_ON
        assert ProviderTimeoutError in DEFAULT_RETRY_ON
        assert ProviderServerError in DEFAULT_RETRY_ON

    def test_default_retry_on_excludes_non_transient(self) -> None:
        assert ProviderAuthError not in DEFAULT_RETRY_ON
        assert ProviderBadRequestError not in DEFAULT_RETRY_ON
        assert ProviderContentFilterError not in DEFAULT_RETRY_ON

    def test_default_retry_on_size(self) -> None:
        assert len(DEFAULT_RETRY_ON) == 3


class TestAsyncRetry:
    async def test_succeeds_first_try(self) -> None:
        @retry(max_attempts=3, initial_wait=0.001, max_wait=0.001)
        async def f() -> int:
            return 42

        assert await f() == 42

    async def test_retries_until_success(self) -> None:
        calls = 0

        @retry(max_attempts=5, initial_wait=0.001, max_wait=0.001)
        async def f() -> int:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ProviderRateLimitError("rate limited")
            return 42

        assert await f() == 42
        assert calls == 3

    async def test_exhausts_attempts_and_reraises(self) -> None:
        calls = 0

        @retry(max_attempts=3, initial_wait=0.001, max_wait=0.001)
        async def f() -> int:
            nonlocal calls
            calls += 1
            raise ProviderRateLimitError("rate limited", provider="anthropic")

        with pytest.raises(ProviderRateLimitError) as excinfo:
            await f()
        assert calls == 3
        assert excinfo.value.provider == "anthropic"

    @pytest.mark.parametrize(
        "non_transient",
        [ProviderAuthError, ProviderBadRequestError, ProviderContentFilterError],
    )
    async def test_no_retry_on_non_transient(
        self,
        non_transient: type[Exception],
    ) -> None:
        calls = 0

        @retry(max_attempts=5, initial_wait=0.001, max_wait=0.001)
        async def f() -> int:
            nonlocal calls
            calls += 1
            raise non_transient("nope")

        with pytest.raises(non_transient):
            await f()
        assert calls == 1

    async def test_custom_retry_on(self) -> None:
        calls = 0

        @retry(
            max_attempts=3,
            initial_wait=0.001,
            max_wait=0.001,
            retry_on=(ProviderAuthError,),
        )
        async def f() -> int:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise ProviderAuthError("creds rotated")
            return 42

        assert await f() == 42
        assert calls == 2

    async def test_custom_retry_on_excludes_default(self) -> None:
        """If `retry_on` is overridden, the defaults no longer apply."""
        calls = 0

        @retry(
            max_attempts=3,
            initial_wait=0.001,
            max_wait=0.001,
            retry_on=(ProviderAuthError,),
        )
        async def f() -> int:
            nonlocal calls
            calls += 1
            raise ProviderRateLimitError("still transient")

        with pytest.raises(ProviderRateLimitError):
            await f()
        assert calls == 1


class TestSyncRetry:
    def test_succeeds_first_try(self) -> None:
        @retry(max_attempts=3, initial_wait=0.001, max_wait=0.001)
        def f() -> int:
            return 42

        assert f() == 42

    def test_retries_until_success(self) -> None:
        calls = 0

        @retry(max_attempts=5, initial_wait=0.001, max_wait=0.001)
        def f() -> int:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ProviderTimeoutError("timed out")
            return 42

        assert f() == 42
        assert calls == 3

    def test_exhausts_attempts_and_reraises(self) -> None:
        calls = 0

        @retry(max_attempts=2, initial_wait=0.001, max_wait=0.001)
        def f() -> int:
            nonlocal calls
            calls += 1
            raise ProviderServerError("500")

        with pytest.raises(ProviderServerError):
            f()
        assert calls == 2

    def test_no_retry_on_auth(self) -> None:
        calls = 0

        @retry(max_attempts=5, initial_wait=0.001, max_wait=0.001)
        def f() -> int:
            nonlocal calls
            calls += 1
            raise ProviderAuthError("bad creds")

        with pytest.raises(ProviderAuthError):
            f()
        assert calls == 1


class TestForms:
    """Both `@retry` (bare) and `@retry(...)` (with args) work."""

    def test_bare_decorator_form(self) -> None:
        @retry
        def f() -> int:
            return 42

        assert f() == 42

    async def test_bare_decorator_form_async(self) -> None:
        @retry
        async def f() -> int:
            return 42

        assert await f() == 42

    def test_args_form(self) -> None:
        @retry(max_attempts=2)
        def f() -> int:
            return 42

        assert f() == 42

    def test_args_form_empty_parens(self) -> None:
        @retry()
        def f() -> int:
            return 42

        assert f() == 42
