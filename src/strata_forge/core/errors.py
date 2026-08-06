"""Exception hierarchy. Every Forge-raised error inherits from ``ForgeError``."""

from __future__ import annotations

__all__ = [
    "BudgetExceededError",
    "CacheError",
    "ConfigError",
    "FallbackExhaustedError",
    "ForgeError",
    "ProviderAuthError",
    "ProviderBadRequestError",
    "ProviderContentFilterError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderServerError",
    "ProviderTimeoutError",
    "RegistryError",
    "ValidationError",
]


class ForgeError(Exception):
    """Root of the Forge exception hierarchy.

    Every exception raised from ``strata_forge.*`` code paths inherits from this class so
    callers can write a single ``except ForgeError`` clause for catch-all behavior
    while still pattern-matching subclasses for finer control.
    """


class ConfigError(ForgeError):
    """Configuration is missing, malformed, or contradictory.

    The optional ``source`` attribute names the offending env var, YAML path, or
    overlay key so that error messages point at the fix.
    """

    def __init__(self, message: str, *, source: str | None = None) -> None:
        super().__init__(message)
        self.source = source

    def __str__(self) -> str:
        base = super().__str__()
        if self.source is not None:
            return f"{base} (source: {self.source})"
        return base


class ProviderError(ForgeError):
    """A provider-side failure surfaced through the LLM client.

    Subclasses distinguish failure categories (rate limit, auth, content filter,
    ...). The seam in ``strata_forge.llm.errors.map_litellm_exception`` is responsible
    for choosing the correct subclass when normalizing a LiteLLM exception, and
    for annotating the raised instance with ``model``/``provider``/``status_code``.
    Underlying provider exceptions are chained via ``raise ... from ...`` and
    available on ``__cause__``.
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.provider = provider
        self.status_code = status_code

    def __str__(self) -> str:
        base = super().__str__()
        annotations: list[str] = []
        if self.model is not None:
            annotations.append(f"model={self.model}")
        if self.provider is not None:
            annotations.append(f"provider={self.provider}")
        if self.status_code is not None:
            annotations.append(f"status={self.status_code}")
        if annotations:
            return f"{base} ({', '.join(annotations)})"
        return base


class ProviderAuthError(ProviderError):
    """Provider rejected the request because of authentication or authorization."""


class ProviderRateLimitError(ProviderError):
    """Provider returned a rate-limit response (HTTP 429 or equivalent)."""


class ProviderTimeoutError(ProviderError):
    """Request to the provider timed out before a response was received."""


class ProviderBadRequestError(ProviderError):
    """Provider rejected the request as malformed (HTTP 400 or equivalent)."""


class ProviderServerError(ProviderError):
    """Provider returned a server-side error (HTTP 5xx)."""


class ProviderContentFilterError(ProviderError):
    """Provider refused to generate output due to content policy.

    Fallback chains treat this as a terminal condition for the entire chain — the
    same content will be filtered on every other provider too, so retrying is
    wasted work.
    """


class BudgetExceededError(ForgeError):
    """Operation aborted because a cost or token ceiling would be exceeded.

    Raised by ``strata_forge.core.budget.BudgetContext`` before the budget is actually
    overrun, so callers see the failure cleanly instead of an unexpected bill.
    All four fields are optional because a budget may set either ``limit_usd`` or
    ``limit_tokens`` (or both); the matching ``spent_*`` field is populated.
    """

    def __init__(
        self,
        message: str,
        *,
        limit_usd: float | None = None,
        limit_tokens: int | None = None,
        spent_usd: float | None = None,
        spent_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.limit_usd = limit_usd
        self.limit_tokens = limit_tokens
        self.spent_usd = spent_usd
        self.spent_tokens = spent_tokens


class ValidationError(ForgeError):
    """Input or output failed Forge-side validation.

    Distinct from Pydantic's own ``ValidationError`` so callers can decide
    whether to catch Forge-validated failures specifically. When wrapping a
    Pydantic error, use ``raise ValidationError(...) from pydantic_error``.
    """


class CacheError(ForgeError):
    """A cache backend operation failed.

    The optional ``backend`` attribute names the backend (e.g. ``"redis"``,
    ``"memory"``) so log lines can pinpoint which subsystem misbehaved.
    """

    def __init__(self, message: str, *, backend: str | None = None) -> None:
        super().__init__(message)
        self.backend = backend

    def __str__(self) -> str:
        base = super().__str__()
        if self.backend is not None:
            return f"{base} (backend: {self.backend})"
        return base


class RegistryError(ForgeError):
    """Model registry lookup or validation failed.

    Raised for unknown models, missing pricing, unsupported ``(model, provider)``
    routes, and inconsistencies detected by ``strata-forge doctor``. The ``reason``
    field is a short machine-friendly tag (``"unknown_model"``,
    ``"missing_pricing"``, ``"unsupported_route"``, ``"capability_missing"``,
    ``"capability_unknown"``).
    """

    def __init__(
        self,
        message: str,
        *,
        model: str | None = None,
        provider: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.model = model
        self.provider = provider
        self.reason = reason

    def __str__(self) -> str:
        base = super().__str__()
        annotations: list[str] = []
        if self.reason is not None:
            annotations.append(f"reason={self.reason}")
        if self.model is not None:
            annotations.append(f"model={self.model}")
        if self.provider is not None:
            annotations.append(f"provider={self.provider}")
        if annotations:
            return f"{base} ({', '.join(annotations)})"
        return base


class FallbackExhaustedError(ForgeError):
    """Every entry in a fallback chain failed.

    Carries the full route history — one ``(model, provider, error)`` triple per
    attempt — so callers can introspect why the chain failed and report it in
    detail. ``provider`` is ``None`` for entries that failed before route
    resolution.
    """

    def __init__(
        self,
        message: str,
        *,
        causes: list[tuple[str, str | None, BaseException]] | None = None,
    ) -> None:
        super().__init__(message)
        self.causes: list[tuple[str, str | None, BaseException]] = list(causes or [])

    def __str__(self) -> str:
        base = super().__str__()
        if not self.causes:
            return base
        cause_lines = [
            f"  - {model}@{provider or 'default'}: {type(err).__name__}: {err}"
            for model, provider, err in self.causes
        ]
        return base + "\n" + "\n".join(cause_lines)
