"""Cross-cutting utilities: errors, retry, logging, budget, repro, ids."""

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
from forge.core.ids import (
    correlation_id_var,
    get_correlation_id,
    new_correlation_id,
    set_correlation_id,
    uuid7,
)
from forge.core.retry import DEFAULT_RETRY_ON, retry

__all__ = [
    "DEFAULT_RETRY_ON",
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
    "correlation_id_var",
    "get_correlation_id",
    "new_correlation_id",
    "retry",
    "set_correlation_id",
    "uuid7",
]
