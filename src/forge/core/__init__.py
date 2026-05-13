"""Cross-cutting utilities: errors, retry, logging, budget, repro, ids."""

from forge.core.budget import BudgetContext, current_budget
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
from forge.core.logging import configure_logging, get_logger, traced_span
from forge.core.retry import DEFAULT_RETRY_ON, retry

__all__ = [
    "DEFAULT_RETRY_ON",
    "BudgetContext",
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
    "configure_logging",
    "correlation_id_var",
    "current_budget",
    "get_correlation_id",
    "get_logger",
    "new_correlation_id",
    "retry",
    "set_correlation_id",
    "traced_span",
    "uuid7",
]
