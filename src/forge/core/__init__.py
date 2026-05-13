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
from forge.core.repro import content_hash, env_snapshot, set_seed
from forge.core.retry import DEFAULT_RETRY_ON, retry
from forge.core.types import JSONValue, PathLike

__all__ = [
    "DEFAULT_RETRY_ON",
    "BudgetContext",
    "BudgetExceededError",
    "CacheError",
    "ConfigError",
    "FallbackExhaustedError",
    "ForgeError",
    "JSONValue",
    "PathLike",
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
    "content_hash",
    "correlation_id_var",
    "current_budget",
    "env_snapshot",
    "get_correlation_id",
    "get_logger",
    "new_correlation_id",
    "retry",
    "set_correlation_id",
    "set_seed",
    "traced_span",
    "uuid7",
]
