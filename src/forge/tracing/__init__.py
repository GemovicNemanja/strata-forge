"""Langfuse observability layered above the rest of `forge.*`.

See `docs/architecture/adr/0008-tracing-as-cross-cutting.md` for the
design and `docs/modules/tracing.md` for the user-facing reference.
"""

from forge.tracing.client import get_client, reset_client
from forge.tracing.decorator import traced
from forge.tracing.litellm_callback import (
    LITELLM_CALLBACK_NAME,
    install_litellm_callback,
    is_litellm_callback_installed,
)
from forge.tracing.metrics import record_categorical_metric, record_numeric_metric
from forge.tracing.score import ScoreValue, score_observation, score_trace
from forge.tracing.span import traced_span

__all__ = [
    "LITELLM_CALLBACK_NAME",
    "ScoreValue",
    "get_client",
    "install_litellm_callback",
    "is_litellm_callback_installed",
    "record_categorical_metric",
    "record_numeric_metric",
    "reset_client",
    "score_observation",
    "score_trace",
    "traced",
    "traced_span",
]
