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
from forge.tracing.span import traced_span

__all__ = [
    "LITELLM_CALLBACK_NAME",
    "get_client",
    "install_litellm_callback",
    "is_litellm_callback_installed",
    "reset_client",
    "traced",
    "traced_span",
]
