"""Langfuse observability layered above the rest of `forge.*`.

See `docs/architecture/adr/0008-tracing-as-cross-cutting.md` for the
design and `docs/modules/tracing.md` for the user-facing reference.
"""

from forge.tracing.client import get_client, reset_client

__all__ = [
    "get_client",
    "reset_client",
]
