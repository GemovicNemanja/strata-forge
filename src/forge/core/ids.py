"""UUIDv7 generation and correlation-ID helpers.

UUIDv7 (RFC 9562) packs a 48-bit unix-millisecond timestamp into the most
significant bits of the UUID, making IDs naturally time-ordered while remaining
unpredictable. Forge uses UUIDv7 as the basis for correlation IDs so logs and
traces are k-sortable across services.
"""

from __future__ import annotations

import secrets
import time
from contextvars import ContextVar, Token
from uuid import UUID

__all__ = [
    "correlation_id_var",
    "get_correlation_id",
    "new_correlation_id",
    "set_correlation_id",
    "uuid7",
]


# Correlation ID lives in a ContextVar so it propagates across `await` boundaries
# automatically. Callers set/get via the helper functions below; the raw var is
# exported only so test fixtures can `reset` it cleanly.
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def uuid7() -> UUID:
    """Generate a UUIDv7 — time-ordered, k-sortable, RFC 9562 compliant.

    Layout (128 bits, big-endian):

    - bits 80-127: unix timestamp in milliseconds (48 bits)
    - bits 76-79: version = 7
    - bits 64-75: rand_a (12 bits, sub-millisecond ordering source)
    - bits 62-63: variant = 0b10 (RFC 4122)
    - bits 0-61: rand_b (62 bits of randomness)
    """
    timestamp_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF
    rand_a = secrets.randbits(12)
    rand_b = secrets.randbits(62)

    uuid_int = (timestamp_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0x2 << 62) | rand_b
    return UUID(int=uuid_int)


def new_correlation_id() -> str:
    """Generate a fresh correlation ID string (UUIDv7 hex, 32 chars)."""
    return uuid7().hex


def get_correlation_id() -> str | None:
    """Return the current correlation ID, or ``None`` if unset in this context."""
    return correlation_id_var.get()


def set_correlation_id(cid: str | None) -> Token[str | None]:
    """Set the current correlation ID; returns a token that can be passed to
    ``correlation_id_var.reset(token)`` to restore the previous value."""
    return correlation_id_var.set(cid)
