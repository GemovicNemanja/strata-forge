"""Langfuse observability and explicit local byte-call recording, layered from above.

See `docs/architecture/adr/0008-tracing-as-cross-cutting.md` for the
design and `docs/modules/tracing.md` for the user-facing reference.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata_forge.tracing.prompt_attempt_types import (
        AttemptFacts as AttemptFacts,
    )
    from strata_forge.tracing.prompt_attempt_types import (
        AttemptMetadata as AttemptMetadata,
    )
    from strata_forge.tracing.prompt_attempt_types import (
        AttemptParents as AttemptParents,
    )
    from strata_forge.tracing.prompt_attempt_types import (
        ByteObservation as ByteObservation,
    )
    from strata_forge.tracing.prompt_attempt_types import (
        DeclaredBodyPin as DeclaredBodyPin,
    )
    from strata_forge.tracing.prompt_attempt_types import (
        InvocationMetadata as InvocationMetadata,
    )
    from strata_forge.tracing.prompt_attempts import (
        CapturedExchange as CapturedExchange,
    )
    from strata_forge.tracing.prompt_attempts import (
        LocalByteResponse as LocalByteResponse,
    )
    from strata_forge.tracing.prompt_attempts import (
        LocalByteTransport as LocalByteTransport,
    )
    from strata_forge.tracing.prompt_attempts import (
        PreparedInvocation as PreparedInvocation,
    )
    from strata_forge.tracing.prompt_attempts import (
        PromptCaptureError as PromptCaptureError,
    )
    from strata_forge.tracing.prompt_attempts import (
        capture_once as capture_once,
    )
    from strata_forge.tracing.prompt_attempts import (
        prepare_invocation as prepare_invocation,
    )

from strata_forge.tracing.client import get_client, reset_client
from strata_forge.tracing.decorator import traced
from strata_forge.tracing.litellm_callback import (
    LITELLM_CALLBACK_NAME,
    install_litellm_callback,
    is_litellm_callback_installed,
)
from strata_forge.tracing.metrics import record_categorical_metric, record_numeric_metric
from strata_forge.tracing.score import ScoreValue, score_observation, score_trace
from strata_forge.tracing.span import traced_span

__all__ = [
    "LITELLM_CALLBACK_NAME",
    "AttemptFacts",
    "AttemptMetadata",
    "AttemptParents",
    "ByteObservation",
    "CapturedExchange",
    "DeclaredBodyPin",
    "InvocationMetadata",
    "LocalByteResponse",
    "LocalByteTransport",
    "PreparedInvocation",
    "PromptCaptureError",
    "ScoreValue",
    "capture_once",
    "get_client",
    "install_litellm_callback",
    "is_litellm_callback_installed",
    "prepare_invocation",
    "record_categorical_metric",
    "record_numeric_metric",
    "reset_client",
    "score_observation",
    "score_trace",
    "traced",
    "traced_span",
]


def __getattr__(name: str) -> object:
    if name in {
        "AttemptFacts",
        "AttemptMetadata",
        "AttemptParents",
        "ByteObservation",
        "DeclaredBodyPin",
        "InvocationMetadata",
    }:
        import strata_forge.tracing.prompt_attempt_types as types

        return getattr(types, name)
    if name in {
        "CapturedExchange",
        "LocalByteResponse",
        "LocalByteTransport",
        "PreparedInvocation",
        "PromptCaptureError",
        "capture_once",
        "prepare_invocation",
    }:
        import strata_forge.tracing.prompt_attempts as attempts

        return getattr(attempts, name)
    raise AttributeError(name)
