"""Langfuse observability and explicit local prompt observations, layered from above.

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
    from strata_forge.tracing.prompt_provider import (
        ProviderObservation as ProviderObservation,
    )
    from strata_forge.tracing.prompt_provider import (
        ProviderObservationError as ProviderObservationError,
    )
    from strata_forge.tracing.prompt_provider import (
        prepare_provider_observation as prepare_provider_observation,
    )
    from strata_forge.tracing.prompt_provider import (
        wrap_provider as wrap_provider,
    )
    from strata_forge.tracing.prompt_provider_types import (
        ProviderAttemptMetadata as ProviderAttemptMetadata,
    )
    from strata_forge.tracing.prompt_provider_types import (
        ProviderObservationCoverage as ProviderObservationCoverage,
    )
    from strata_forge.tracing.prompt_provider_types import (
        ProviderObservationLoss as ProviderObservationLoss,
    )
    from strata_forge.tracing.prompt_provider_types import (
        ProviderObservationMetadata as ProviderObservationMetadata,
    )
    from strata_forge.tracing.prompt_provider_types import (
        ProviderProjectionMetadata as ProviderProjectionMetadata,
    )
    from strata_forge.tracing.prompt_stream_types import (
        ProviderStreamMetadata as ProviderStreamMetadata,
    )
    from strata_forge.tracing.prompt_stream_types import (
        ProviderStreamObservationMetadata as ProviderStreamObservationMetadata,
    )
    from strata_forge.tracing.prompt_stream_types import StreamInputBinding as StreamInputBinding
    from strata_forge.tracing.prompt_stream_types import (
        StreamObservationCoverage as StreamObservationCoverage,
    )
    from strata_forge.tracing.prompt_stream_types import (
        StreamObservationLoss as StreamObservationLoss,
    )
    from strata_forge.tracing.prompt_streams import (
        ProviderStreamObservation as ProviderStreamObservation,
    )
    from strata_forge.tracing.prompt_streams import StreamObservationError as StreamObservationError
    from strata_forge.tracing.prompt_streams import (
        prepare_stream_observation as prepare_stream_observation,
    )
    from strata_forge.tracing.prompt_streams import (
        wrap_streaming_provider as wrap_streaming_provider,
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
    "ProviderAttemptMetadata",
    "ProviderObservation",
    "ProviderObservationCoverage",
    "ProviderObservationError",
    "ProviderObservationLoss",
    "ProviderObservationMetadata",
    "ProviderProjectionMetadata",
    "ProviderStreamMetadata",
    "ProviderStreamObservation",
    "ProviderStreamObservationMetadata",
    "ScoreValue",
    "StreamInputBinding",
    "StreamObservationCoverage",
    "StreamObservationError",
    "StreamObservationLoss",
    "capture_once",
    "get_client",
    "install_litellm_callback",
    "is_litellm_callback_installed",
    "prepare_invocation",
    "prepare_provider_observation",
    "prepare_stream_observation",
    "record_categorical_metric",
    "record_numeric_metric",
    "reset_client",
    "score_observation",
    "score_trace",
    "traced",
    "traced_span",
    "wrap_provider",
    "wrap_streaming_provider",
]


def __getattr__(name: str) -> object:
    if name in {
        "ProviderStreamObservation",
        "StreamObservationError",
        "prepare_stream_observation",
        "wrap_streaming_provider",
    }:
        import strata_forge.tracing.prompt_streams as streams

        return getattr(streams, name)
    if name in {
        "ProviderStreamMetadata",
        "ProviderStreamObservationMetadata",
        "StreamInputBinding",
        "StreamObservationCoverage",
        "StreamObservationLoss",
    }:
        import strata_forge.tracing.prompt_stream_types as stream_types

        return getattr(stream_types, name)
    if name in {
        "ProviderObservation",
        "ProviderObservationError",
        "prepare_provider_observation",
        "wrap_provider",
    }:
        import strata_forge.tracing.prompt_provider as provider

        return getattr(provider, name)
    if name in {
        "ProviderAttemptMetadata",
        "ProviderObservationCoverage",
        "ProviderObservationLoss",
        "ProviderObservationMetadata",
        "ProviderProjectionMetadata",
    }:
        import strata_forge.tracing.prompt_provider_types as provider_types

        return getattr(provider_types, name)
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
