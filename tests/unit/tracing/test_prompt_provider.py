"""Closed reported metadata is not an issued capture handle or provider assertion."""

from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from strata_forge.tracing.prompt_provider_types import (
    BASE_PROVIDER_GAPS,
    ProviderAttemptMetadata,
    ProviderObservationMetadata,
)
from strata_forge.tracing.prompt_provider_values import project_chat, project_response


def reported_attempt(**changes: object) -> ProviderAttemptMetadata:
    fields: dict[str, object] = {
        "domain": "strata-provider-prompt-attempt-v1",
        "version": 1,
        "observation_id": "a" * 32,
        "attempt_id": "b" * 32,
        "entry_index": 0,
        "invocation_id": "c" * 32,
        "build_id": "d" * 64,
        "render_id": "e" * 64,
        "input_fingerprint": "f" * 64,
        "source_kind": "caller_bytes",
        "method": "acompletion",
        "declared_provider_slot": "openai_compat",
        "observed_provider_model_id": "fixture:model",
        "logical_model": None,
        "served_model": None,
        "model_revision": None,
        "source_relation": "unverified_argument",
        "started_at": "2026-11-07T00:00:00.000000Z",
        "ended_at": "2026-11-07T00:00:00.000001Z",
        "duration_us": 1,
        "call_outcome": "returned",
        "request_entered": True,
        "return_observed": True,
        "error_category": None,
        "request_projection": project_chat("fixture:model", [], {}),
        "response_projection": project_response("acompletion", {"choices": []}),
        "capture_status": "recorded",
        "capture_issues": (),
        "parents": None,
        "relation": "unknown",
        "render_relation": "unknown",
        "usage": None,
        "cost": None,
        "retention": "metadata_only",
        "export_status": "not_configured",
        "gaps": BASE_PROVIDER_GAPS,
    }
    return ProviderAttemptMetadata.model_validate({**fields, **changes})


def reported_observation(**changes: object) -> ProviderObservationMetadata:
    attempt = reported_attempt()
    fields: dict[str, object] = {
        "domain": "strata-provider-prompt-observation-v1",
        "version": 1,
        "observation_id": attempt.observation_id,
        "invocation_id": attempt.invocation_id,
        "source_kind": "caller_bytes",
        "source_partition": "fixture:metadata",
        "source_reference": None,
        "build_id": attempt.build_id,
        "render_id": attempt.render_id,
        "input_fingerprint": attempt.input_fingerprint,
        "created_at": "2026-11-07T00:00:00.000000Z",
        "closed": False,
        "admitted_attempts": 1,
        "attempts": (attempt,),
        "loss": {
            "over_capacity_count": 0,
            "projection_limited_count": 0,
            "metadata_lost_count": 0,
            "scope_mismatch_seen": False,
            "counter_saturated": False,
        },
        "coverage": {
            "all_calls_observed": False,
            "streaming_observed": False,
            "cache_events_observed": False,
            "post_close_calls_counted": False,
        },
        "retention": "metadata_only",
        "export_status": "not_configured",
    }
    return ProviderObservationMetadata.model_validate({**fields, **changes})


def test_reported_metadata_hashes_are_domain_separated_and_have_no_provider_claim() -> None:
    attempt = reported_attempt()
    observation = reported_observation()
    assert len(attempt.metadata_sha256) == len(observation.metadata_sha256) == 64
    assert attempt.metadata_sha256 != observation.metadata_sha256
    assert attempt.logical_model is attempt.served_model is attempt.model_revision is None
    assert attempt.parents is None
    assert observation.coverage.all_calls_observed is False
    assert observation.source_kind == "caller_bytes"


@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"domain": "wrong"},
        {"request_entered": False},
        {"return_observed": 1},
        {"entry_index": 64},
        {"duration_us": 0.7},
        {"declared_provider_slot": "verified_vendor"},
        {"logical_model": "secret"},
        {"served_model": "secret"},
        {"model_revision": "a" * 40},
        {"parents": {"retry_of": "a" * 32}},
        {"relation": "retry"},
        {"render_relation": "exact"},
        {"usage": 0},
        {"cost": 0},
        {"error_category": "CANARY"},
        {"extra": "CANARY"},
        {"capture_issues": ("response_opaque",)},
        {"capture_status": "partial"},
        {"gaps": ()},
        {"gaps": (*BASE_PROVIDER_GAPS, "response_projection_unavailable")},
        {"call_outcome": "raised"},
        {"return_observed": False},
        {"started_at": "2026-02-30T00:00:00.000000Z"},
        {"attempt_id": "a" * 32 + "\n"},
    ],
)
def test_forged_reported_facts_refuse_with_content_free_errors(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError) as caught:
        reported_attempt(**changes)
    assert "CANARY" not in str(caught.value)


@pytest.mark.parametrize("ending", ["\n", "\r", "\u2028", "\u2029"])
def test_observed_label_has_complete_ascii_grammar(ending: str) -> None:
    with pytest.raises(ValidationError):
        reported_attempt(observed_provider_model_id="fixture" + ending)


def test_unknown_response_projection_is_partial_not_a_failed_provider_call() -> None:
    attempt = reported_attempt(
        response_projection=project_response("acompletion", object()),
        capture_status="partial",
        capture_issues=("response_opaque",),
        gaps=(*BASE_PROVIDER_GAPS, "response_projection_unavailable"),
    )
    assert attempt.call_outcome == "returned"
    assert attempt.return_observed
    assert attempt.response_projection.canonical_tagged_sha256 is None
    assert attempt.error_category is None


def test_honest_metadata_loss_keeps_admission_holes_and_rejects_reuse_or_false_counts() -> None:
    observation = reported_observation()
    fields = observation.model_dump()
    loss = cast("dict[str, object]", fields["loss"])
    honest = reported_observation(attempts=(), loss={**loss, "metadata_lost_count": 1})
    assert honest.admitted_attempts == 1
    with pytest.raises(ValidationError):
        reported_observation(attempts=())
    with pytest.raises(ValidationError):
        reported_observation(attempts=(reported_attempt(), reported_attempt()), admitted_attempts=2)
    with pytest.raises(ValidationError):
        reported_observation(
            coverage={
                "all_calls_observed": True,
                "streaming_observed": False,
                "cache_events_observed": False,
                "post_close_calls_counted": False,
            }
        )
