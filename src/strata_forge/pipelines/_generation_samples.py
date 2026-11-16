"""Stable logical sample identities and explicit seed inputs without native execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from strata_forge.llm._generation_schema import GenerationError, canonical_json, scalar_text
from strata_forge.llm.generation import DerivedSeed, GenerationRequest
from strata_forge.pipelines.generation_spec import revalidate_generation_spec

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from strata_forge.pipelines.generation_spec import GenerationRunSpecV2

MAX_PLANNED_SAMPLES = 1_000_000


@dataclass(frozen=True)
class PlannedGenerationSample:
    item_id: str
    sample_index: int
    effective_seed_decimal: str | None


def _seed(spec: GenerationRunSpecV2, item_id: str, sample_index: int) -> str | None:
    policy = spec.generation.seed
    if not isinstance(policy, DerivedSeed):
        return None
    encoded = canonical_json(
        [
            "strata.generation.seed/v1",
            policy.base_seed,
            spec.dataset_ref,
            spec.dataset_config,
            spec.split,
            item_id,
            sample_index,
        ]
    )
    return str(int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (2**63 - 1))


def plan_samples(
    spec: GenerationRunSpecV2, item_ids: Sequence[str]
) -> Iterator[PlannedGenerationSample]:
    """Validate the full bounded ID membership before exposing any fan-out element."""
    current = revalidate_generation_spec(spec)
    selection = current.selection
    selected_limit = (
        min(selection.row_limit, selection.row_end - selection.row_start)
        if selection.row_end is not None
        else selection.row_limit
    )
    if len(item_ids) > selected_limit:
        raise GenerationError("generation_selection_mismatch")
    if len(item_ids) * current.generation.samples > MAX_PLANNED_SAMPLES:
        raise GenerationError("generation_sample_limit_exceeded")
    identities: list[str] = []
    seen: set[str] = set()
    for item_id in item_ids:
        if len(identities) >= selected_limit:
            raise GenerationError("generation_selection_mismatch")
        if (len(identities) + 1) * current.generation.samples > MAX_PLANNED_SAMPLES:
            raise GenerationError("generation_sample_limit_exceeded")
        if (
            not scalar_text(item_id)
            or not item_id
            or len(item_id) > 256
            or len(item_id.encode("utf-8")) > 256
            or item_id in seen
        ):
            raise GenerationError("generation_item_id_invalid")
        seen.add(item_id)
        identities.append(item_id)
    if not identities:
        raise GenerationError("generation_population_empty")

    def values() -> Iterator[PlannedGenerationSample]:
        for item_id in identities:
            for index in range(current.generation.samples):
                yield PlannedGenerationSample(item_id, index, _seed(current, item_id, index))

    return values()


def sample_request(spec: GenerationRunSpecV2, sample: PlannedGenerationSample) -> GenerationRequest:
    """Produce one effective request; raw sample metadata never grants native capability."""
    current = revalidate_generation_spec(spec)
    if (
        type(sample) is not PlannedGenerationSample
        or type(sample.sample_index) is not int
        or not 0 <= sample.sample_index < current.generation.samples
        or not scalar_text(sample.item_id)
        or not sample.item_id
        or len(sample.item_id.encode("utf-8")) > 256
        or sample.effective_seed_decimal != _seed(current, sample.item_id, sample.sample_index)
    ):
        raise GenerationError("generation_sample_invalid")
    raw = current.generation.wire_payload()
    raw.pop("samples")
    raw.pop("seed")
    return GenerationRequest.model_validate(
        {
            **raw,
            "model_ref": current.model_ref,
            "profile": current.profile,
            "effective_seed_decimal": sample.effective_seed_decimal,
            "deadline_seconds": current.serving.deadline_seconds,
        }
    )
