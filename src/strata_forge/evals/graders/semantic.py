"""Semantic-similarity grader — cosine distance between embeddings.

The grader needs an async embedding function; callers supply one
(typically a thin wrapper over LiteLLM's ``aembedding``,
``strata_forge.llm.LLMClient``'s embedding endpoint when it lands, or any
provider SDK). The grader is intentionally agnostic about which
embedding model produces the vectors — it only needs an
``async (text) -> Sequence[float]`` callable.

Embeddings for the candidate response and the item's
``expected_output`` are cosine-compared; ``pass_threshold`` controls
the score → passed projection. Items without ``expected_output``
fail with an explanation — similarity against an absent reference is
undefined.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from strata_forge.evals.experiment import GraderResult

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from strata_forge.datasets.schema import DatasetItem
    from strata_forge.llm.responses import LLMResponse


__all__ = [
    "EmbedFn",
    "SemanticSimilarity",
    "cosine_similarity",
]


type EmbedFn = Callable[[str], Awaitable[Sequence[float]]]
"""Async embedding function — takes a string, returns a fixed-length vector."""


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 when either vector is all zeros (undefined direction)
    so callers don't have to special-case that path. Raises
    :class:`ValueError` when the vectors have different lengths.
    """
    if len(a) != len(b):
        msg = f"vector length mismatch: {len(a)} vs {len(b)}"
        raise ValueError(msg)
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticSimilarity:
    """Cosine similarity between response and reference embeddings.

    Args:
        embed: An async function that returns an embedding vector
            for a single string. The grader calls it twice per
            ``grade`` invocation — once for ``response.text``, once
            for ``item.expected_output``.
        pass_threshold: Minimum cosine similarity for a passing
            verdict. Default ``0.8``.
        name: Override the grader's ``.name``.

    The grader maps the cosine similarity ``c`` (which lives in
    ``[-1, 1]``) onto the GraderResult ``score`` field as
    ``max(0.0, c)`` so it composes cleanly with metrics that assume
    non-negative scores. Pass/fail is decided against the raw
    similarity vs ``pass_threshold`` so callers can keep an
    intuitive threshold even when responses are orthogonal or
    anti-aligned.
    """

    def __init__(
        self,
        embed: EmbedFn,
        *,
        pass_threshold: float = 0.8,
        name: str | None = None,
    ) -> None:
        if not (-1.0 <= pass_threshold <= 1.0):
            msg = f"pass_threshold must be in [-1, 1]; got {pass_threshold}"
            raise ValueError(msg)
        self._embed = embed
        self._pass_threshold = pass_threshold
        self._name = name or "semantic_similarity"

    @property
    def name(self) -> str:
        return self._name

    @property
    def pass_threshold(self) -> float:
        return self._pass_threshold

    async def grade(self, *, item: DatasetItem, response: LLMResponse) -> GraderResult:
        if item.expected_output is None:
            return GraderResult(
                grader_name=self.name,
                score=0.0,
                passed=False,
                explanation="item has no expected_output",
            )

        response_vec = await self._embed(response.text)
        reference_vec = await self._embed(str(item.expected_output))
        similarity = cosine_similarity(response_vec, reference_vec)

        return GraderResult(
            grader_name=self.name,
            score=max(0.0, similarity),
            passed=similarity >= self._pass_threshold,
            explanation=f"cosine_similarity={similarity:.4f}",
            metadata={"cosine_similarity": similarity},
        )
