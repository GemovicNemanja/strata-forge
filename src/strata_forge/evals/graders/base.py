"""The :class:`Grader` Protocol — the only contract graders must satisfy.

Any object with the right shape is a grader; subclassing isn't
required. The Protocol is async because LLM-judge graders need
:meth:`LLMClient.complete_structured`. Pure-text graders just don't
``await`` anything inside their :meth:`grade` body — the
``async def`` is there for shape uniformity.

See [ADR 0010](../../../../docs/architecture/adr/0010-evals-experiment-as-data-pluggable-graders.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from strata_forge.datasets.schema import DatasetItem
    from strata_forge.evals.experiment import GraderResult
    from strata_forge.llm.responses import LLMResponse

__all__ = ["Grader"]


@runtime_checkable
class Grader(Protocol):
    """The contract every grader satisfies.

    ``name`` is used in reports and as the key in
    :attr:`Outcome.grader_results`; the runner enforces uniqueness
    across the graders attached to a single experiment.
    """

    @property
    def name(self) -> str: ...  # pragma: no cover — Protocol body

    async def grade(
        self,
        *,
        item: DatasetItem,
        response: LLMResponse,
    ) -> GraderResult: ...  # pragma: no cover — Protocol body
