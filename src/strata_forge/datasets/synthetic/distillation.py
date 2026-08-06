"""Teacher-student distillation primitive.

Runs a teacher :class:`~strata_forge.llm.LLMClient` over each input in a
dataset to produce a fresh ``expected_output``. The result is a new
:class:`~strata_forge.datasets.Dataset` — the input is never mutated.

The function is intentionally minimal: it serializes each item's
``input`` dict as JSON inside a single user message, asks the teacher
for a plain-text completion, and stores the response text as
``expected_output``. Callers needing richer prompt construction (e.g.
templating, role play, chain-of-thought) can pre-build the desired
messages and call :meth:`LLMClient.complete` directly — this helper
exists for the common case.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

from strata_forge.datasets.schema import Dataset, DatasetItem
from strata_forge.llm.messages import SystemMessage, UserMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.llm.client import LLMClient

__all__ = [
    "distill",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert annotator. Read the user's input carefully and "
    "produce the best possible answer. Reply with only the answer — no "
    "preamble, no explanation unless the input explicitly asks for it."
)


async def distill(
    *,
    inputs: Sequence[DatasetItem] | Dataset,
    teacher: LLMClient,
    name: str,
    description: str = "",
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    concurrency: int = 5,
    skip_existing: bool = True,
    temperature: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> Dataset:
    """Run ``teacher`` on each item's ``input`` to populate ``expected_output``.

    When ``skip_existing`` is True (default), items that already have
    a non-``None`` ``expected_output`` pass through unchanged — only
    items lacking labels are sent to the teacher. Set it to False to
    re-label every item.

    Teacher calls are dispatched concurrently via :func:`asyncio.gather`
    with a semaphore bounded by ``concurrency``. Item order in the
    output dataset matches the input order; failed calls (whatever
    error the teacher raises) propagate.

    Returns a fresh :class:`Dataset` named ``name``. Per-item metadata
    is tagged with ``{"synthetic": "distillation"}`` (merged with any
    explicit ``metadata`` argument).
    """
    if concurrency <= 0:
        msg = f"concurrency must be positive, got {concurrency}"
        raise ValueError(msg)

    source_items: tuple[DatasetItem, ...] = (
        inputs.items if isinstance(inputs, Dataset) else tuple(inputs)
    )

    extra_meta = dict(metadata) if metadata is not None else {}
    base_meta: dict[str, Any] = {"synthetic": "distillation", **extra_meta}

    semaphore = asyncio.Semaphore(concurrency)

    async def _label(item: DatasetItem) -> DatasetItem:
        if skip_existing and item.expected_output is not None:
            return item
        async with semaphore:
            response = await teacher.complete(
                messages=[
                    SystemMessage(content=system_prompt),
                    UserMessage(content=json.dumps(item.input, sort_keys=True)),
                ],
                temperature=temperature,
            )
        # Build a fresh item to preserve frozen-model semantics.
        merged_meta = {**dict(item.metadata), **base_meta}
        return DatasetItem(
            id=item.id,
            input=dict(item.input),
            expected_output=response.text,
            metadata=merged_meta,
        )

    labeled = await asyncio.gather(*(_label(item) for item in source_items))

    return Dataset(
        name=name,
        items=tuple(labeled),
        description=description,
        metadata={"synthetic": "distillation", "source_count": len(source_items)},
    )
