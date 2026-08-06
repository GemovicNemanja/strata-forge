"""Self-instruct synthetic-data generation.

Asks an :class:`~strata_forge.llm.LLMClient` to produce new dataset items
that follow the pattern of a small set of seed examples. The function
batches generation requests, deduplicates by content-hash ID, and
loops until ``n`` distinct items accumulate or ``max_attempts``
batches have been issued without progress.

The wrapper schema (:class:`SelfInstructBatch` / :class:`SelfInstructItem`)
is what the LLM is asked to produce via
:meth:`LLMClient.complete_structured`. Items emitted by the LLM are
converted into :class:`strata_forge.datasets.DatasetItem` instances with
content-hash IDs derived from ``input + expected_output`` — duplicates
across batches are dropped without re-prompting.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.datasets.schema import Dataset, DatasetItem
from strata_forge.llm.messages import SystemMessage, UserMessage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strata_forge.llm.client import LLMClient
    from strata_forge.llm.messages import AnyMessage

__all__ = [
    "SelfInstructBatch",
    "SelfInstructItem",
    "self_instruct",
]

DEFAULT_METADATA: dict[str, Any] = {"synthetic": "self_instruct"}


class SelfInstructItem(BaseModel):
    """One generated item, as the LLM is asked to emit it."""

    model_config = ConfigDict(extra="forbid")

    input: dict[str, Any] = Field(
        description="Input dict for the new dataset item; same shape as the seeds."
    )
    expected_output: Any | None = Field(
        default=None,
        description="Reference output for the new item, when applicable.",
    )


class SelfInstructBatch(BaseModel):
    """A batch of items the LLM emits in one structured-output call."""

    model_config = ConfigDict(extra="forbid")

    items: list[SelfInstructItem] = Field(description="Generated items in this batch.")


def _seeds_to_user_message(seeds: Sequence[DatasetItem]) -> str:
    blocks: list[str] = []
    for index, seed in enumerate(seeds):
        block = (
            f"Example {index + 1}:\n"
            f"input: {json.dumps(seed.input, sort_keys=True)}\n"
            f"expected_output: {json.dumps(seed.expected_output, sort_keys=True)}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


def _build_messages(
    *,
    seeds: Sequence[DatasetItem],
    instructions: str,
    batch_size: int,
    already_generated: int,
    target: int,
) -> list[AnyMessage]:
    system = SystemMessage(
        content=(
            "You generate new dataset items by following the pattern of the "
            "examples the user provides. Match the field names and shapes of "
            "the seed examples exactly. Each generated item must be distinct "
            "from every seed and from items you have produced earlier. Reply "
            "with the structured-output schema."
        )
    )
    user = UserMessage(
        content=(
            f"Instructions: {instructions}\n\n"
            f"Seed examples:\n\n{_seeds_to_user_message(seeds)}\n\n"
            f"Generate {batch_size} new items in the same shape. "
            f"({already_generated} of {target} done so far.)"
        )
    )
    return [system, user]


async def self_instruct(
    *,
    seeds: Sequence[DatasetItem] | Dataset,
    instructions: str,
    n: int,
    client: LLMClient,
    name: str,
    description: str = "",
    batch_size: int = 10,
    temperature: float = 0.9,
    max_attempts: int = 8,
    metadata: dict[str, Any] | None = None,
) -> Dataset:
    """Generate ``n`` new :class:`DatasetItem` instances from seed examples.

    The function asks the LLM via
    :meth:`LLMClient.complete_structured` for batches of up to
    ``batch_size`` items at a time, accumulating distinct items
    (deduplicated by content-hash ID against the seeds and prior
    batches) until ``n`` distinct items have been produced.

    ``max_attempts`` caps the number of LLM batches issued. The loop
    also terminates early when an entire batch produces no new items
    (signalling the LLM has run out of novel patterns); in that case
    the partial dataset of whatever was produced is returned.

    Items default to ``metadata = {"synthetic": "self_instruct"}``;
    pass ``metadata`` to override or extend.
    """
    if n <= 0:
        msg = f"n must be positive, got {n}"
        raise ValueError(msg)
    if batch_size <= 0:
        msg = f"batch_size must be positive, got {batch_size}"
        raise ValueError(msg)

    seed_items: tuple[DatasetItem, ...] = (
        seeds.items if isinstance(seeds, Dataset) else tuple(seeds)
    )
    if not seed_items:
        msg = "seeds must contain at least one DatasetItem"
        raise ValueError(msg)

    item_metadata = dict(metadata) if metadata is not None else dict(DEFAULT_METADATA)
    seen_ids: set[str] = {item.id for item in seed_items}
    generated: list[DatasetItem] = []

    for _attempt in range(max_attempts):
        if len(generated) >= n:
            break
        remaining = n - len(generated)
        ask_size = min(batch_size, remaining)
        messages = _build_messages(
            seeds=seed_items,
            instructions=instructions,
            batch_size=ask_size,
            already_generated=len(generated),
            target=n,
        )
        response = await client.complete_structured(
            messages=messages,
            schema=SelfInstructBatch,
            temperature=temperature,
        )
        new_in_batch = 0
        for raw_item in response.parsed.items:
            item = DatasetItem.from_input(
                raw_item.input,
                expected_output=raw_item.expected_output,
                metadata=dict(item_metadata),
            )
            if item.id in seen_ids:
                continue
            seen_ids.add(item.id)
            generated.append(item)
            new_in_batch += 1
            if len(generated) >= n:
                break
        if new_in_batch == 0:
            # The LLM produced no novel items this round. Further
            # batches are unlikely to help; bail out with what we have.
            break

    return Dataset(
        name=name,
        items=tuple(generated),
        description=description,
        metadata={"synthetic": "self_instruct", "seed_count": len(seed_items)},
    )
