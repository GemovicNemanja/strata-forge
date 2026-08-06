"""Sequence packing for SFT.

Greedy first-fit packing of multiple short sequences into
fixed-length packs. Useful for SFT throughput: short conversations
otherwise waste most of the model's context.

Each packed sequence concatenates tokenized sources separated by
an EOS token so the model still learns conversation boundaries.
The remaining slack at the end of each pack is filled with a
pad token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "PackedSequence",
    "pack_sequences",
]


@dataclass(frozen=True, slots=True)
class PackedSequence:
    """One output of :func:`pack_sequences`.

    Attributes:
        input_ids: Token IDs, padded to ``max_length``.
        attention_mask: 1 for real tokens, 0 for pad tokens.
        source_indices: Indices into the original input list that
            were packed into this sequence, in order.
    """

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    source_indices: tuple[int, ...]


def pack_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    max_length: int,
    eos_token_id: int,
    pad_token_id: int,
) -> tuple[PackedSequence, ...]:
    """Greedy first-fit pack ``sequences`` into ``max_length`` packs.

    Each input sequence is treated atomically — a sequence is
    never split across packs. Sequences longer than
    ``max_length - 1`` (1 reserved for the EOS) are truncated
    before packing.

    Args:
        sequences: Tokenized inputs to pack.
        max_length: Target length of each output pack.
        eos_token_id: Token appended after each source sequence.
        pad_token_id: Token used to fill the slack at the end of
            each pack.

    Returns:
        A tuple of :class:`PackedSequence`, in the order they were
        produced by the greedy algorithm.

    Raises:
        ValueError: ``max_length`` is not at least 2.
    """
    if max_length < 2:
        err = f"max_length must be >= 2; got {max_length}"
        raise ValueError(err)

    packs: list[PackedSequence] = []
    current_ids: list[int] = []
    current_sources: list[int] = []

    def _flush() -> None:
        if not current_ids:
            return
        attention = [1] * len(current_ids)
        pad_count = max_length - len(current_ids)
        packed_ids = (*current_ids, *([pad_token_id] * pad_count))
        packed_mask = (*attention, *([0] * pad_count))
        packs.append(
            PackedSequence(
                input_ids=packed_ids,
                attention_mask=packed_mask,
                source_indices=tuple(current_sources),
            )
        )
        current_ids.clear()
        current_sources.clear()

    for idx, seq in enumerate(sequences):
        # Truncate sources too long to ever fit, leaving room for EOS.
        truncated = list(seq[: max_length - 1])
        needed = len(truncated) + 1  # plus eos
        if needed > max_length - len(current_ids):
            _flush()
        current_ids.extend(truncated)
        current_ids.append(eos_token_id)
        current_sources.append(idx)
        if len(current_ids) == max_length:
            _flush()

    _flush()
    return tuple(packs)
