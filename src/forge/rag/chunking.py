"""Document + Chunk data shapes; Chunker Protocol + RecursiveChunker.

A :class:`Document` is the raw source text the pipeline ingests.
:meth:`Chunker.chunk` splits a :class:`Document` into a sequence of
:class:`Chunk` instances, each carrying its source ``document_id``
and any metadata the chunker (or the document) propagated.

:class:`RecursiveChunker` is the default splitter: it tries a list
of separators in order, preferring paragraph boundaries before
falling back to sentences, words, and characters. It tags chunks
with ``character_start`` / ``character_end`` in their metadata so
callers can highlight the source span in UI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Chunk",
    "Chunker",
    "Document",
    "RecursiveChunker",
]


class Document(BaseModel):
    """A raw source document the RAG pipeline ingests.

    Attributes:
        id: Stable identifier. Callers supply one (filename, URL,
            content hash) — the RAG module doesn't derive it.
        text: The document's textual content.
        metadata: Arbitrary JSON-serializable annotations
            (source URL, language tag, published_at, …).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str
    metadata: dict[str, Any] = Field(default={})


class Chunk(BaseModel):
    """A piece of a :class:`Document` after splitting.

    Attributes:
        id: Stable chunk identifier — typically
            ``f"{document_id}#{n}"`` for chunked documents, or any
            unique string for stand-alone chunks.
        text: The chunk's textual content.
        metadata: Arbitrary JSON-serializable annotations.
            :class:`RecursiveChunker` populates ``character_start``
            and ``character_end`` here so callers can highlight the
            source span.
        document_id: The source :class:`Document.id` when chunked
            from one, ``None`` for stand-alone chunks (e.g. inline
            snippets injected into an index).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    text: str
    metadata: dict[str, Any] = Field(default={})
    document_id: str | None = None


@runtime_checkable
class Chunker(Protocol):
    """The contract every chunker satisfies.

    Synchronous because chunking is CPU-bound and benefits from the
    GIL-free path. Async variants can be added if a remote splitter
    (e.g. an LLM-based semantic chunker that calls an API) lands
    later.
    """

    def chunk(self, document: Document) -> Sequence[Chunk]:
        """Split ``document`` into a sequence of chunks."""
        ...  # pragma: no cover — Protocol body


class RecursiveChunker:
    """Recursive character-based splitter with paragraph awareness.

    Tries separators in order (paragraph break → newline →
    sentence break → space → empty string) and splits the document
    along the first one that yields chunks within the size budget.
    Adjacent chunks share ``chunk_overlap`` characters to preserve
    cross-boundary context.

    Args:
        chunk_size: Maximum chunk length in characters. Default 1000.
        chunk_overlap: Number of characters of overlap between
            consecutive chunks. Must be < ``chunk_size``. Default
            100.
        separators: Ordered tuple of separator strings to try. The
            chunker prefers earlier separators (paragraph >
            sentence > word). Default
            ``("\\n\\n", "\\n", ". ", " ")``.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Sequence[str] = ("\n\n", "\n", ". ", " "),
    ) -> None:
        if chunk_size < 1:
            err = f"chunk_size must be >= 1; got {chunk_size}"
            raise ValueError(err)
        if chunk_overlap < 0:
            err = f"chunk_overlap must be >= 0; got {chunk_overlap}"
            raise ValueError(err)
        if chunk_overlap >= chunk_size:
            err = f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})"
            raise ValueError(err)
        if not separators:
            err = "separators must contain at least one entry"
            raise ValueError(err)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._separators = tuple(separators)

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_overlap(self) -> int:
        return self._chunk_overlap

    @property
    def separators(self) -> tuple[str, ...]:
        return self._separators

    def chunk(self, document: Document) -> Sequence[Chunk]:
        text = document.text
        if not text:
            return ()
        spans = self._split_with_offsets(text, 0)
        return tuple(
            Chunk(
                id=f"{document.id}#{index}",
                text=text[start:end],
                metadata={
                    **dict(document.metadata),
                    "character_start": start,
                    "character_end": end,
                    "chunk_index": index,
                },
                document_id=document.id,
            )
            for index, (start, end) in enumerate(spans)
        )

    def _split_with_offsets(self, text: str, base_offset: int) -> list[tuple[int, int]]:
        """Return ``[(start, end), ...]`` character spans for the chunks.

        Recurses through the separator list: tries the first
        separator on the whole text; if the resulting pieces are
        too large, the recursion drops to the next separator within
        the large piece.
        """
        if len(text) <= self._chunk_size:
            return [(base_offset, base_offset + len(text))]

        for separator in self._separators:
            if not separator or separator not in text:
                continue
            # Split, keeping track of character offsets.
            parts: list[tuple[int, str]] = []
            cursor = 0
            for piece in text.split(separator):
                parts.append((cursor, piece))
                cursor += len(piece) + len(separator)
            spans = self._merge_parts(parts, separator, base_offset)
            # If any single piece is still too big, recursively split
            # the offending span with the next separator.
            final: list[tuple[int, int]] = []
            for start, end in spans:
                if end - start <= self._chunk_size:
                    final.append((start, end))
                else:
                    sub = self._split_with_offsets(
                        text[start - base_offset : end - base_offset], start
                    )
                    final.extend(sub)
            return self._apply_overlap(final, base_offset, len(text))

        # No separator matched — hard-split by chunk_size with overlap.
        return self._hard_split(text, base_offset)

    def _merge_parts(
        self,
        parts: list[tuple[int, str]],
        separator: str,
        base_offset: int,
    ) -> list[tuple[int, int]]:
        """Greedy merge of adjacent ``separator``-separated parts up to chunk_size."""
        spans: list[tuple[int, int]] = []
        current_start: int | None = None
        current_end: int | None = None
        for offset, piece in parts:
            piece_start = base_offset + offset
            piece_end = piece_start + len(piece)
            if current_start is None:
                current_start, current_end = piece_start, piece_end
                continue
            tentative_end = piece_end
            if tentative_end - current_start <= self._chunk_size:
                current_end = tentative_end
            else:
                assert current_end is not None  # noqa: S101 — loop invariant
                spans.append((current_start, current_end))
                current_start, current_end = piece_start, piece_end
        if current_start is not None and current_end is not None:
            spans.append((current_start, current_end))
        return spans

    def _hard_split(self, text: str, base_offset: int) -> list[tuple[int, int]]:
        """Fixed-size split with overlap when no separator helps."""
        step = max(self._chunk_size - self._chunk_overlap, 1)
        spans: list[tuple[int, int]] = []
        i = 0
        while i < len(text):
            end = min(i + self._chunk_size, len(text))
            spans.append((base_offset + i, base_offset + end))
            if end == len(text):
                break
            i += step
        return spans

    def _apply_overlap(
        self,
        spans: list[tuple[int, int]],
        base_offset: int,
        total_length: int,
    ) -> list[tuple[int, int]]:
        """Extend each chunk's start leftward by ``chunk_overlap`` characters.

        Overlap doesn't apply to the very first chunk (no preceding
        text). The text is bounded by ``[base_offset, base_offset +
        total_length]``.
        """
        if self._chunk_overlap == 0 or len(spans) <= 1:
            return spans
        upper_bound = base_offset + total_length
        adjusted: list[tuple[int, int]] = [spans[0]]
        for start, end in spans[1:]:
            new_start = max(base_offset, start - self._chunk_overlap)
            new_end = min(end, upper_bound)
            adjusted.append((new_start, new_end))
        return adjusted
