"""Unit tests for `strata_forge.rag.chunking`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from strata_forge.rag.chunking import Chunk, Chunker, Document, RecursiveChunker

# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------


class TestDocument:
    def test_basic_construction(self) -> None:
        doc = Document(id="doc-1", text="hello world")
        assert doc.id == "doc-1"
        assert doc.text == "hello world"
        assert doc.metadata == {}

    def test_with_metadata(self) -> None:
        doc = Document(id="d", text="t", metadata={"source": "test"})
        assert doc.metadata == {"source": "test"}

    def test_empty_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Document(id="", text="t")

    def test_is_frozen(self) -> None:
        doc = Document(id="d", text="t")
        with pytest.raises(ValidationError, match="frozen"):
            doc.id = "other"  # type: ignore[misc]

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Document(id="d", text="t", unknown="x")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


class TestChunk:
    def test_basic_construction(self) -> None:
        chunk = Chunk(id="c-1", text="hello", document_id="doc-1")
        assert chunk.id == "c-1"
        assert chunk.text == "hello"
        assert chunk.document_id == "doc-1"
        assert chunk.metadata == {}

    def test_document_id_optional(self) -> None:
        chunk = Chunk(id="c", text="t")
        assert chunk.document_id is None

    def test_is_frozen(self) -> None:
        chunk = Chunk(id="c", text="t")
        with pytest.raises(ValidationError, match="frozen"):
            chunk.text = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RecursiveChunker — construction validation
# ---------------------------------------------------------------------------


class TestChunkerConstruction:
    def test_default_values(self) -> None:
        chunker = RecursiveChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100
        assert chunker.separators == ("\n\n", "\n", ". ", " ")

    def test_invalid_chunk_size(self) -> None:
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveChunker(chunk_size=0)
        with pytest.raises(ValueError, match="chunk_size"):
            RecursiveChunker(chunk_size=-1)

    def test_negative_overlap_rejected(self) -> None:
        with pytest.raises(ValueError, match="chunk_overlap"):
            RecursiveChunker(chunk_overlap=-1)

    def test_overlap_exceeds_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="smaller than chunk_size"):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="smaller than chunk_size"):
            RecursiveChunker(chunk_size=100, chunk_overlap=200)

    def test_empty_separators_rejected(self) -> None:
        with pytest.raises(ValueError, match="separators"):
            RecursiveChunker(separators=())

    def test_satisfies_protocol(self) -> None:
        chunker = RecursiveChunker()
        assert isinstance(chunker, Chunker)


# ---------------------------------------------------------------------------
# RecursiveChunker — chunking behaviour
# ---------------------------------------------------------------------------


class TestChunking:
    def test_short_document_one_chunk(self) -> None:
        doc = Document(id="d", text="short")
        chunker = RecursiveChunker(chunk_size=1000)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "short"
        assert chunks[0].document_id == "d"

    def test_empty_text_returns_no_chunks(self) -> None:
        doc = Document(id="d", text="")
        chunks = RecursiveChunker().chunk(doc)
        assert chunks == ()

    def test_paragraph_split(self) -> None:
        text = "paragraph one\n\nparagraph two\n\nparagraph three"
        # chunk_size=20 forces each paragraph to be its own chunk
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(Document(id="d", text=text))
        assert len(chunks) == 3
        assert "paragraph one" in chunks[0].text
        assert "paragraph two" in chunks[1].text
        assert "paragraph three" in chunks[2].text

    def test_chunk_ids_namespaced_to_document(self) -> None:
        text = "para a\n\npara b\n\npara c"
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(Document(id="my-doc", text=text))
        assert [c.id for c in chunks] == ["my-doc#0", "my-doc#1", "my-doc#2"]

    def test_chunk_metadata_includes_offsets(self) -> None:
        text = "first chunk content\n\nsecond chunk content"
        chunker = RecursiveChunker(chunk_size=25, chunk_overlap=0)
        chunks = chunker.chunk(Document(id="d", text=text))
        # Each chunk has character_start, character_end, chunk_index.
        assert chunks[0].metadata["chunk_index"] == 0
        assert chunks[0].metadata["character_start"] == 0
        assert chunks[1].metadata["character_start"] > 0

    def test_document_metadata_propagated_to_chunks(self) -> None:
        doc = Document(id="d", text="a\n\nb", metadata={"source": "wiki"})
        chunker = RecursiveChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["source"] == "wiki"

    def test_falls_back_to_smaller_separator(self) -> None:
        # No paragraph breaks; chunker falls back to single-newline, then "."
        text = "one. two. three. four. five."
        chunker = RecursiveChunker(chunk_size=12, chunk_overlap=0)
        chunks = chunker.chunk(Document(id="d", text=text))
        assert len(chunks) > 1
        # Each chunk should be within chunk_size of the budget.
        for chunk in chunks:
            assert len(chunk.text) <= 12

    def test_hard_split_when_no_separator_matches(self) -> None:
        # No separator in the text — hard fixed-size split.
        text = "abcdefghijklmnopqrstuvwxyz"  # no spaces / newlines
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0, separators=("X",))
        chunks = chunker.chunk(Document(id="d", text=text))
        # 26 chars / 10 per chunk = 3 chunks (last is partial).
        assert len(chunks) == 3
        # First two are full length; last is the remainder.
        assert len(chunks[0].text) == 10
        assert len(chunks[1].text) == 10
        assert len(chunks[2].text) == 6

    def test_overlap_extends_subsequent_chunks(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz"
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=3, separators=("X",))
        chunks = chunker.chunk(Document(id="d", text=text))
        # Second chunk should start 3 characters before the boundary.
        assert chunks[1].text.startswith(chunks[0].text[-3:])

    def test_no_overlap_when_single_chunk(self) -> None:
        text = "fits"
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        chunks = chunker.chunk(Document(id="d", text=text))
        assert chunks[0].text == "fits"
