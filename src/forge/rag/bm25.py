"""Pure-Python BM25 retriever — sparse / lexical search.

:class:`BM25Retriever` indexes a collection of chunks at construction
time, then serves queries via the Okapi BM25 scoring function. The
implementation is intentionally dep-free: a couple of dicts and a few
arithmetic operations, no external SDKs.

BM25 complements dense retrieval: dense embeddings catch
paraphrases and conceptual matches, BM25 catches keyword overlap
that embeddings can drown out (numeric tokens, identifiers, rare
proper nouns). The hybrid retriever in
:mod:`forge.rag.hybrid` fuses both signals.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING

from forge.rag.retrieval import RetrievalResult

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from forge.rag.chunking import Chunk

__all__ = [
    "BM25Retriever",
    "tokenize",
]


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lower-case word-character tokenizer.

    The default tokenizer used by :class:`BM25Retriever` — exposed
    so callers can build matching tokenizers for custom domains.
    """
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


class BM25Retriever:
    """In-memory BM25 retriever.

    Args:
        chunks: Corpus to index. The chunks are tokenized once at
            construction time; subsequent retrievals only walk the
            index.
        k1: BM25 term-frequency saturation parameter. Default
            ``1.5``. Larger values let term frequency contribute
            more before saturating.
        b: BM25 length-normalization parameter. ``0`` disables
            length normalization, ``1`` fully normalizes. Default
            ``0.75``.

    The retriever raises :class:`ValueError` when the corpus is
    empty — :meth:`retrieve` on an empty index has no defined
    behaviour.
    """

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            err = "BM25Retriever: chunks must be non-empty"
            raise ValueError(err)
        if k1 < 0:
            err = f"k1 must be >= 0; got {k1}"
            raise ValueError(err)
        if not (0.0 <= b <= 1.0):
            err = f"b must be in [0, 1]; got {b}"
            raise ValueError(err)
        self._k1 = k1
        self._b = b
        self._chunks = list(chunks)
        self._tokens: list[list[str]] = [tokenize(c.text) for c in self._chunks]
        self._doc_lengths: list[int] = [len(toks) for toks in self._tokens]
        self._term_freqs: list[Counter[str]] = [Counter(toks) for toks in self._tokens]
        self._avg_dl: float = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        # Inverted index: term -> set of doc indices containing it.
        self._postings: dict[str, set[int]] = {}
        for doc_idx, freqs in enumerate(self._term_freqs):
            for term in freqs:
                self._postings.setdefault(term, set()).add(doc_idx)
        # IDF cache: idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
        n_docs = len(self._chunks)
        self._idf: dict[str, float] = {}
        for term, doc_set in self._postings.items():
            df = len(doc_set)
            self._idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    @property
    def k1(self) -> float:
        return self._k1

    @property
    def b(self) -> float:
        return self._b

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)

    def _score_doc(self, query_terms: Iterable[str], doc_idx: int) -> float:
        dl = self._doc_lengths[doc_idx]
        if dl == 0:
            return 0.0
        freqs = self._term_freqs[doc_idx]
        score = 0.0
        for term in query_terms:
            tf = freqs.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            length_norm = 1.0 - self._b + self._b * (dl / self._avg_dl)
            denom = tf + self._k1 * length_norm
            score += idf * (tf * (self._k1 + 1.0)) / denom
        return score

    async def retrieve(self, query: str, *, top_k: int = 5) -> tuple[RetrievalResult, ...]:
        """Return up to ``top_k`` chunks ranked by BM25 score."""
        if not query:
            err = "BM25Retriever.retrieve: query must be non-empty"
            raise ValueError(err)
        if top_k <= 0:
            err = f"top_k must be >= 1; got {top_k}"
            raise ValueError(err)

        query_terms = tokenize(query)
        # Restrict to docs that contain at least one query term — for
        # large corpora this is much cheaper than scoring every doc.
        candidates: set[int] = set()
        for term in query_terms:
            postings = self._postings.get(term)
            if postings is not None:
                candidates.update(postings)
        if not candidates:
            return ()

        scored = [(doc_idx, self._score_doc(query_terms, doc_idx)) for doc_idx in candidates]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return tuple(
            RetrievalResult(chunk=self._chunks[doc_idx], score=score)
            for doc_idx, score in scored[:top_k]
            if score > 0.0
        )
