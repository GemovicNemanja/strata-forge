"""How a trained verifier's input is rendered: ONE definition for the trainer and the scorer.

A ``seq_cls`` artifact records its rendering in ``strata_verifier.json`` (the ``input`` block).
The trainer renders every training row with :func:`render_row` and :func:`encode_rows`, and the
scorer (``hf_classify`` with ``artifact: strata_verifier``) renders every scored row with the
same two functions over the :class:`Rendering` it reads back from that file. Nothing about the
rendering is taken from a run's parameters at scoring time, so a verifier is scored exactly as it
was trained.

The rendering:

- **Premises** are a non-empty list of ``{pid, text}`` objects with pids ``p1..pN`` in order (a
  string or a list of strings is refused). Under ``premise_format="ids"`` each is the line
  ``"{pid}: {text}"``, under ``plain`` the text alone; lines are joined by ``"\\n"``.
- **An encoder** sees the tokenizer's pair encoding of ``(premises, hypothesis)``.
- **A decoder** sees one text, ``decoder_template`` filled in one pass (``{premises}`` and
  ``{hypothesis}``), with the tokenizer's own special tokens; its answer is read at the final
  token.
- **Support markers** are each premise id's first character; its token is the first one of the
  first sequence whose character span covers it.

Pure Python over a tokenizer object; no torch.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from strata_forge.training.dataset_format import DatasetFormatError

__all__ = [
    "PREMISE_SEPARATOR",
    "RenderedRow",
    "Rendering",
    "encode_rows",
    "fill_template",
    "marker_tokens",
    "render_premises",
    "render_row",
]

#: Between two premise lines.
PREMISE_SEPARATOR = "\n"
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]+)\}")


@dataclass(frozen=True, slots=True)
class Rendering:
    """The rendering a verifier was trained with, as ``strata_verifier.json`` records it."""

    encoding: Literal["pair", "template"]
    premise_format: Literal["ids", "plain"]
    decoder_template: str | None
    max_length: int

    @property
    def decoder(self) -> bool:
        return self.encoding == "template"

    @classmethod
    def from_verifier(cls, verifier: Mapping[str, Any]) -> Rendering:
        """Read the ``input`` block back, refusing one this engine does not render the same way."""
        block: Any = verifier.get("input")
        if not isinstance(block, Mapping):
            msg = "strata_verifier.json has no input block"
            raise ValueError(msg)
        spec: Mapping[str, Any] = block  # pyright: ignore[reportUnknownVariableType] - JSON
        encoding: Any = spec.get("encoding")
        premise_format: Any = spec.get("premise_format")
        template: Any = spec.get("decoder_template")
        max_length: Any = spec.get("max_length")
        if encoding not in ("pair", "template") or premise_format not in ("ids", "plain"):
            msg = "strata_verifier.json names an encoding or premise format this engine lacks"
            raise ValueError(msg)
        if (encoding == "template") != isinstance(template, str):
            msg = "strata_verifier.json: a decoder template exactly when the encoding is template"
            raise ValueError(msg)
        if isinstance(template, str):
            # Read back under the trainer's own rules, and against the digest it recorded.
            found = sorted(_PLACEHOLDER_RE.findall(template))
            if found != ["hypothesis", "premises"]:
                msg = (
                    "strata_verifier.json: the decoder template must hold {premises} and "
                    "{hypothesis} exactly once each, and no other placeholder"
                )
                raise ValueError(msg)
            recorded: Any = spec.get("decoder_template_sha256")
            if recorded != hashlib.sha256(template.encode()).hexdigest():
                msg = (
                    "strata_verifier.json: the decoder template does not match its recorded digest"
                )
                raise ValueError(msg)
        if spec.get("premise_separator", PREMISE_SEPARATOR) != PREMISE_SEPARATOR:
            msg = "strata_verifier.json names a premise separator this engine does not render"
            raise ValueError(msg)
        if spec.get("special_tokens", True) is not True:
            msg = (
                "strata_verifier.json renders without special tokens; this engine always adds them"
            )
            raise ValueError(msg)
        if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
            msg = "strata_verifier.json has no positive max_length"
            raise ValueError(msg)
        return cls(
            encoding=encoding,
            premise_format=premise_format,
            decoder_template=template if encoding == "template" else None,
            max_length=max_length,
        )

    def to_verifier(self) -> dict[str, Any]:
        """The ``input`` block this rendering writes (``from_verifier`` reads it back)."""
        return {
            "encoding": self.encoding,
            "premise_format": self.premise_format,
            "premise_line": "{pid}: {text}" if self.premise_format == "ids" else "{text}",
            "premise_separator": PREMISE_SEPARATOR,
            "decoder_template": self.decoder_template,
            "special_tokens": True,
            "max_length": self.max_length,
        }


@dataclass(frozen=True, slots=True)
class RenderedRow:
    """One row as text: the first sequence, the second (encoders only), and the premise markers
    as ``(pid, character offset in text)``."""

    text: str
    pair: str | None
    markers: list[tuple[str, int]]


def render_premises(value: Any, premise_format: str) -> tuple[str, list[tuple[str, int]]]:
    """Render a ``premises`` cell and return ``(text, [(pid, first_char_offset), ...])``.

    A cell is a non-empty list of ``{pid, text}`` objects whose pids are ``p1``, ``p2``, ... in
    order; a bare string or a list of strings is refused, because a premise without its declared
    id cannot be supervised or audited. Under ``ids`` each premise is the line
    ``"{pid}: {text}"`` and its marker is the pid's first character; under ``plain`` the line is
    the text and there are no markers.
    """
    if isinstance(value, str | bytes) or not isinstance(value, Sequence) or not value:
        msg = "premises must be a non-empty list of {pid, text} objects"
        raise DatasetFormatError(msg)
    entries: list[tuple[str, str]] = []
    for i, item in enumerate(value):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType] - a dataset cell
        if not (isinstance(item, Mapping) and "pid" in item and "text" in item):
            msg = "each premise must be an object with pid and text"
            raise DatasetFormatError(msg)
        pid: Any = item["pid"]  # pyright: ignore[reportUnknownVariableType] - a dataset cell
        text: Any = item["text"]  # pyright: ignore[reportUnknownVariableType] - a dataset cell
        if not isinstance(text, str):
            msg = "each premise's text must be a string"
            raise DatasetFormatError(msg)
        if pid != f"p{i + 1}":
            msg = f"premise ids must be p1..pN in order; premise {i + 1} is not p{i + 1}"
            raise DatasetFormatError(msg)
        entries.append((str(pid), text))  # pyright: ignore[reportUnknownArgumentType] - checked equal to a str
    lines: list[str] = []
    markers: list[tuple[str, int]] = []
    offset = 0
    for pid, text in entries:
        line = f"{pid}: {text}" if premise_format == "ids" else text
        if premise_format == "ids":
            markers.append((pid, offset))
        lines.append(line)
        offset += len(line) + len(PREMISE_SEPARATOR)
    return PREMISE_SEPARATOR.join(lines), markers


def fill_template(template: str, premises: str, hypothesis: str) -> tuple[str, int]:
    """Fill the template in one pass; return the text and where the premises start in it."""
    start = -1
    pieces: list[str] = []
    cursor = 0
    length = 0
    for match in _PLACEHOLDER_RE.finditer(template):
        literal = template[cursor : match.start()]
        pieces.append(literal)
        length += len(literal)
        value = premises if match.group(1) == "premises" else hypothesis
        if match.group(1) == "premises":
            start = length
        pieces.append(value)
        length += len(value)
        cursor = match.end()
    pieces.append(template[cursor:])
    return "".join(pieces), start


def render_row(rendering: Rendering, premises: Any, hypothesis: Any) -> RenderedRow:
    """One row's text under ``rendering``; a malformed cell is a :class:`DatasetFormatError`."""
    text, markers = render_premises(premises, rendering.premise_format)
    if not isinstance(hypothesis, str):
        msg = "hypothesis must be a string"
        raise DatasetFormatError(msg)
    if rendering.decoder:
        filled, start = fill_template(rendering.decoder_template or "", text, hypothesis)
        return RenderedRow(filled, None, [(pid, start + off) for pid, off in markers])
    return RenderedRow(text, hypothesis, markers)


def encode_rows(tokenizer: Any, rows: Sequence[RenderedRow], **kwargs: Any) -> Any:
    """Tokenise rendered rows with special tokens and no truncation (``kwargs`` pass through)."""
    texts = [row.text for row in rows]
    pairs = [row.pair or "" for row in rows]
    decoder = any(row.pair is None for row in rows)
    return tokenizer(
        texts,
        *([] if decoder else [pairs]),
        add_special_tokens=True,
        truncation=False,
        **kwargs,
    )


def marker_tokens(
    markers: Sequence[tuple[str, int]],
    offsets: Sequence[tuple[int, int]],
    sequence_ids: Sequence[int | None],
) -> list[int] | None:
    """Each marker's token: the first token of the first sequence covering its character.

    ``None`` when some marker has no covering token.
    """
    positions: list[int] = []
    for _pid, char in markers:
        token = next(
            (
                j
                for j, (start, end) in enumerate(offsets)
                if start <= char < end and sequence_ids[j] == 0
            ),
            None,
        )
        if token is None:
            return None
        positions.append(token)
    return positions
