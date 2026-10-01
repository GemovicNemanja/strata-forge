"""Secret redaction: one implementation for every place a secret could surface.

A process that holds a credential also writes text that other people read: an error message, a
progress caption, a column of a results file, a console log relayed to a browser. A
:class:`Redactor` removes from such text the specific secrets it was given, in every encoding a
log is likely to carry them, plus anything shaped like a well-known credential, whoever it
belongs to. :meth:`Redactor.stream` does the same for text that arrives in pieces, including a
secret split across two of them.

Each maximal run of redacted characters becomes one fixed :data:`PLACEHOLDER`, so the output
never encodes how long a secret was. Streamed output is identical to redacting the whole text at
once, which is the property that makes the stream trustworthy: a piece boundary can move what is
released when, never what is released.

Two limits are deliberate. Text is matched as given, never Unicode-normalised, because the
output is the input with spans replaced and normalising would rewrite everything else; instead a
value is matched in each of its four normal forms. And a token longer than its pattern's bound
is redacted for its first ``max_chars`` characters only, which is enough to make it unusable.
"""

from __future__ import annotations

import base64
import json
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from strata_forge.core.errors import ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "DEFAULT_PATTERNS",
    "MIN_SECRET_CHARS",
    "PLACEHOLDER",
    "RedactingStream",
    "Redactor",
    "TokenPattern",
]

PLACEHOLDER = "***"
# A shorter value would be found inside ordinary words, and redacting those both mangles the text
# and, by where the placeholders land, describes the secret.
MIN_SECRET_CHARS = 8
# A value form longer than this is matched as overlapping fragments rather than whole. Every
# position of a full occurrence lies in some fragment, so a full occurrence is still covered end to
# end; a partial print of a long secret (a clipped log) is covered for all but its edges, where
# matching the whole form would miss it entirely; and the stream's hold-back stays small whatever
# the secrets' length.
_FRAGMENT_CHARS = 64
_FRAGMENT_STEP = 32
# How far a private-key block with no footer is redacted past its header. Larger than the PEM
# encoding of any key in practical use, so a block whose footer was clipped stays covered.
_PEM_BODY_CHARS = 16_384
_PEM_HEADER = re.compile(r"-----BEGIN[A-Z0-9 ]{0,64}PRIVATE KEY(?: BLOCK)?-----")
_PEM_HEADER_MAX = 10 + 64 + 11 + 6 + 5
_PEM_FOOTER = re.compile(r"-----END[A-Z0-9 ]{0,64}PRIVATE KEY(?: BLOCK)?-----")
_PEM_FOOTER_MAX = 8 + 64 + 11 + 6 + 5
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_PERCENT_ESCAPE = re.compile(r"%[0-9A-F]{2}")
_URLSAFE = str.maketrans("+/", "-_")
_NORMAL_FORMS: tuple[Literal["NFC", "NFD", "NFKC", "NFKD"], ...] = ("NFC", "NFD", "NFKC", "NFKD")
_SPAN_GROUP = "secret"


@dataclass(frozen=True, slots=True)
class TokenPattern:
    """A credential shape, redacted wherever it appears.

    ``max_chars`` bounds how far one match reaches from where it starts, lookahead included. It
    is what lets a stream decide that text is final, so a pattern must never match further: a
    match that does raises :class:`~strata_forge.core.errors.ValidationError` rather than being
    trusted. A pattern may not use lookbehind. When the regex has a group named ``secret``, only
    that group is redacted and the rest of the match stays readable.
    """

    name: str
    regex: re.Pattern[str]
    max_chars: int


@dataclass(frozen=True, slots=True, repr=False)
class _Rules:
    """What a redactor matches, shared read-only by every stream it opens."""

    patterns: tuple[TokenPattern, ...]
    needles: tuple[str, ...]
    # The longest stretch one match can span. Text this close to the end of what a stream has
    # received may still turn out to be part of a secret, so the stream holds it back.
    max_len: int


def _pattern(name: str, regex: str, max_chars: int) -> TokenPattern:
    return TokenPattern(name, re.compile(regex), max_chars)


DEFAULT_PATTERNS: tuple[TokenPattern, ...] = (
    _pattern("huggingface", r"hf_[A-Za-z0-9]{8,256}", 3 + 256),
    # OpenAI (`sk-`, `sk-proj-`) and Anthropic (`sk-ant-`) keys share the prefix.
    _pattern("openai-anthropic", r"sk-[A-Za-z0-9_-]{16,256}", 3 + 256),
    # Classic, OAuth, user-to-server, server-to-server and refresh tokens.
    _pattern("github", r"gh[opusr]_[A-Za-z0-9]{20,255}", 4 + 255),
    _pattern("github-fine-grained", r"github_pat_[A-Za-z0-9_]{20,255}", 11 + 255),
    _pattern("aws-access-key-id", r"(?:AKIA|ASIA)[0-9A-Z]{16}", 20),
    # Header, dot, then payload and signature as one run: a payload longer than the bound is still
    # caught by its start, where a separately bounded segment would fail to match at all.
    _pattern("jwt", r"eyJ[A-Za-z0-9_-]{8,256}\.[A-Za-z0-9_.-]{8,256}", 3 + 256 + 1 + 256),
    # `scheme://user:password@host`: the scheme and host stay readable.
    _pattern(
        "url-userinfo",
        rf"://(?P<{_SPAN_GROUP}>[^\s/@:]{{1,128}}:[^\s/@]{{1,256}})@",
        3 + 128 + 1 + 256 + 1,
    ),
    # The scheme word stays readable; only the credential after it is redacted.
    _pattern(
        "bearer",
        rf"(?i:bearer)\s{{1,8}}(?P<{_SPAN_GROUP}>[A-Za-z0-9._~+/=-]{{8,512}})",
        6 + 8 + 512,
    ),
)


def _base64_forms(value: str) -> set[str]:
    """The base64 characters a value contributes wherever it sits in a larger encoded blob.

    Base64 maps three bytes to four characters, so the characters a value produces depend on its
    byte offset modulo three, and the ones at its edges also depend on its neighbours. For each
    of the three alignments, keep exactly the characters that only the value's own bits determine:
    those match whatever surrounds the value, as in ``Basic base64(user:token)``.
    """
    raw = value.encode()
    forms: set[str] = set()
    for offset in range(3):
        encoded = base64.b64encode(bytes(offset) + raw + bytes(3)).decode("ascii")
        stable = encoded[-(-8 * offset // 6) : 8 * (offset + len(raw)) // 6]
        forms.add(stable)
        forms.add(stable.translate(_URLSAFE))
    return forms


def _url_forms(value: str) -> set[str]:
    """Percent-encodings: the standard ones, one that escapes every non-alphanumeric character
    (which a URL may legally do even to ``_`` or ``-``), and each with lower-case hex."""
    escape_all = "".join(
        char if char.isascii() and char.isalnum() else "".join(f"%{b:02X}" for b in char.encode())
        for char in value
    )
    forms = {
        urllib.parse.quote(value, safe=""),
        urllib.parse.quote_plus(value, safe=""),
        urllib.parse.quote(value),
        escape_all,
    }
    return forms | {_PERCENT_ESCAPE.sub(lambda m: m.group(0).lower(), form) for form in forms}


def _value_forms(value: str) -> set[str]:
    """Every spelling of ``value`` this redactor matches."""
    # The text is never normalised, so a value is matched in every normal form it could be
    # printed in instead.
    forms = {value} | {unicodedata.normalize(form, value) for form in _NORMAL_FORMS}
    # A multi-line secret (a private key, a known_hosts file) is printed line by line as often as
    # whole, and not always with the line endings it was stored with.
    forms.update(line.strip() for line in _LINE_BREAK.split(value))
    forms |= _base64_forms(value) | _url_forms(value)
    # As it appears inside a JSON string (a structured progress line) or a Python repr (an
    # exception rendered into an error column).
    forms.add(json.dumps(value)[1:-1])
    forms.add(json.dumps(value, ensure_ascii=False)[1:-1])
    forms.add(repr(value)[1:-1])
    return {form for form in forms if len(form) >= MIN_SECRET_CHARS}


def _merge(spans: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sorted, with overlapping and touching spans joined."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _needles(forms: Iterable[str]) -> tuple[str, ...]:
    needles: set[str] = set()
    for form in forms:
        if len(form) <= _FRAGMENT_CHARS:
            needles.add(form)
            continue
        last = len(form) - _FRAGMENT_CHARS
        needles.update(form[i : i + _FRAGMENT_CHARS] for i in range(0, last, _FRAGMENT_STEP))
        needles.add(form[last:])
    return tuple(sorted(needles))


class Redactor:
    """Removes given secrets and credential-shaped text from strings and streams.

    ``values`` are the secrets a caller holds. Each must be at least :data:`MIN_SECRET_CHARS`
    long: a shorter one would match inside ordinary text, and the redactor refuses it rather than
    quietly skipping it. ``patterns`` defaults to :data:`DEFAULT_PATTERNS`; pass ``()`` to redact
    the values alone.

    Never logs, and never repeats in an error message, anything it was given.
    """

    __slots__ = ("_rules", "_value_count")

    def __init__(
        self,
        values: Iterable[str] = (),
        *,
        patterns: Iterable[TokenPattern] = DEFAULT_PATTERNS,
    ) -> None:
        forms: set[str] = set()
        count = 0
        for value in values:
            if len(value) < MIN_SECRET_CHARS:
                msg = (
                    f"refusing a redaction value shorter than {MIN_SECRET_CHARS} characters: "
                    "it would also match ordinary text"
                )
                raise ValidationError(msg)
            forms |= _value_forms(value)
            count += 1
        needles = _needles(forms)
        chosen = tuple(patterns)
        max_len = max(
            [_PEM_HEADER_MAX, _PEM_FOOTER_MAX]
            + [len(needle) for needle in needles]
            + [pattern.max_chars for pattern in chosen]
        )
        self._rules = _Rules(patterns=chosen, needles=needles, max_len=max_len)
        self._value_count = count

    def __repr__(self) -> str:
        return f"Redactor(values={self._value_count}, patterns={len(self._rules.patterns)})"

    @property
    def max_len(self) -> int:
        """The longest span one match can cover; a stream holds back ``max_len - 1`` chars."""
        return self._rules.max_len

    def redact(self, text: str) -> str:
        """Return ``text`` with every secret and credential-shaped span replaced."""
        stream = RedactingStream(self._rules)
        return stream.feed(text) + stream.flush()

    def stream(self) -> RedactingStream:
        """A stateful redactor for one stream of text that arrives in pieces."""
        return RedactingStream(self._rules)


class RedactingStream:
    """Redacts one stream of text piece by piece; obtained from :meth:`Redactor.stream`.

    :meth:`feed` returns what is final so far and holds back the last ``max_len - 1``
    characters, the only ones a later piece can still turn into part of a secret.
    :meth:`flush` releases the rest when the stream ends and readies the object for a new one.
    The concatenated output equals :meth:`Redactor.redact` of the concatenated input.

    A reader that LOSES text between two pieces (a console read that reports dropped bytes)
    must call :meth:`gap` there. A secret the hole cut in two cannot be recognised from either
    side, so the held tail and the first ``max_len - 1`` characters after the hole are replaced;
    a private-key block open at the hole stays open.
    """

    __slots__ = (
        "_blocks",
        "_buf",
        "_buf_start",
        "_carried",
        "_emitted",
        "_footer_pos",
        "_header_pos",
        "_literal_pos",
        "_pattern_pos",
        "_prev_covered",
        "_received",
        "_rules",
    )

    def __init__(self, rules: _Rules) -> None:
        self._rules = rules
        self._reset()

    def __repr__(self) -> str:
        return "RedactingStream()"

    def _reset(self) -> None:
        # Positions are absolute character offsets into the whole stream.
        self._buf = ""  # what has been received from `_buf_start` on
        self._buf_start = 0
        self._received = 0
        self._emitted = 0  # everything before this has been released
        # Redacted spans already found that reach past `_emitted`, merged.
        self._carried: list[tuple[int, int]] = []
        self._prev_covered = False  # whether the last released character was redacted
        # Where each search resumes. Every match starting before `_emitted` has been found.
        self._pattern_pos = [0] * len(self._rules.patterns)
        self._literal_pos = 0
        self._header_pos = 0
        self._footer_pos = 0
        self._blocks: list[tuple[int, int]] = []  # open key blocks: (header start, header end)

    def feed(self, chunk: str) -> str:
        """Add the next piece; return the output that is now final."""
        self._buf += chunk
        self._received += len(chunk)
        return self._advance(final=False)

    def flush(self) -> str:
        """End the stream; return the rest of the output."""
        out = self._advance(final=True)
        self._reset()
        return out

    def gap(self) -> str:
        """Mark lost text before the next piece; return the output up to the hole."""
        self._carried.append((self._emitted, self._received + self._rules.max_len - 1))
        return self._advance(final=True, keep_blocks=True)

    def _advance(self, *, final: bool, keep_blocks: bool = False) -> str:
        n = self._received
        # A match starting before `decided` is final: everything it can depend on has arrived.
        decided = n if final else max(self._emitted, n - self._rules.max_len + 1)
        spans = self._carried + self._pattern_spans(decided) + self._literal_spans(decided)
        spans += self._key_block_spans(decided, close_open=final and not keep_blocks)
        spans.sort()
        out = self._release(spans, decided)
        self._carried = _merge((max(start, decided), end) for start, end in spans if end > decided)
        self._buf = self._buf[decided - self._buf_start :]
        self._buf_start = decided
        return out

    def _pattern_spans(self, decided: int) -> list[tuple[int, int]]:
        base, buf = self._buf_start, self._buf
        spans: list[tuple[int, int]] = []
        for index, pattern in enumerate(self._rules.patterns):
            group = _SPAN_GROUP if _SPAN_GROUP in pattern.regex.groupindex else 0
            pos = self._pattern_pos[index]
            while (match := pattern.regex.search(buf, pos - base)) is not None:
                if match.start() + base >= decided:
                    break
                if match.end() - match.start() > pattern.max_chars:
                    msg = f"token pattern {pattern.name!r} matched beyond its max_chars"
                    raise ValidationError(msg)
                start, end = match.span(group)
                if end > start:
                    spans.append((start + base, end + base))
                pos = max(match.end(), match.start() + 1) + base
            self._pattern_pos[index] = max(pos, decided)
        return spans

    def _literal_spans(self, decided: int) -> list[tuple[int, int]]:
        base, buf = self._buf_start, self._buf
        spans: list[tuple[int, int]] = []
        for needle in self._rules.needles:
            hit = buf.find(needle, self._literal_pos - base)
            while hit != -1 and hit + base < decided:
                spans.append((hit + base, hit + base + len(needle)))
                hit = buf.find(needle, hit + 1)
        self._literal_pos = decided
        return spans

    def _key_block_spans(self, decided: int, *, close_open: bool) -> list[tuple[int, int]]:
        """Spans of private-key blocks: header to the first footer within the window after it.

        With no footer in the window the block runs to the window's end, so a key whose footer
        was clipped off is still covered. Footers are taken in order, so each one closes every
        block that is still waiting for it.
        """
        base, buf = self._buf_start, self._buf
        spans: list[tuple[int, int]] = []
        pos = self._header_pos
        while (match := _PEM_HEADER.search(buf, pos - base)) is not None:
            if match.start() + base >= decided:
                break
            self._blocks.append((match.start() + base, match.end() + base))
            pos = match.end() + base
        self._header_pos = max(pos, decided)

        pos = self._footer_pos
        while (match := _PEM_FOOTER.search(buf, pos - base)) is not None:
            footer = match.start() + base
            if footer >= decided:
                break
            waiting: list[tuple[int, int]] = []
            for start, header_end in self._blocks:
                if header_end > footer:
                    waiting.append((start, header_end))
                elif footer <= header_end + _PEM_BODY_CHARS:
                    spans.append((start, match.end() + base))
                else:
                    spans.append((start, header_end + _PEM_BODY_CHARS))
            self._blocks = waiting
            pos = match.end() + base
        self._footer_pos = max(pos, decided)

        waiting = []
        for start, header_end in self._blocks:
            window_end = header_end + _PEM_BODY_CHARS
            if decided > window_end or close_open:
                spans.append((start, min(window_end, self._received)))
            else:
                # Everything from the header up to `decided` is redacted whatever comes next:
                # the block ends at a footer not yet seen, or at the end of its window.
                waiting.append((start, header_end))
                spans.append((start, decided))
        self._blocks = waiting
        return spans

    def _release(self, spans: list[tuple[int, int]], until: int) -> str:
        """Emit ``[_emitted, until)`` from sorted ``spans``: raw text, each maximal redacted run
        as one placeholder."""
        base, buf = self._buf_start, self._buf
        cursor = self._emitted
        covered = self._prev_covered
        parts: list[str] = []
        for span_start, span_end in spans:
            start, end = max(span_start, cursor), min(span_end, until)
            if end <= start:
                continue
            if start > cursor:
                parts.append(buf[cursor - base : start - base])
                covered = False
            if not covered:
                parts.append(PLACEHOLDER)
                covered = True
            cursor = end
        if cursor < until:
            parts.append(buf[cursor - base : until - base])
            covered = False
        self._emitted = until
        self._prev_covered = covered
        return "".join(parts)
