"""Whether two Hugging Face split specs can share rows.

A split spec is ``datasets``' read-instruction syntax: one or more ``+``-joined parts, each a split
name with an optional slice in rows or percent (``train``, ``train[:80%]``, ``train[70%:]``,
``train[100:200]``) and an optional rounding (``(closest)`` or ``(pct1_dropremainder)``).

:func:`may_overlap` answers ``True`` whenever sharing rows cannot be ruled out from the two specs
alone: an unparseable spec, a slice in rows against one in percent of the same split, a negative row
index, or two percent slices with different roundings. Callers that must keep two splits apart refuse
on ``True``. Pure, so it is checked before anything is loaded.

:func:`parse_single_split` is the strict subset a dataset recipe accepts: exactly one part, a split
name with at most a ``[start:end]`` slice in non-negative rows, no ``+``, no percent and no
rounding, so what a recipe reads is one named split and nothing a name could hide.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = ["SplitSlice", "may_overlap", "parse_single_split"]

_PART = re.compile(
    r"^(?P<name>[\w.-]+)"
    r"(?:\[(?:(?P<start>-?\d+)(?P<start_pct>%)?)?:(?:(?P<end>-?\d+)(?P<end_pct>%)?)?\])?"
    r"(?:\((?P<rounding>closest|pct1_dropremainder)\))?$"
)


@dataclass(frozen=True)
class _Part:
    name: str
    #: "all" (the whole split), "pct" or "rows"; None when the slice mixes units.
    unit: str | None
    start: float
    end: float
    rounding: str
    negative: bool


def _parse(spec: str) -> list[_Part] | None:
    parts: list[_Part] = []
    for raw in spec.split("+"):
        found = _PART.match(raw.strip())
        if found is None:
            return None
        start, end = found.group("start"), found.group("end")
        if "[" not in raw:
            parts.append(_Part(found.group("name"), "all", 0, math.inf, "", negative=False))
            continue
        pct = {
            bool(found.group(f"{side}_pct"))
            for side, value in (("start", start), ("end", end))
            if value is not None
        }
        unit = None if len(pct) > 1 else ("pct" if pct == {True} else "rows")
        top = 100.0 if unit == "pct" else math.inf
        lo = float(start) if start is not None else 0.0
        hi = float(end) if end is not None else top
        negative = lo < 0 or hi < 0
        if unit == "pct":
            lo = min(max(lo + 100 if lo < 0 else lo, 0.0), 100.0)
            hi = min(max(hi + 100 if hi < 0 else hi, 0.0), 100.0)
            negative = False
        parts.append(
            _Part(found.group("name"), unit, lo, hi, found.group("rounding") or "", negative)
        )
    return parts


def _parts_overlap(a: _Part, b: _Part) -> bool:
    if a.name != b.name:
        return False
    if a.unit is None or b.unit is None or a.negative or b.negative:
        return True  # undecidable from the specs alone
    if a.unit == "all" or b.unit == "all":
        other = b if a.unit == "all" else a
        return other.unit == "all" or other.start < other.end
    if a.unit != b.unit:
        return True  # rows against percent: depends on the split's length
    if a.unit == "pct" and a.rounding != b.rounding:
        return True  # different roundings can move a shared boundary
    return max(a.start, b.start) < min(a.end, b.end)


def may_overlap(first: str, second: str) -> bool:
    """Whether ``first`` and ``second`` can share rows; ``True`` when it cannot be decided."""
    a, b = _parse(first), _parse(second)
    if a is None or b is None:
        return True
    return any(_parts_overlap(x, y) for x in a for y in b)


@dataclass(frozen=True)
class SplitSlice:
    """One split read by name, optionally sliced to rows ``[start, end)``."""

    name: str
    start: int | None
    end: int | None


def parse_single_split(spec: str) -> SplitSlice | None:
    """``spec`` as one split and an optional row slice, or ``None`` when it is anything else.

    Parsed with the same read-instruction grammar as :func:`may_overlap`, then held to one part:
    ``train`` and ``train[100:200]`` parse; ``train+test``, ``train[:10%]``, ``train[-5:]`` and
    ``train(closest)`` do not.
    """
    if spec != spec.strip() or "+" in spec:
        return None
    parts = _parse(spec)
    if parts is None or len(parts) != 1:
        return None
    found = _PART.match(spec)
    if (
        found is None
        or found.group("rounding")
        or found.group("start_pct")
        or found.group("end_pct")
    ):
        return None
    start, end = found.group("start"), found.group("end")
    if (start or "").startswith("-") or (end or "").startswith("-"):
        return None
    lo = int(start) if start is not None else None
    hi = int(end) if end is not None else None
    if lo is not None and hi is not None and hi < lo:
        return None
    return SplitSlice(found.group("name"), lo, hi)
