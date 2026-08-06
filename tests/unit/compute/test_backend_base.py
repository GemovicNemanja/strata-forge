"""Unit tests for the `strata_forge.compute.backends.base` workdir-confinement guard.

`safe_workdir_relpath` is the security boundary that keeps `Backend.read_file`
inside a job's working directory. These lock its lexical contract directly (the
backends exercise it indirectly) so a future refactor can't silently widen it.
"""

from __future__ import annotations

import pytest

from strata_forge.compute import safe_workdir_relpath

# Workdir-relative inputs that MUST be accepted, with their normalized form.
ALLOWED = [
    ("progress.jsonl", "progress.jsonl"),
    ("a/b", "a/b"),
    ("a/b/c.json", "a/b/c.json"),
    ("./x", "x"),
    ("a/../b", "b"),  # stays within
    ("a/./b", "a/b"),
]

# Inputs that MUST be rejected (escape, absolute, or malformed).
REJECTED = [
    "",
    "/etc/passwd",
    "//etc/passwd",
    "..",
    "../x",
    "../../etc/passwd",
    "a/../../x",  # nets one level above the workdir
    "./../x",
    "x/./../../y",
    "foo/../../bar",
    "a\\b",  # backslash separator (POSIX normalizer would miss it)
    "..\\..\\x",
    "C:\\windows",
    "\x00",
    "a/\x00/b",
]


@pytest.mark.parametrize(("raw", "normalized"), ALLOWED)
def test_allows_in_workdir_paths(raw: str, normalized: str) -> None:
    assert safe_workdir_relpath(raw) == normalized


@pytest.mark.parametrize("raw", REJECTED)
def test_rejects_escapes_and_malformed(raw: str) -> None:
    with pytest.raises(ValueError, match="workdir"):
        safe_workdir_relpath(raw)
