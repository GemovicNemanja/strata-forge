"""Whether two split specs can share rows."""

from __future__ import annotations

import pytest

from strata_forge.pipelines._split_overlap import may_overlap


@pytest.mark.parametrize(
    ("train", "evaluation"),
    [
        ("train", "validation"),
        ("train[:80%]", "train[80%:]"),
        ("train[:80%]", "train[-20%:]"),
        ("train[:100]", "train[100:200]"),
        ("train[:50%](closest)", "train[50%:](closest)"),
        ("train+test", "validation"),
        ("train[:0]", "train"),
    ],
)
def test_disjoint_specs_pass(train: str, evaluation: str) -> None:
    assert not may_overlap(train, evaluation)


@pytest.mark.parametrize(
    ("train", "evaluation"),
    [
        ("train", "train"),
        ("train", "train[:100%]"),
        ("train[:80%]", "train[70%:]"),
        ("train", "train[80%:]"),
        ("train[:100]", "train[50:]"),
        ("train+validation", "validation[:10%]"),
        # Undecidable from the specs alone: refused.
        ("train[:80%]", "train[1000:]"),
        ("train[:100]", "train[-50:]"),
        ("train[:50%](closest)", "train[50%:](pct1_dropremainder)"),
        ("train[:10%:]", "validation"),
        ("train", "train[:50%"),
        ("train[10:20%]", "train[50%:]"),
    ],
)
def test_overlapping_or_undecidable_specs_are_refused(train: str, evaluation: str) -> None:
    assert may_overlap(train, evaluation)
