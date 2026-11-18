"""The load guard: a model whose weights were not all filled from the checkpoint is refused."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from strata_forge.core.errors import ForgeError
from strata_forge.training.loading import LoadRefusedError, check_loading_info

if TYPE_CHECKING:
    from collections.abc import Sequence

_HEAD = ("classifier.", "score.", "head.")


def _info(**lists: Sequence[object]) -> dict[str, Sequence[object]]:
    return {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
        **lists,
    }


def test_a_clean_load_passes_and_reports_nothing_ignored() -> None:
    assert check_loading_info(_info(), head_prefixes=_HEAD) == []


def test_a_missing_head_is_refused_by_name() -> None:
    # The headline case: the head would be re-initialised at random and still answer.
    with pytest.raises(
        LoadRefusedError, match=r"missing weights: classifier\.bias, classifier\.weight"
    ):
        check_loading_info(
            _info(missing_keys=["classifier.weight", "classifier.bias"]), head_prefixes=_HEAD
        )


def test_a_shape_mismatch_is_refused_by_key() -> None:
    mismatch = [("classifier.weight", (2, 8), (3, 8))]
    with pytest.raises(LoadRefusedError, match=r"mismatched shape: classifier\.weight"):
        check_loading_info(_info(mismatched_keys=mismatch), head_prefixes=_HEAD)


def test_load_errors_are_refused() -> None:
    with pytest.raises(LoadRefusedError, match="load errors: bad tensor"):
        check_loading_info(_info(error_msgs=["bad tensor"]), head_prefixes=_HEAD)


def test_ignorable_unexpected_keys_are_returned_not_refused() -> None:
    extras = ["bert.pooler.dense.weight", "bert.embeddings.position_ids"]
    assert check_loading_info(_info(unexpected_keys=extras), head_prefixes=_HEAD) == sorted(extras)


def test_an_unexpected_head_tensor_is_refused() -> None:
    with pytest.raises(
        LoadRefusedError, match=r"head tensors the model did not load: score\.weight"
    ):
        check_loading_info(_info(unexpected_keys=["score.weight"]), head_prefixes=_HEAD)


def test_a_long_list_is_quoted_in_part() -> None:
    keys = [f"layer.{i}.weight" for i in range(20)]
    with pytest.raises(LoadRefusedError, match=r"\(and 12 more\)"):
        check_loading_info(_info(missing_keys=list(keys)), head_prefixes=_HEAD)


def test_the_refusal_is_a_forge_error() -> None:
    assert issubclass(LoadRefusedError, ForgeError)
