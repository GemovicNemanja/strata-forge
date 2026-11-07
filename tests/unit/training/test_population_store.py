"""Safe private cache codec and quota counterexamples with fictional local bytes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from strata_forge.training._population_store import (
    PopulationReader,
    PopulationRecord,
    PopulationStore,
    _pack,  # pyright: ignore[reportPrivateUsage] — direct malformed packed-codec witness
    _unpack,  # pyright: ignore[reportPrivateUsage] — direct truncated packed-codec witness
    disk_bytes,
    write_private,
)
from strata_forge.training.population import PopulationError

if TYPE_CHECKING:
    from pathlib import Path


def test_packed_integer_codec_rejects_bool_range_and_truncation() -> None:
    for values in ((True,), (-1,), (2**32,)):
        with pytest.raises(PopulationError):
            _pack(values)
    with pytest.raises(PopulationError):
        _unpack(b"bad")
    assert _unpack(_pack((0, 2**32 - 1))) == (0, 2**32 - 1)


def test_private_output_budget_and_symlink_refuse(tmp_path: Path) -> None:
    with pytest.raises(PopulationError):
        write_private(tmp_path, "receipt", b"1234", 3)
    (tmp_path / "link").symlink_to(tmp_path / "absent")
    with pytest.raises(PopulationError):
        disk_bytes(tmp_path)


def test_reader_checks_count_and_missing_ordinal(tmp_path: Path) -> None:
    store = PopulationStore(tmp_path, 32 * 1024**2)
    store.write([(0, "row", PopulationRecord(0, "row", (1, 2), None, (), (), "[1.0,0.0]"))])
    store.close()
    with pytest.raises(PopulationError):
        PopulationReader(tmp_path, expected=2, max_length=16)
    reader = PopulationReader(tmp_path, expected=1, max_length=16)
    with pytest.raises(PopulationError):
        reader.get(1)
    reader.close()
    reader = PopulationReader(tmp_path, expected=1, max_length=1)
    with pytest.raises(PopulationError):
        reader.get(0)
    reader.close()


@pytest.mark.parametrize(("positions", "pids"), [((5,), ("p1",)), ((0,), ()), ((), ("p1",))])
def test_reader_does_not_trust_malformed_cached_geometry(
    tmp_path: Path, positions: tuple[int, ...], pids: tuple[str, ...]
) -> None:
    store = PopulationStore(tmp_path, 32 * 1024**2)
    store.write([(0, "row", PopulationRecord(0, "row", (1, 2), None, positions, pids, "{}"))])
    store.close()
    reader = PopulationReader(tmp_path, expected=1, max_length=16)
    with pytest.raises(PopulationError):
        reader.get(0)
    reader.close()
