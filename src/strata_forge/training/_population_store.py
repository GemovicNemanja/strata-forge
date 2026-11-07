"""Private bounded SQLite records; no model imports, pickle or caller SQL."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from strata_forge.training.population import POPULATION_DISK_RESERVE, PopulationError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


@dataclass(frozen=True)
class PopulationRecord:
    source_ordinal: int
    item_id: str
    input_ids: tuple[int, ...]
    token_type_ids: tuple[int, ...] | None
    marker_positions: tuple[int, ...]
    premise_ids: tuple[str, ...]
    targets_json: str


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(65_536):
            digest.update(chunk)
    return digest.hexdigest()


def disk_bytes(directory: Path) -> int:
    total = 0
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            raise PopulationError("population_cache_invalid")
        total += path.stat().st_size
    return total


def write_private(directory: Path, name: str, value: bytes, budget: int) -> None:
    temporary = directory / (name + ".writing")
    if disk_bytes(directory) + len(value) > budget:
        raise PopulationError("population_limit")
    with temporary.open("xb") as handle:
        temporary.chmod(0o600)
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(directory / name)


def _pack(values: tuple[int, ...]) -> bytes:
    if any(type(value) is not int or not 0 <= value <= 2**32 - 1 for value in values):
        raise PopulationError("population_cache_invalid")
    return struct.pack(f"<{len(values)}I", *values)


def _unpack(value: bytes) -> tuple[int, ...]:
    if type(value) is not bytes or len(value) % 4:
        raise PopulationError("population_cache_invalid")
    return struct.unpack(f"<{len(value) // 4}I", value)


class PopulationStore:
    def __init__(self, directory: Path, budget: int) -> None:
        self.directory, self.budget = directory, budget
        self.high_water = disk_bytes(directory)
        self.connection = sqlite3.connect(directory / "cache.sqlite", check_same_thread=False)
        try:
            (directory / "cache.sqlite").chmod(0o600)
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA cache_size=-2048")
            # Fixed indexed statements do not sort or create temporary B-trees.
            self.connection.execute("PRAGMA temp_store=MEMORY")
            self.connection.execute(
                f"PRAGMA max_page_count={(budget - POPULATION_DISK_RESERVE) // 8192}"
            )
            self.connection.execute(
                "CREATE TABLE seen(item_id TEXT PRIMARY KEY, source_ordinal INTEGER NOT NULL UNIQUE)"
            )
            self.connection.execute(
                "CREATE TABLE rows(selected_ordinal INTEGER PRIMARY KEY, source_ordinal INTEGER NOT NULL UNIQUE,item_id TEXT NOT NULL UNIQUE,ids BLOB NOT NULL,types BLOB,markers BLOB NOT NULL,pids TEXT NOT NULL,targets TEXT NOT NULL)"
            )
            self.connection.commit()
        except Exception:
            self.connection.close()
            raise
        self.selected = 0

    def write(self, rows: Iterable[tuple[int, str, PopulationRecord | None]]) -> None:
        with self.connection:
            for ordinal, item_id, record in rows:
                self.connection.execute("INSERT INTO seen VALUES(?,?)", (item_id, ordinal))
                if record is not None:
                    self.connection.execute(
                        "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?)",
                        (
                            self.selected,
                            ordinal,
                            item_id,
                            _pack(record.input_ids),
                            None if record.token_type_ids is None else _pack(record.token_type_ids),
                            _pack(record.marker_positions),
                            canonical_json(record.premise_ids).decode(),
                            record.targets_json,
                        ),
                    )
                    self.selected += 1
                self.high_water = max(self.high_water, disk_bytes(self.directory))
                if self.high_water > self.budget:
                    raise PopulationError("population_limit")
        self.high_water = max(self.high_water, disk_bytes(self.directory))
        if self.high_water > self.budget:
            raise PopulationError("population_limit")

    def close(self) -> None:
        self.connection.close()
        self.high_water = max(self.high_water, disk_bytes(self.directory))


class PopulationReader:
    def __init__(self, directory: Path, *, expected: int, max_length: int) -> None:
        self.connection = sqlite3.connect(
            (directory / "cache.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
        )
        self.connection.execute("PRAGMA cache_size=-2048")
        self.max_length = max_length
        if self.connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0] != expected:
            self.close()
            raise PopulationError("population_cache_invalid")

    def get(self, ordinal: int) -> PopulationRecord:
        row = self.connection.execute(
            "SELECT source_ordinal,item_id,ids,types,markers,pids,targets FROM rows WHERE selected_ordinal=?",
            (ordinal,),
        ).fetchone()
        if row is None:
            raise PopulationError("population_cache_invalid")
        source, item_id, ids, types, markers, pids, targets = row
        tokens, positions = _unpack(ids), _unpack(markers)
        token_types = None if types is None else _unpack(types)
        if (
            not tokens
            or len(tokens) > self.max_length
            or (token_types is not None and len(token_types) != len(tokens))
            or any(value >= len(tokens) for value in positions)
        ):
            raise PopulationError("population_cache_invalid")
        premise_ids: object = json.loads(pids)
        if (
            type(premise_ids) is not list
            or any(type(pid) is not str for pid in cast("list[object]", premise_ids))
            or len(positions) != len(cast("list[object]", premise_ids))
        ):
            raise PopulationError("population_cache_invalid")
        return PopulationRecord(
            source,
            item_id,
            tokens,
            token_types,
            positions,
            tuple(cast("list[str]", premise_ids)),
            targets,
        )

    def close(self) -> None:
        self.connection.close()
