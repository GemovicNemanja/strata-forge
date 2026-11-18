"""What a plugin wrote, checked and copied before anything is pushed.

The plugin's output directory is the plugin's: it may hold links, hard links to files it could
read, special files, or files that change after a check. So the runner never pushes it. Once the
plugin's processes are gone, :func:`stage_output` walks the directory without following any link
(each directory and file opened relative to its parent's descriptor with ``O_NOFOLLOW``), refuses
the whole output on a symbolic link, a special file or a file with more than one hard link, copies
every regular file into a fresh directory the runner owns, and checks each copy's SHA-256 against
what was read. Everything after that (the score check, the manifest, the push) reads the copy.

:func:`check_scores_file` is the runner's own check of a step scorer's ``scores.parquet``, done on
the copy and trusting nothing the plugin's process checked: the schema, one row per input step
with the exact ``(item_id, step_index)`` sequence and every kept column equal to the input's,
every score finite and inside its declared ``score_kind`` (a probability in [0, 1], never
clipped), a null score exactly when a declared reason is given.
"""

from __future__ import annotations

import hashlib
import os
import stat
from typing import TYPE_CHECKING, Any

from strata_forge.plugins.manifest import EntryPointDecl, PluginManifestError
from strata_forge.plugins.step_scoring import REASON_COLUMN, SCORE_COLUMN, Unscored, check_result

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["MAX_OUTPUT_FILES", "OutputError", "check_scores_file", "stage_output"]

#: The most files one plugin run may push.
MAX_OUTPUT_FILES = 10_000
_CHUNK = 1 << 20


class OutputError(Exception):
    """The plugin's output cannot be pushed."""


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while chunk := os.read(fd, _CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def _copy_file(src_dir: int, name: str, dest: Path, expected: os.stat_result) -> str:
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=src_dir)
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino)
        ):
            msg = f"the plugin output's {name} changed while it was read; nothing is pushed"
            raise OutputError(msg)
        out = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        digest = hashlib.sha256()
        try:
            while chunk := os.read(fd, _CHUNK):
                digest.update(chunk)
                os.write(out, chunk)
            os.fsync(out)
        finally:
            os.close(out)
    finally:
        os.close(fd)
    copied = os.open(dest, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        if _hash_fd(copied) != digest.hexdigest():
            msg = f"the copy of the plugin output's {name} does not match what was read"
            raise OutputError(msg)
    finally:
        os.close(copied)
    return digest.hexdigest()


def _walk(dir_fd: int, dest: Path, prefix: str, hashes: dict[str, str]) -> None:
    for entry in sorted(os.listdir(dir_fd)):
        info = os.stat(entry, dir_fd=dir_fd, follow_symlinks=False)
        relative = f"{prefix}{entry}"
        if stat.S_ISLNK(info.st_mode):
            msg = f"the plugin output holds a link ({relative}); nothing is pushed"
            raise OutputError(msg)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(entry, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dir_fd)
            try:
                (dest / entry).mkdir(mode=0o700)
                _walk(child, dest / entry, f"{relative}/", hashes)
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(info.st_mode):
            msg = f"the plugin output holds a special file ({relative}); nothing is pushed"
            raise OutputError(msg)
        if info.st_nlink != 1:
            msg = f"the plugin output's {relative} is a hard link; nothing is pushed"
            raise OutputError(msg)
        if len(hashes) >= MAX_OUTPUT_FILES:
            msg = f"the plugin wrote more than {MAX_OUTPUT_FILES} files; nothing is pushed"
            raise OutputError(msg)
        hashes[relative] = _copy_file(dir_fd, entry, dest / entry, info)


def stage_output(source: Path, staging: Path) -> dict[str, str]:
    """Copy the plugin's regular files from ``source`` into ``staging`` (which must not exist),
    refusing links, special files and hard links. Returns ``{relative path: sha256}``."""
    try:
        info = os.lstat(source)
    except FileNotFoundError:
        msg = "the plugin wrote no output directory; nothing is pushed"
        raise OutputError(msg) from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        msg = "the plugin output directory is a link or not a directory; nothing is pushed"
        raise OutputError(msg)
    staging.mkdir(mode=0o700)
    root = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        hashes: dict[str, str] = {}
        _walk(root, staging, "", hashes)
    finally:
        os.close(root)
    return hashes


def check_scores_file(
    entry: EntryPointDecl, scores_path: Path, steps_path: Path, keep: list[str]
) -> dict[str, Any]:
    """The scoring record for a staged ``scores.parquet`` checked against the input, or
    :class:`OutputError`. A count or identity mismatch is an infrastructure fault."""
    pa: Any = __import__("pyarrow")
    pq: Any = __import__("pyarrow.parquet", fromlist=["parquet"])
    if entry.score_kind is None:  # the manifest model refuses this; a typed guard
        msg = f"{entry.name!r} declares no score_kind"
        raise OutputError(msg)
    expected = pq.read_table(str(steps_path), columns=keep)
    try:
        written = pq.read_table(str(scores_path))
    except Exception as exc:  # pyarrow raises its own error types for a corrupt file
        msg = "infrastructure fault: the scores file cannot be read"
        raise OutputError(msg) from exc
    if written.column_names != [*keep, SCORE_COLUMN, REASON_COLUMN]:
        msg = f"infrastructure fault: the scores have the columns {written.column_names}"
        raise OutputError(msg)
    if written.num_rows != expected.num_rows:
        msg = (
            f"infrastructure fault: {written.num_rows} score rows for {expected.num_rows} input "
            "steps; nothing is pushed"
        )
        raise OutputError(msg)
    schema = written.schema
    if not schema.field(SCORE_COLUMN).type.equals(pa.float64()) or not schema.field(
        REASON_COLUMN
    ).type.equals(pa.string()):
        msg = "infrastructure fault: score must be float64 and reason a string"
        raise OutputError(msg)
    for name in keep:
        if not schema.field(name).type.equals(expected.schema.field(name).type) or not (
            written.column(name).equals(expected.column(name))
        ):
            msg = f"infrastructure fault: the scores' {name!r} column does not match the input"
            raise OutputError(msg)
    reasons = frozenset(entry.unscored_reasons)
    unscored: dict[str, int] = dict.fromkeys(entry.unscored_reasons, 0)
    scored = 0
    rows: list[tuple[str, int, float | None]] = []
    item_ids = written.column("item_id").to_pylist()
    indices = written.column("step_index").to_pylist()
    for n, (score, reason) in enumerate(
        zip(
            written.column(SCORE_COLUMN).to_pylist(),
            written.column(REASON_COLUMN).to_pylist(),
            strict=True,
        )
    ):
        if (score is None) == (reason is None):
            msg = f"scores row {n}: exactly one of score and reason must be set"
            raise OutputError(msg)
        try:
            value, why = check_result(
                score if reason is None else Unscored(reason), entry.score_kind, reasons
            )
        except PluginManifestError as exc:
            raise OutputError(f"scores row {n}: {exc}") from exc
        if why is None:
            scored += 1
        else:
            unscored[why] += 1
        rows.append((str(item_ids[n]), int(indices[n]), value))
    seen = {(item, index) for item, index, _ in rows}
    if len(seen) != len(rows):
        msg = "the steps input repeats an (item_id, step_index) pair"
        raise OutputError(msg)
    return {
        "score_kind": entry.score_kind,
        "rows": written.num_rows,
        "scored": scored,
        "unscored": unscored,
        "_rows": rows,
    }
