"""Local complete-source preflight and one exact cached classifier epoch, without dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import struct
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from strata_forge.training._population_store import (
    PopulationReader,
    PopulationRecord,
    PopulationStore,
    canonical_json,
    disk_bytes,
    hash_file,
    write_private,
)
from strata_forge.training.classifier import (
    ClassifierExample,
    _targets,  # pyright: ignore[reportPrivateUsage] — same pure SDK target contract
)
from strata_forge.training.population import (
    MAX_POPULATION_ROW_BYTES,
    POPULATION_BATCH_BYTES,
    POPULATION_BATCH_ROWS,
    POPULATION_READ_BYTES,
    PopulationEpochReceipt,
    PopulationError,
    PopulationPreparationReceipt,
    PreparedTrainingPopulation,
    PrimaryPopulationTask,
    ResolvedPopulationTokenizer,
    TrainingPopulationPlanV2,
)
from strata_forge.training.verifier_classifier import (
    _tokenize_verifier_geometry,  # pyright: ignore[reportPrivateUsage] — same fixed geometry tokenizer call
)
from strata_forge.training.verifier_config import render_verifier_input
from strata_forge.training.verifier_geometry import (
    VerifierExample,
    render_verifier_geometry,
    resolve_pid_markers,
)
from strata_forge.training.verifier_loss import ResearchLossRow

if TYPE_CHECKING:
    from collections.abc import Callable

    from strata_forge.training._classifier_heads import InitializedClassifier
    from strata_forge.training.population import PopulationAuthority, TrustedPopulationSource
    from strata_forge.training.verifier_classifier import InitializedVerifierClassifier

# Resolved optional native objects cross only this local SDK boundary.
type _ML = Any
_ISSUED: weakref.WeakKeyDictionary[PreparedTrainingPopulation, tuple[str, str, str]] = (
    weakref.WeakKeyDictionary()
)


def _raw(value: object) -> object:
    if isinstance(value, BaseModel):
        if value.model_extra or set(value.__dict__) != set(type(value).model_fields):
            raise PopulationError()
        return {name: _raw(getattr(value, name)) for name in type(value).model_fields}
    if type(value) is dict:
        return {key: _raw(item) for key, item in cast("dict[object, object]", value).items()}
    if type(value) is tuple:
        return tuple(_raw(item) for item in cast("tuple[object, ...]", value))
    if type(value) is list:
        return [_raw(item) for item in cast("list[object]", value)]
    return value


def _plan(value: TrainingPopulationPlanV2) -> TrainingPopulationPlanV2:
    return TrainingPopulationPlanV2.model_validate(_raw(value))


def _resolved_tokenizer(
    plan: TrainingPopulationPlanV2, resolved: ResolvedPopulationTokenizer
) -> _ML:
    if type(resolved) is not ResolvedPopulationTokenizer or resolved.identity != plan.tokenizer:
        raise PopulationError("population_tokenizer_invalid")
    identity = type(plan.tokenizer).model_validate(_raw(resolved.identity))
    if identity != plan.tokenizer:
        raise PopulationError("population_tokenizer_invalid")
    return resolved.tokenizer


def _fingerprint(tokenizer: _ML) -> str:
    if (
        tokenizer.is_fast is not True
        or tokenizer.padding_side != "right"
        or type(tokenizer.pad_token_id) is not int
        or tokenizer.pad_token_id < 0
    ):
        raise PopulationError("population_tokenizer_invalid")
    data = tokenizer.backend_tokenizer.to_str()
    if type(data) is not str or len(data.encode("utf-8")) > 16 * 1024**2:
        raise PopulationError("population_tokenizer_invalid")
    metadata = {
        "backend": data,
        "class": type(tokenizer).__name__,
        "padding_side": tokenizer.padding_side,
        "pad": tokenizer.pad_token_id,
        "cls": tokenizer.cls_token_id,
        "eos": tokenizer.eos_token_id,
        "model_inputs": list(tokenizer.model_input_names),
    }
    return hashlib.sha256(canonical_json(metadata)).hexdigest()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PopulationError("population_source_invalid")
        result[key] = value
    return result


def _invalid_constant(_: str) -> object:
    raise PopulationError("population_source_invalid")


def _integers(values: _ML, *, binary: bool = False) -> tuple[int, ...]:
    if (
        type(values) is not list
        or not values
        or any(
            type(value) is not int or not 0 <= value <= (1 if binary else 2**32 - 1)
            for value in cast("list[object]", values)
        )
    ):
        raise PopulationError("population_tokenizer_invalid")
    return tuple(cast("list[int]", values))


def _encode_row(
    plan: TrainingPopulationPlanV2, tokenizer: _ML, raw: bytes, ordinal: int
) -> tuple[str, bytes, PopulationRecord | None, bool]:
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_invalid_constant)
    if type(value) is not dict:
        raise PopulationError("population_source_invalid")
    value = cast("dict[str, _ML]", value)
    markers: tuple[int, ...] = ()
    pids: tuple[str, ...] = ()
    if isinstance(plan.task, PrimaryPopulationTask):
        row = ClassifierExample.model_validate_json(raw)
        config = plan.task.config
        target = _targets(config, row)
        if plan.family != "bert" and row.text_pair is not None:
            raise PopulationError("population_training_unsupported")
        if config.rendering is not None:
            rendered = render_verifier_input(config.rendering, row.premises, row.hypothesis)
            if rendered.error_code is not None or (row.text, row.text_pair) != (
                rendered.text,
                rendered.text_pair,
            ):
                raise PopulationError("population_source_invalid")
        elif row.premises is not None or row.hypothesis is not None:
            raise PopulationError("population_source_invalid")
        encoded = tokenizer(row.text, text_pair=row.text_pair, truncation=False, padding=False)
        ids = _integers(encoded["input_ids"])
        attention = _integers(encoded["attention_mask"], binary=True)
        types = (
            None
            if "token_type_ids" not in encoded or plan.family != "bert"
            else _integers(encoded["token_type_ids"])
        )
        targets = canonical_json(target).decode()
        canonical = canonical_json(row.model_dump(mode="json"))
        item_id = row.item_id
    else:
        if set(value) != {"example", "targets"} or type(value["targets"]) is not dict:
            raise PopulationError("population_source_invalid")
        target_fields = cast("dict[str, object]", value["targets"])
        if set(target_fields) != {"primary", "flag_target", "premise_ids", "support_set"}:
            raise PopulationError("population_source_invalid")
        primary = target_fields["primary"]
        if type(primary) is not dict or "label_space" not in cast("dict[str, object]", primary):
            raise PopulationError("population_source_invalid")
        example = VerifierExample.model_validate_json(canonical_json(value["example"]))
        supervision = ResearchLossRow.model_validate_json(canonical_json(target_fields))
        pids = tuple(premise.pid for premise in example.premises)
        if (
            supervision.premise_ids != pids
            or (example.support_set is not None and example.support_set != supervision.support_set)
            or (supervision.flag_eligible and "inconsistent" not in plan.task.geometry.outputs)
            or (supervision.support_eligible and "support_set" not in plan.task.geometry.outputs)
        ):
            raise PopulationError("population_source_invalid")
        rendered = render_verifier_geometry(plan.task.geometry, example)
        encoded = _tokenize_verifier_geometry(
            tokenizer, (rendered,), plan.task.geometry, padding=False, return_tensors=None
        )
        ids = _integers(encoded["input_ids"][0])
        attention = _integers(encoded["attention_mask"][0], binary=True)
        types = (
            None
            if "token_type_ids" not in encoded or plan.family != "bert"
            else _integers(encoded["token_type_ids"][0])
        )
        if plan.task.geometry.pooling == "cls" and (
            type(tokenizer.cls_token_id) is not int or ids[0] != tokenizer.cls_token_id
        ):
            raise PopulationError("population_tokenizer_invalid")
        if len(ids) <= plan.max_length:
            markers = resolve_pid_markers(
                rendered,
                offsets=encoded["offset_mapping"][0],
                sequence_ids=encoded.sequence_ids(0),
                special_tokens_mask=encoded["special_tokens_mask"][0],
                attention_mask=list(attention),
            )
        targets = canonical_json(supervision.model_dump(mode="json")).decode()
        canonical = canonical_json(
            {
                "example": example.model_dump(mode="json"),
                "targets": supervision.model_dump(mode="json"),
            }
        )
        item_id = example.item_id
    if (
        len(attention) != len(ids)
        or any(value != 1 for value in attention)
        or (types is not None and len(types) != len(ids))
    ):
        raise PopulationError("population_tokenizer_invalid")
    overflow = len(ids) > plan.max_length
    selected = plan.source_start <= ordinal < plan.source_start + plan.selected_rows
    record = (
        PopulationRecord(ordinal, item_id, ids, types, markers, pids, targets)
        if selected and not overflow
        else None
    )
    return item_id, canonical, record, overflow


async def _drained(function: Callable[[], _ML], cancelled: threading.Event) -> _ML:
    task = asyncio.create_task(asyncio.to_thread(function))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        cancelled.set()
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if task.done() and not task.cancelled():
            task.exception()
        raise


async def _authority(
    authority: PopulationAuthority, plan: TrainingPopulationPlanV2, phase: _ML
) -> None:
    denied = False
    try:
        await authority.check(plan, phase=phase)
    except Exception:
        denied = True
    if denied:
        raise PopulationError("population_authority_denied")


@dataclass
class _Scan:
    observed: int = 0
    selected: int = 0
    overflow: int = 0
    selected_overflow: int = 0
    first: str | None = None
    max_rows: int = 0
    max_bytes: int = 0


async def prepare_training_population(
    plan: TrainingPopulationPlanV2,
    *,
    source: TrustedPopulationSource,
    tokenizer: ResolvedPopulationTokenizer,
    authority: PopulationAuthority,
    attempt_dir: Path,
) -> PreparedTrainingPopulation:
    """Scan every named-source row to EOF; only complete zero-overflow caches become usable."""
    cancelled = threading.Event()
    store: PopulationStore | None = None
    directory: Path | None = None
    scan = _Scan()
    stream_hash, canonical_hash, selected_hash = (hashlib.sha256() for _ in range(3))
    total_bytes = 0
    complete = False
    fingerprint = ""
    code: str | None = None
    cancellation: asyncio.CancelledError | None = None
    try:
        plan = _plan(plan)
        resolved_tokenizer = tokenizer
        tokenizer = _resolved_tokenizer(plan, resolved_tokenizer)
        if source.identity != plan.source:
            raise PopulationError("population_source_invalid")
        fingerprint = await _drained(lambda: _fingerprint(tokenizer), cancelled)
        await _authority(authority, plan, "before_read")

        def reserve() -> PopulationStore:
            nonlocal directory, store
            if cancelled.is_set():
                raise PopulationError("population_cancelled")
            if (
                not issubclass(type(cast("object", attempt_dir)), Path)
                or not attempt_dir.is_absolute()
                or attempt_dir.exists()
                or attempt_dir.is_symlink()
                or attempt_dir.parent.resolve() != attempt_dir.parent
                or shutil.disk_usage(attempt_dir.parent).free < plan.limits.disk_bytes
            ):
                raise PopulationError("population_attempt_unavailable")
            attempt_dir.mkdir(mode=0o700)
            directory = attempt_dir
            write_private(
                attempt_dir,
                "owner.json",
                canonical_json({"version": 2, "plan_sha256": plan.digest()}),
                plan.limits.disk_bytes,
            )
            store = PopulationStore(attempt_dir, plan.limits.disk_bytes)
            return store

        store = await _drained(reserve, cancelled)
        directory = attempt_dir
        pending: list[bytes] = []
        pending_bytes = 0
        buffer = bytearray()

        def flush() -> None:
            nonlocal pending_bytes
            if store is None:
                raise PopulationError("population_cache_invalid")
            scan.max_rows = max(scan.max_rows, len(pending))
            scan.max_bytes = max(scan.max_bytes, pending_bytes)
            writes: list[tuple[int, str, PopulationRecord | None]] = []
            for raw in pending:
                if cancelled.is_set():
                    raise PopulationError("population_cancelled")
                if scan.observed >= plan.source.expected_rows:
                    raise PopulationError("population_source_invalid")
                item_id, canonical, record, overflow = _encode_row(
                    plan, tokenizer, raw, scan.observed
                )
                selected = (
                    plan.source_start <= scan.observed < plan.source_start + plan.selected_rows
                )
                canonical_hash.update(canonical + b"\n")
                if selected:
                    selected_hash.update(canonical + b"\n")
                    scan.selected += 1
                if overflow:
                    scan.overflow += 1
                    scan.selected_overflow += int(selected)
                    if scan.first is None:
                        scan.first = item_id
                writes.append((scan.observed, item_id, record))
                scan.observed += 1
            store.write(writes)
            pending.clear()
            pending_bytes = 0

        async def frame(raw: bytes) -> None:
            nonlocal pending_bytes
            if not raw or len(raw) > MAX_POPULATION_ROW_BYTES:
                raise PopulationError("population_limit")
            if (
                len(pending) >= POPULATION_BATCH_ROWS
                or pending_bytes + len(raw) > POPULATION_BATCH_BYTES
            ):
                await _drained(flush, cancelled)
            pending.append(raw)
            pending_bytes += len(raw)

        while True:
            chunk = await source.read(POPULATION_READ_BYTES)
            if type(chunk) is not bytes or len(chunk) > POPULATION_READ_BYTES:
                raise PopulationError("population_source_invalid")
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > plan.limits.serialized_bytes:
                raise PopulationError("population_limit")
            stream_hash.update(chunk)
            buffer.extend(chunk)
            while (newline := buffer.find(b"\n")) >= 0:
                raw = bytes(buffer[:newline])
                del buffer[: newline + 1]
                await frame(raw)
            if len(buffer) > MAX_POPULATION_ROW_BYTES:
                raise PopulationError("population_limit")
        if buffer:
            await frame(bytes(buffer))
        if pending:
            await _drained(flush, cancelled)
        if (
            source.identity != plan.source
            or scan.observed != plan.source.expected_rows
            or scan.selected != plan.selected_rows
        ):
            raise PopulationError("population_source_invalid")
        if (
            _resolved_tokenizer(plan, resolved_tokenizer) is not tokenizer
            or await _drained(lambda: _fingerprint(tokenizer), cancelled) != fingerprint
        ):
            raise PopulationError("population_tokenizer_invalid")
        await _authority(authority, plan, "after_read")
        complete = True
        if scan.overflow:
            code = "population_overflow"
    except asyncio.CancelledError as error:
        cancelled.set()
        cancellation = error
        code = "population_cancelled"
    except Exception as error:
        code = str(error) if type(error) is PopulationError else "population_source_invalid"
    finally:
        try:
            await _drained(lambda: store.close() if store is not None else None, cancelled)
        except asyncio.CancelledError as error:
            cancellation, code = error, "population_cancelled"
        except Exception:
            code = "population_cache_invalid"
        try:
            close_task = asyncio.create_task(source.aclose())
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                while not close_task.done():
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                if close_task.done() and not close_task.cancelled():
                    close_task.exception()
                raise
        except asyncio.CancelledError as error:
            cancellation, code = error, "population_cancelled"
        except Exception:
            code = "population_source_invalid"

    def finish() -> PreparedTrainingPopulation | None:
        if directory is None:
            return None
        handle: PreparedTrainingPopulation | None = None
        cache = directory / "cache.sqlite"
        cache_bytes = cache.stat().st_size if cache.exists() else 0
        high_water = max(disk_bytes(directory), store.high_water if store is not None else 0)
        receipt = PopulationPreparationReceipt(
            status="prepared_complete"
            if code is None
            else "overflow"
            if code == "population_overflow"
            else "cancelled"
            if cancellation
            else "invalid",
            plan_sha256=plan.digest(),
            tokenizer_sha256=fingerprint,
            scan_complete=complete,
            expected_source_rows=plan.source.expected_rows,
            source_rows_observed=scan.observed,
            selected_rows_observed=scan.selected,
            source_overflow_rows=scan.overflow,
            selected_overflow_rows=scan.selected_overflow,
            first_overflow_item_id=scan.first,
            serialized_bytes=total_bytes,
            source_stream_sha256=stream_hash.hexdigest() if complete else None,
            canonical_source_sha256=canonical_hash.hexdigest() if complete else None,
            selected_population_sha256=selected_hash.hexdigest() if complete else None,
            cache_sha256=hash_file(cache) if code is None else None,
            cache_bytes=cache_bytes,
            disk_high_water_bytes=high_water,
            max_buffered_rows=scan.max_rows,
            max_buffered_bytes=scan.max_bytes,
        )
        for _ in range(3):
            size = len(canonical_json(receipt.model_dump(mode="json")))
            receipt = PopulationPreparationReceipt.model_validate(
                {
                    **receipt.model_dump(),
                    "disk_high_water_bytes": max(high_water, disk_bytes(directory) + size),
                }
            )
        encoded_receipt = canonical_json(receipt.model_dump(mode="json"))
        write_private(directory, "preflight-receipt.json", encoded_receipt, plan.limits.disk_bytes)
        if code is None and not cancelled.is_set():
            handle = PreparedTrainingPopulation(
                directory, plan, receipt, hashlib.sha256(encoded_receipt).hexdigest()
            )

        if code is not None or cancelled.is_set():
            return None
        return handle

    finish_error: str | None = None
    handle = None
    try:
        handle = await _drained(finish, cancelled)
    except asyncio.CancelledError as error:
        cancellation, code = error, "population_cancelled"
        while True:
            try:
                await _drained(finish, cancelled)
                break
            except asyncio.CancelledError:
                continue
            except Exception:
                break
    except Exception as error:
        finish_error = str(error) if type(error) is PopulationError else "population_cache_invalid"
    if cancellation is not None:
        raise cancellation
    if finish_error is not None:
        raise PopulationError(finish_error)
    if handle is not None:
        _ISSUED[handle] = (str(handle.directory), handle.plan.digest(), handle.receipt_sha256)
    if cancellation is not None:
        raise cancellation
    if code is not None or handle is None:
        raise PopulationError(code or "population_source_invalid")
    return handle


def _backend(
    plan: TrainingPopulationPlanV2, initialized: _ML, tokenizer: _ML, cancelled: threading.Event
) -> _ML:
    from strata_forge.training._population_native import NativePopulationBackend

    return NativePopulationBackend(plan, initialized, tokenizer, cancelled)


def _verify(prepared: PreparedTrainingPopulation, tokenizer: _ML) -> TrainingPopulationPlanV2:
    if type(prepared) is not PreparedTrainingPopulation or prepared not in _ISSUED:
        raise PopulationError("population_cache_invalid")
    plan = _plan(prepared.plan)
    tokenizer = _resolved_tokenizer(plan, tokenizer)
    if _ISSUED[prepared] != (str(prepared.directory), plan.digest(), prepared.receipt_sha256):
        raise PopulationError("population_cache_invalid")
    if (
        prepared.directory.is_symlink()
        or prepared.directory.resolve() != prepared.directory
        or disk_bytes(prepared.directory) > plan.limits.disk_bytes
    ):
        raise PopulationError("population_cache_invalid")
    allowed = {
        "cache.sqlite",
        "owner.json",
        "preflight-receipt.json",
        "epoch-started",
        "epoch-receipt.json",
    }
    if any(path.name not in allowed for path in prepared.directory.iterdir()):
        raise PopulationError("population_cache_invalid")
    with (prepared.directory / "preflight-receipt.json").open("rb") as handle:
        receipt_bytes = handle.read(65_537)
    if (
        len(receipt_bytes) > 65_536
        or hashlib.sha256(receipt_bytes).hexdigest() != prepared.receipt_sha256
        or receipt_bytes != canonical_json(prepared.receipt.model_dump(mode="json"))
        or prepared.receipt.status != "prepared_complete"
        or not prepared.receipt.scan_complete
        or prepared.receipt.plan_sha256 != plan.digest()
        or prepared.receipt.source_overflow_rows != 0
        or prepared.receipt.source_rows_observed != plan.source.expected_rows
        or prepared.receipt.selected_rows_observed != plan.selected_rows
        or prepared.receipt.tokenizer_sha256 != _fingerprint(tokenizer)
        or prepared.receipt.cache_sha256 != hash_file(prepared.directory / "cache.sqlite")
        or disk_bytes(prepared.directory) > plan.limits.disk_bytes
    ):
        raise PopulationError("population_cache_invalid")
    return plan


def _epoch(
    prepared: PreparedTrainingPopulation,
    plan: TrainingPopulationPlanV2,
    initialized: _ML,
    tokenizer: _ML,
    cancelled: threading.Event,
) -> PopulationEpochReceipt:
    from strata_forge.core.repro import set_seed
    from strata_forge.training._classifier_heads import classifier_modules

    torch, _ = classifier_modules()
    if _verify(prepared, tokenizer).digest() != plan.digest():
        raise PopulationError("population_cache_invalid")
    tokenizer = _resolved_tokenizer(plan, tokenizer)
    set_seed(plan.seed)
    permutation = torch.randperm(
        plan.selected_rows,
        device="cpu",
        dtype=torch.int64,
        generator=torch.Generator(device="cpu").manual_seed(plan.seed),
    )
    planned, applied = (
        hashlib.sha256(b"population-source-ordinals-v1\0"),
        hashlib.sha256(b"population-source-ordinals-v1\0"),
    )
    for offset in range(0, plan.selected_rows, 32):
        for index in permutation[offset : offset + 32].tolist():
            planned.update(struct.pack("<Q", plan.source_start + index))
    reader = PopulationReader(
        prepared.directory, expected=plan.selected_rows, max_length=plan.max_length
    )
    backend = None
    count = steps = 0
    in_doubt = False
    status = "failed"
    try:
        if cancelled.is_set():
            status = "cancelled"
        else:
            backend = _backend(plan, initialized, tokenizer, cancelled)
            for offset in range(0, plan.selected_rows, 32):
                if cancelled.is_set():
                    status = "cancelled"
                    break
                records = tuple(
                    reader.get(index) for index in permutation[offset : offset + 32].tolist()
                )
                in_doubt = True
                backend.apply(records)
                in_doubt = False
                count += len(records)
                steps += 1
                for record in records:
                    applied.update(struct.pack("<Q", record.source_ordinal))
                backend.after_step()
            else:
                status = "cancelled" if cancelled.is_set() else "complete"
    finally:
        reader.close()
        unchanged = prepared.receipt.cache_sha256 == hash_file(prepared.directory / "cache.sqlite")
        receipt = PopulationEpochReceipt(
            status=cast("_ML", status),
            plan_sha256=plan.digest(),
            preparation_sha256=prepared.receipt_sha256,
            selected_rows=plan.selected_rows,
            confirmed_applied_rows=count,
            confirmed_optimizer_steps=steps,
            expected_optimizer_steps=plan.steps,
            planned_order_sha256=planned.hexdigest(),
            applied_order_sha256=applied.hexdigest(),
            optimizer_in_doubt=in_doubt,
            optimizer_profile={} if backend is None else backend.profile,
            native_versions={} if backend is None else backend.versions,
            cache_unchanged=unchanged,
        )
        write_private(
            prepared.directory,
            "epoch-receipt.json",
            canonical_json(receipt.model_dump(mode="json")),
            plan.limits.disk_bytes,
        )
    return receipt


async def train_prepared_population(
    prepared: PreparedTrainingPopulation,
    *,
    initialized: InitializedClassifier | InitializedVerifierClassifier,
    tokenizer: ResolvedPopulationTokenizer,
    authority: PopulationAuthority,
) -> PopulationEpochReceipt:
    """Consume one issued cache exactly once; callers own model aliases and final safe export."""
    cancelled = threading.Event()
    result = None
    code = "population_training_failed"
    try:
        plan = await _drained(lambda: _verify(prepared, tokenizer), cancelled)
        await _authority(authority, plan, "before_train")

        def claim() -> None:
            path = prepared.directory / "epoch-started"
            with path.open("xb") as handle:
                path.chmod(0o600)
                handle.write(plan.digest().encode())

        await _drained(claim, cancelled)
        result = await _drained(
            lambda: _epoch(prepared, plan, initialized, tokenizer, cancelled), cancelled
        )
        try:
            await _authority(authority, plan, "after_train")
        except (PopulationError, asyncio.CancelledError) as failure:
            status = (
                "cancelled" if isinstance(failure, asyncio.CancelledError) else "authority_denied"
            )
            denied_receipt = PopulationEpochReceipt.model_validate(
                {**result.model_dump(), "status": status}
            )
            await _drained(
                lambda: write_private(
                    prepared.directory,
                    "epoch-receipt.json",
                    canonical_json(denied_receipt.model_dump(mode="json")),
                    plan.limits.disk_bytes,
                ),
                cancelled,
            )
            raise
        if (
            result.status != "complete"
            or not result.cache_unchanged
            or result.confirmed_applied_rows != plan.selected_rows
            or result.confirmed_optimizer_steps != plan.steps
            or result.planned_order_sha256 != result.applied_order_sha256
        ):
            result = None
    except asyncio.CancelledError:
        raise
    except ImportError:
        raise ImportError("Install 'strata-forge[finetuning]' for population training.") from None
    except Exception as error:
        code = str(error) if type(error) is PopulationError else "population_training_failed"
        result = None
    if result is None:
        raise PopulationError(code)
    return result
