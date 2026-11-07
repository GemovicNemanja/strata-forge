"""Bounded local CPU SDK timings over resolved objects; no loading, persistence or dispatch."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import platform
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from strata_forge.core.errors import ValidationError
from strata_forge.pipelines._hub_ids import validate_hub_ref
from strata_forge.pipelines.benchmark_spec import (
    MAX_BENCHMARK_APPLICATIONS,
    MAX_BENCHMARK_LENGTH,
    MAX_BENCHMARK_ROW_BYTES,
    MAX_BENCHMARK_ROWS,
    MAX_BENCHMARK_TENSOR_BYTES,
    MAX_BENCHMARK_TEXT_BYTES,
    MAX_DURATION_NS,
    BenchmarkEnvironment,
    BenchmarkError,
    BenchmarkReason,
    BenchmarkTargetDescriptor,
    ClassifierBenchmarkPlanV1,
    ClassifierBenchmarkResult,
    ResolvedBenchmarkTarget,
    _statistics,  # pyright: ignore[reportPrivateUsage] — exact shared benchmark arithmetic leaf
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel

    from strata_forge.pipelines._scoring import ScoringInput
    from strata_forge.training.verifier_classifier import VerifierBatch
    from strata_forge.training.verifier_geometry import VerifierExample

# Optional native models/tokenizers/tensors have dynamic versioned APIs at this resolved seam.
type _ML = Any
_CLOCK: Callable[[], int] = time.perf_counter_ns
_OWNERS: dict[int, object] = {}
_OWNERS_LOCK = threading.Lock()


def _json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _raw(value: BaseModel) -> dict[str, object]:
    if value.model_extra or set(value.__dict__) != set(type(value).model_fields):
        raise BenchmarkError()
    return {name: getattr(value, name) for name in type(value).model_fields}


def _plan(value: ClassifierBenchmarkPlanV1) -> ClassifierBenchmarkPlanV1:
    if type(value) is not ClassifierBenchmarkPlanV1:
        raise BenchmarkError()
    return ClassifierBenchmarkPlanV1.model_validate(_raw(value))


class _Clock:
    def __init__(self) -> None:
        self.last: int | None = None

    def now(self) -> int:
        value = _CLOCK()
        if type(value) is not int or (self.last is not None and value < self.last):
            raise BenchmarkError("benchmark_timer_invalid")
        self.last = value
        return value


def _duration(start: int, end: int) -> int:
    value = end - start
    if not 0 <= value <= MAX_DURATION_NS:
        raise BenchmarkError("benchmark_timer_invalid")
    return value


async def _drain(awaitable: Awaitable[_ML], stopped: threading.Event | None = None) -> _ML:
    task = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        if stopped is not None:
            stopped.set()
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


def _tokenizer_hash(tokenizer: _ML) -> str:
    if (
        tokenizer.is_fast is not True
        or tokenizer.padding_side not in ("left", "right")
        or type(tokenizer.pad_token_id) is not int
        or tokenizer.pad_token_id < 0
    ):
        raise BenchmarkError("benchmark_target_unsupported")
    backend = tokenizer.backend_tokenizer.to_str()
    if type(backend) is not str or len(backend.encode("utf-8")) > 16 * 1024**2:
        raise BenchmarkError("benchmark_target_unsupported")
    configuration = json.loads(backend)
    if type(configuration) is not dict:
        raise BenchmarkError("benchmark_target_unsupported")
    semantic = {
        key: value
        for key, value in cast("dict[str, object]", configuration).items()
        if key not in ("padding", "truncation")
    }
    return hashlib.sha256(
        _json(
            {
                "profile": "benchmark_tokenizer_semantic_v1",
                "backend": semantic,
                "pad": tokenizer.pad_token_id,
                "cls": tokenizer.cls_token_id,
                "eos": tokenizer.eos_token_id,
                "side": tokenizer.padding_side,
            }
        )
    ).hexdigest()


def _native(model: _ML, tokenizer: _ML, family: str, maximum: int, torch: _ML) -> int:
    if (
        family not in ("bert", "gpt2", "modernbert-decoder")
        or type(maximum) is not int
        or not 1 <= maximum <= MAX_BENCHMARK_LENGTH
    ):
        raise BenchmarkError("benchmark_target_unsupported")
    parameters = tuple(model.parameters())
    buffers = tuple(model.buffers())
    modules = tuple(model.modules())
    if (
        not parameters
        or any(value.device.type != "cpu" or value.dtype != torch.float32 for value in parameters)
        or any(
            value.device.type != "cpu"
            or (value.is_floating_point() and value.dtype != torch.float32)
            for value in buffers
        )
        or torch.get_default_device().type != "cpu"
        or getattr(model, "is_quantized", False)
        or getattr(model, "is_loaded_in_4bit", False)
        or getattr(model, "is_loaded_in_8bit", False)
        or any(
            getattr(module, "_orig_mod", None) is not None
            or getattr(module, "_compiled_call_impl", None) is not None
            for module in modules
        )
    ):
        raise BenchmarkError("benchmark_target_unsupported")
    body = model.bert if family == "bert" else getattr(model, "backbone", None)
    if body is None:
        body = model.transformer if family == "gpt2" else model.model
    configured = getattr(
        model.config, "n_positions" if family == "gpt2" else "max_position_embeddings", None
    )
    actual = (
        body.embeddings.position_embeddings.num_embeddings
        if family == "bert"
        else body.wpe.num_embeddings
        if family == "gpt2"
        else body.rotary_emb.original_max_seq_len
    )
    if (
        type(configured) is not int
        or type(actual) is not int
        or actual < 1
        or actual != configured
        or maximum > actual
        or (family == "gpt2" and getattr(model.config, "max_position_embeddings", None) != actual)
    ):
        raise BenchmarkError("benchmark_target_unsupported")
    embedding = body.get_input_embeddings()
    vocab = tokenizer.get_vocab()
    if (
        not isinstance(embedding, torch.nn.Embedding)
        or type(embedding.num_embeddings) is not int
        or embedding.num_embeddings < 1
        or embedding.weight.shape[0] != embedding.num_embeddings
        or type(vocab) is not dict
        or not vocab
        or any(
            type(key) is not str
            or type(value) is not int
            or not 0 <= value < embedding.num_embeddings
            for key, value in cast("dict[object,object]", vocab).items()
        )
        or model.config.pad_token_id != tokenizer.pad_token_id
    ):
        raise BenchmarkError("benchmark_target_unsupported")
    return actual


def _embedding_rows(model: _ML, family: str) -> int:
    body = model.bert if family == "bert" else getattr(model, "backbone", None)
    if body is None:
        body = model.transformer if family == "gpt2" else model.model
    return cast("int", body.get_input_embeddings().num_embeddings)


def _environment(model: _ML, torch: _ML, transformers: _ML) -> BenchmarkEnvironment:
    info = time.get_clock_info("perf_counter")
    attention = getattr(model.config, "_attn_implementation", None)
    if info.monotonic is not True or attention not in (None, "eager", "sdpa"):
        raise BenchmarkError("benchmark_target_unsupported")
    return BenchmarkEnvironment(
        python=sys.version.split()[0],
        torch=str(torch.__version__),
        transformers=str(transformers.__version__),
        operating_system=platform.system(),
        architecture=platform.machine(),
        intraop_threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        attention=attention,
        clock_implementation=info.implementation,
        clock_resolution_ns=float(info.resolution * 1e9),
    )


def _labels(labels: tuple[str, ...]) -> str:
    if (
        not 2 <= len(labels) <= 16
        or len(set(labels)) != len(labels)
        or any(type(label) is not str or not 1 <= len(label) <= 64 for label in labels)
    ):
        raise BenchmarkError("benchmark_target_unsupported")
    return hashlib.sha256(_json(labels)).hexdigest()


def _text_bytes(value: str) -> int:
    if type(value) is not str:
        raise BenchmarkError("benchmark_input_invalid")
    size = len(value.encode("utf-8"))
    if size > MAX_BENCHMARK_ROW_BYTES:
        raise BenchmarkError("benchmark_input_invalid")
    return size


def _probabilities(value: object, labels: tuple[str, ...]) -> None:
    if type(value) is not dict:
        raise BenchmarkError("benchmark_failed")
    values = cast("dict[object,object]", value)
    if (
        set(values) != set(labels)
        or any(
            type(p) not in (int, float)
            or not math.isfinite(cast("float", p))
            or not 0 <= cast("float", p) <= 1
            for p in values.values()
        )
        or not math.isclose(
            math.fsum(cast("float", p) for p in values.values()), 1.0, abs_tol=1e-6, rel_tol=0
        )
    ):
        raise BenchmarkError("benchmark_failed")


class _Operation(Protocol):
    descriptor: BenchmarkTargetDescriptor
    environment: BenchmarkEnvironment
    stopped: threading.Event

    def fingerprint(self) -> str: ...
    async def call(self) -> object: ...
    def validate(self, value: object) -> None: ...


@dataclass(repr=False)
class _ScorerOperation:
    scorer: _ML
    rows: tuple[_ML, ...]
    descriptor: BenchmarkTargetDescriptor
    environment: BenchmarkEnvironment
    batch_size: int
    labels: tuple[str, ...]
    torch: _ML
    model: _ML = field(init=False)
    stopped: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.model = self.scorer._model

    def fingerprint(self) -> str:
        from strata_forge.pipelines._scoring import ScoringInput

        raw: list[dict[str, object]] = []
        total = 0
        for row in self.rows:
            if type(row) is not ScoringInput:
                raise BenchmarkError("benchmark_input_invalid")
            parsed = ScoringInput.model_validate(_raw(row), strict=True)
            row_bytes = _text_bytes(parsed.text) + (
                0 if parsed.text_pair is None else _text_bytes(parsed.text_pair)
            )
            if row_bytes > MAX_BENCHMARK_ROW_BYTES:
                raise BenchmarkError("benchmark_input_invalid")
            total += row_bytes
            if total > MAX_BENCHMARK_TEXT_BYTES:
                raise BenchmarkError("benchmark_input_invalid")
            raw.append(parsed.model_dump(mode="json"))
        if (
            self.scorer._model is not self.model
            or self.scorer._torch is not self.torch
            or self.scorer._engine != self.descriptor.engine
            or self.scorer._device != "cpu"
            or self.scorer._max_length != self.descriptor.max_length
            or any(module.training for module in self.model.modules())
        ):
            raise BenchmarkError("benchmark_input_invalid")
        _native(
            self.model,
            self.scorer._tokenizer,
            self.descriptor.family,
            self.descriptor.max_length,
            self.torch,
        )
        if (
            self.scorer.labels != self.labels
            or _tokenizer_hash(self.scorer._tokenizer) != self.descriptor.tokenizer_sha256
        ):
            raise BenchmarkError("benchmark_input_invalid")
        return hashlib.sha256(b"strata-benchmark-hf-input-v1\0" + _json(raw)).hexdigest()

    async def call(self) -> object:
        if self.stopped.is_set():
            raise asyncio.CancelledError()
        return await self.scorer.score(
            self.rows, batch_size=self.batch_size, on_progress=None, overflow_policy="error"
        )

    def validate(self, value: object) -> None:
        from strata_forge.pipelines._scoring import ScoredRow, ScoringResult

        if (
            type(value) is not ScoringResult
            or len(value.rows) != len(self.rows)
            or any(
                type(count) is not int or count != 0
                for count in (value.failed_rows, value.overflow_rows, value.truncated_rows)
            )
        ):
            raise BenchmarkError("benchmark_failed")
        for index, row in enumerate(value.rows):
            if (
                type(row) is not ScoredRow
                or type(row.index) is not int
                or row.index != index
                or row.error is not None
                or row.overflow is not False
                or row.truncated is not False
            ):
                raise BenchmarkError("benchmark_failed")
            _probabilities(row.probabilities, self.labels)
            if (
                row.pred
                != self.labels[
                    max(
                        range(len(self.labels)),
                        key=lambda position: row.probabilities[self.labels[position]],
                    )
                ]
            ):
                raise BenchmarkError("benchmark_failed")
            if self.descriptor.engine == "hf_label_logits":
                if (
                    type(row.mass_outside) not in (int, float)
                    or not math.isfinite(cast("float", row.mass_outside))
                    or not 0 <= cast("float", row.mass_outside) <= 1
                ):
                    raise BenchmarkError("benchmark_failed")
            elif row.mass_outside is not None:
                raise BenchmarkError("benchmark_failed")


@dataclass(repr=False)
class _VerifierOperation:
    initialized: _ML
    tokenizer: _ML
    examples: tuple[_ML, ...] | None
    batch: _ML
    descriptor: BenchmarkTargetDescriptor
    environment: BenchmarkEnvironment
    labels: tuple[str, ...]
    item_ids: tuple[str, ...]
    premise_ids: tuple[tuple[str, ...], ...]
    torch: _ML
    embedding_rows: int
    stopped: threading.Event = field(default_factory=threading.Event)

    def fingerprint(self) -> str:
        from strata_forge.training.verifier_geometry import VerifierExample

        digest = hashlib.sha256(b"strata-benchmark-verifier-input-v1\0")
        if self.examples is not None:
            values: list[dict[str, object]] = []
            total = 0
            for example in self.examples:
                if type(example) is not VerifierExample:
                    raise BenchmarkError("benchmark_input_invalid")
                row = VerifierExample.model_validate(example)
                count = sum(_text_bytes(premise.text) for premise in row.premises) + _text_bytes(
                    row.hypothesis
                )
                if count > MAX_BENCHMARK_ROW_BYTES:
                    raise BenchmarkError("benchmark_input_invalid")
                total += count
                if total > MAX_BENCHMARK_TEXT_BYTES:
                    raise BenchmarkError("benchmark_input_invalid")
                values.append(row.model_dump(mode="json"))
            digest.update(_json(values))
        else:
            _validate_batch(
                self.batch,
                self.initialized.geometry,
                self.tokenizer,
                self.torch,
                self.embedding_rows,
            )
            digest.update(
                _json(
                    {
                        "geometry": self.batch.geometry_sha256,
                        "item_ids": self.batch.item_ids,
                        "premise_ids": self.batch.premise_ids,
                        "markers": self.batch.marker_positions,
                    }
                )
            )
            for name, value in sorted(self.batch.model_inputs.items()):
                digest.update(
                    _json({"name": name, "shape": list(value.shape), "dtype": "int64-le"})
                )
                digest.update(
                    value.detach().contiguous().numpy().astype("<i8", copy=False).tobytes()
                )
        from strata_forge.training.verifier_classifier import (
            _validate_binding,  # pyright: ignore[reportPrivateUsage] — repeat accepted native metadata validation
            _validate_state,  # pyright: ignore[reportPrivateUsage] — repeat accepted native state validation
        )

        _validate_binding(
            self.initialized.model.config, self.initialized.geometry, self.initialized.receipt
        )
        _validate_state(self.initialized.model, self.initialized.receipt)
        _native(
            self.initialized.model,
            self.tokenizer,
            self.descriptor.family,
            self.descriptor.max_length,
            self.torch,
        )
        if (
            self.initialized.geometry.digest() != self.descriptor.geometry_sha256
            or _tokenizer_hash(self.tokenizer) != self.descriptor.tokenizer_sha256
        ):
            raise BenchmarkError("benchmark_input_invalid")
        return digest.hexdigest()

    async def call(self) -> object:
        from strata_forge.training.verifier_classifier import (
            encode_verifier_batch,
            score_verifier_batch,
        )

        if self.stopped.is_set():
            raise asyncio.CancelledError()
        batch = self.batch
        if self.examples is not None:
            batch = await encode_verifier_batch(
                self.tokenizer, self.examples, geometry=self.initialized.geometry
            )
        if self.stopped.is_set():
            raise asyncio.CancelledError()
        return await score_verifier_batch(self.initialized, batch)

    def validate(self, value: object) -> None:
        if type(value) is not tuple or len(cast("tuple[object, ...]", value)) != len(self.item_ids):
            raise BenchmarkError("benchmark_failed")
        for index, item in enumerate(cast("tuple[object,...]", value)):
            if type(item) is not dict:
                raise BenchmarkError("benchmark_failed")
            row = cast("dict[str,object]", item)
            if (
                set(row) != {"item_id", "probabilities", "p_inconsistent", "support_scores"}
                or row["item_id"] != self.item_ids[index]
            ):
                raise BenchmarkError("benchmark_failed")
            _probabilities(row["probabilities"], self.labels)
            flag = row["p_inconsistent"]
            if "inconsistent" in self.descriptor.outputs:
                if (
                    type(flag) not in (int, float)
                    or not math.isfinite(cast("float", flag))
                    or not 0 <= cast("float", flag) <= 1
                ):
                    raise BenchmarkError("benchmark_failed")
            elif flag is not None:
                raise BenchmarkError("benchmark_failed")
            support = row["support_scores"]
            if "support_set" in self.descriptor.outputs:
                if (
                    type(support) is not dict
                    or set(cast("dict[str,object]", support)) != set(self.premise_ids[index])
                    or any(
                        type(p) not in (int, float) or not math.isfinite(cast("float", p))
                        for p in cast("dict[str,object]", support).values()
                    )
                ):
                    raise BenchmarkError("benchmark_failed")
            elif support is not None:
                raise BenchmarkError("benchmark_failed")


def _validate_batch(
    batch: _ML, geometry: _ML, tokenizer: _ML, torch: _ML, embedding_rows: int
) -> tuple[int, ...]:
    from strata_forge.training.verifier_classifier import VerifierBatch

    if (
        type(batch) is not VerifierBatch
        or batch.geometry_sha256 != geometry.digest()
        or type(batch.model_inputs) is not dict
        or type(batch.item_ids) is not tuple
        or type(batch.marker_positions) is not tuple
        or type(batch.premise_ids) is not tuple
        or any(type(item) is not str or not 1 <= len(item) <= 1024 for item in batch.item_ids)
        or not {"input_ids", "attention_mask"}
        <= set(batch.model_inputs)
        <= (
            {"input_ids", "attention_mask", "token_type_ids"}
            if geometry.pooling == "cls"
            else {"input_ids", "attention_mask"}
        )
        or not 1 <= len(batch.item_ids) <= MAX_BENCHMARK_ROWS
        or len(set(batch.item_ids)) != len(batch.item_ids)
        or len(batch.marker_positions) != len(batch.item_ids)
        or len(batch.premise_ids) != len(batch.item_ids)
    ):
        raise BenchmarkError("benchmark_input_invalid")
    ids = batch.model_inputs["input_ids"]
    mask = batch.model_inputs["attention_mask"]
    if type(ids) is not torch.Tensor or type(mask) is not torch.Tensor:
        raise BenchmarkError("benchmark_input_invalid")
    total = 0
    for value in batch.model_inputs.values():
        if (
            type(value) is not torch.Tensor
            or value.device.type != "cpu"
            or value.dtype != torch.int64
            or value.layout != torch.strided
            or value.ndim != 2
            or value.shape != ids.shape
            or value.requires_grad
        ):
            raise BenchmarkError("benchmark_input_invalid")
        total += value.numel() * value.element_size()
    if (
        total > MAX_BENCHMARK_TENSOR_BYTES
        or ids.shape[0] != len(batch.item_ids)
        or not 1 <= ids.shape[1] <= geometry.max_length
        or not bool(((mask == 0) | (mask == 1)).all())
        or not bool((mask.sum(-1) > 0).all())
        or not bool(((ids >= 0) & (ids < embedding_rows)).all())
    ):
        raise BenchmarkError("benchmark_input_invalid")
    lengths = tuple(int(value) for value in mask.sum(-1).tolist())
    for index, (positions, pids, length) in enumerate(
        zip(batch.marker_positions, batch.premise_ids, lengths, strict=True)
    ):
        expected = (
            [1] * length + [0] * (ids.shape[1] - length)
            if tokenizer.padding_side == "right"
            else [0] * (ids.shape[1] - length) + [1] * length
        )
        if type(positions) is not tuple or type(pids) is not tuple:
            raise BenchmarkError("benchmark_input_invalid")
        if (
            mask[index].tolist() != expected
            or not bool((ids[index][mask[index] == 0] == tokenizer.pad_token_id).all())
            or len(positions) != len(pids)
            or len(pids) > 20
            or len(set(pids)) != len(pids)
            or any(type(pid) is not str for pid in pids)
            or any(
                type(position) is not int
                or not 0 <= position < ids.shape[1]
                or mask[index, position].item() != 1
                for position in positions
            )
        ):
            raise BenchmarkError("benchmark_input_invalid")
    if geometry.pooling == "cls" and (
        type(tokenizer.cls_token_id) is not int
        or not bool((ids[:, 0] == tokenizer.cls_token_id).all())
        or not bool((mask[:, 0] == 1).all())
    ):
        raise BenchmarkError("benchmark_input_invalid")
    return lengths


def _prepare(
    plan: ClassifierBenchmarkPlanV1,
    target: ResolvedBenchmarkTarget,
    rows: object,
    examples: object,
    batch: object,
) -> _Operation:
    from strata_forge.pipelines._scoring import Scorer, ScoringInput
    from strata_forge.training._classifier_heads import classifier_modules, decoder_classifier_class
    from strata_forge.training._verifier_heads import verifier_classifier_class
    from strata_forge.training.verifier_classifier import (
        InitializedVerifierClassifier,
        _validate_binding,  # pyright: ignore[reportPrivateUsage] — fixed accepted verifier metadata validation
        _validate_state,  # pyright: ignore[reportPrivateUsage] — fixed accepted verifier state validation
    )
    from strata_forge.training.verifier_geometry import VerifierExample, VerifierGeometryConfig

    torch, transformers = classifier_modules()
    labels: tuple[str, ...]
    item_ids: tuple[str, ...]
    pids: tuple[tuple[str, ...], ...]
    if plan.boundary == "hf_scoring_sdk":
        if (
            type(target.scorer) is not Scorer
            or target.verifier is not None
            or target.tokenizer is not None
            or examples is not None
            or batch is not None
            or type(rows) is not tuple
            or not 1 <= len(cast("tuple[object, ...]", rows)) <= MAX_BENCHMARK_ROWS
            or any(type(row) is not ScoringInput for row in cast("tuple[object,...]", rows))
        ):
            raise BenchmarkError("benchmark_input_invalid")
        scorer = cast("_ML", target.scorer)
        model = scorer._model
        tokenizer = scorer._tokenizer
        family = model.config.model_type
        allowed = [decoder_classifier_class()] if scorer._engine == "hf_classify" else []
        for name in (
            (
                "BertForSequenceClassification",
                "GPT2ForSequenceClassification",
                "ModernBertDecoderForSequenceClassification",
            )
            if scorer._engine == "hf_classify"
            else ("GPT2LMHeadModel", "ModernBertDecoderForCausalLM")
        ):
            cls = getattr(transformers, name, None)
            if cls is not None:
                allowed.append(cls)
        if (
            type(model) not in allowed
            or scorer._device != "cpu"
            or scorer._torch is not torch
            or scorer._engine not in ("hf_classify", "hf_label_logits")
            or tokenizer.padding_side != "right"
            or scorer.loading_audit.status != "accepted"
            or any(module.training for module in model.modules())
        ):
            raise BenchmarkError("benchmark_target_unsupported")
        labels = scorer.labels
        label_sha = _labels(labels)
        context = _native(model, tokenizer, family, scorer._max_length, torch)
        shape = "single" if cast("tuple[_ML,...]", rows)[0].text_pair is None else "pair"
        if any(
            (row.text_pair is None) != (shape == "single") for row in cast("tuple[_ML,...]", rows)
        ):
            raise BenchmarkError("benchmark_input_invalid")
        descriptor = BenchmarkTargetDescriptor(
            model_ref=target.model_ref,
            tokenizer_ref=target.tokenizer_ref,
            boundary=plan.boundary,
            engine=scorer._engine,
            family=family,
            probability_family="full_class_softmax"
            if scorer._engine == "hf_classify"
            else "restricted_label_token_softmax",
            label_count=len(labels),
            ordered_labels_sha256=label_sha,
            tokenizer_sha256=_tokenizer_hash(tokenizer),
            geometry_sha256=None,
            outputs=("label",),
            rows_per_call=len(cast("tuple[object, ...]", rows)),
            max_length=scorer._max_length,
            native_context=context,
            input_shape=shape,
            mode_policy="scorer_preexisting_eval",
            exact_token_lengths=None,
            token_lengths_reason="sdk_has_no_length_receipt",  # noqa: S106 — fixed metadata reason, not a credential
            padding_side="right",
        )
        operation: _Operation = _ScorerOperation(
            scorer,
            cast("tuple[_ML,...]", rows),
            descriptor,
            _environment(model, torch, transformers),
            plan.batch_size,
            labels,
            torch,
        )
    else:
        if (
            target.scorer is not None
            or type(target.verifier) is not InitializedVerifierClassifier
            or target.tokenizer is None
            or rows is not None
            or (examples is None) == (batch is None)
            or (plan.boundary == "verifier_preencoded_sdk" and batch is None)
            or (plan.boundary == "verifier_encode_score_sdk" and examples is None)
        ):
            raise BenchmarkError("benchmark_input_invalid")
        initialized = cast("_ML", target.verifier)
        tokenizer = cast("_ML", target.tokenizer)
        if type(initialized.model) is not verifier_classifier_class():
            raise BenchmarkError("benchmark_target_unsupported")
        geometry = VerifierGeometryConfig.model_validate(initialized.geometry)
        _validate_binding(initialized.model.config, geometry, initialized.receipt)
        _validate_state(initialized.model, initialized.receipt)
        labels = geometry.labels
        context = _native(
            initialized.model,
            tokenizer,
            initialized.receipt.primary.family,
            geometry.max_length,
            torch,
        )
        lengths = None
        if examples is not None:
            if (
                type(examples) is not tuple
                or not 1 <= len(cast("tuple[object, ...]", examples)) <= MAX_BENCHMARK_ROWS
                or any(
                    type(row) is not VerifierExample for row in cast("tuple[object,...]", examples)
                )
            ):
                raise BenchmarkError("benchmark_input_invalid")
            typed_examples = cast("tuple[VerifierExample, ...]", examples)
            item_ids = tuple(row.item_id for row in typed_examples)
            pids = tuple(tuple(p.pid for p in row.premises) for row in typed_examples)
        else:
            lengths = _validate_batch(
                batch,
                geometry,
                tokenizer,
                torch,
                _embedding_rows(initialized.model, initialized.receipt.primary.family),
            )
            native_batch = cast("_ML", batch)
            item_ids = cast("tuple[str, ...]", native_batch.item_ids)
            pids = cast("tuple[tuple[str, ...], ...]", native_batch.premise_ids)
        if len(set(item_ids)) != len(item_ids) or plan.batch_size != len(item_ids):
            raise BenchmarkError("benchmark_input_invalid")
        descriptor = BenchmarkTargetDescriptor(
            model_ref=target.model_ref,
            tokenizer_ref=target.tokenizer_ref,
            boundary=plan.boundary,
            engine="forge_auxiliary_verifier_v1",
            family=initialized.receipt.primary.family,
            probability_family="full_class_softmax",
            label_count=len(labels),
            ordered_labels_sha256=_labels(labels),
            tokenizer_sha256=_tokenizer_hash(tokenizer),
            geometry_sha256=geometry.digest(),
            outputs=geometry.outputs,
            rows_per_call=len(item_ids),
            max_length=geometry.max_length,
            native_context=context,
            input_shape="preencoded"
            if batch is not None
            else "pair"
            if geometry.pooling == "cls"
            else "single",
            mode_policy="verifier_save_eval_restore",
            exact_token_lengths=lengths,
            token_lengths_reason=None if lengths is not None else "sdk_has_no_length_receipt",
            padding_side=tokenizer.padding_side,
        )
        operation = _VerifierOperation(
            initialized,
            tokenizer,
            cast("tuple[_ML,...] | None", examples),
            batch,
            descriptor,
            _environment(initialized.model, torch, transformers),
            labels,
            item_ids,
            pids,
            torch,
            _embedding_rows(initialized.model, initialized.receipt.primary.family),
        )
    if (
        descriptor.rows_per_call * (plan.warmup_calls + plan.measured_calls)
        > MAX_BENCHMARK_APPLICATIONS
    ):
        raise BenchmarkError("benchmark_input_invalid")
    return operation


def _prepare_checked(
    plan: ClassifierBenchmarkPlanV1,
    target: ResolvedBenchmarkTarget,
    rows: object,
    examples: object,
    batch: object,
) -> _Operation:
    operation = None
    code = "benchmark_target_unsupported"
    try:
        operation = _prepare(plan, target, rows, examples, batch)
    except ImportError:
        raise ImportError(
            "Install 'strata-forge[serving]' or 'strata-forge[finetuning]' for classifier benchmarks."
        ) from None
    except Exception as error:
        if type(error) is BenchmarkError:
            code = str(error)
    if operation is None:
        raise BenchmarkError(code)
    return operation


def _claim(target: ResolvedBenchmarkTarget, batch: object) -> tuple[tuple[int, ...], object]:
    from strata_forge.pipelines._scoring import Scorer
    from strata_forge.training.verifier_classifier import (
        InitializedVerifierClassifier,
        VerifierBatch,
    )

    if type(target) is not ResolvedBenchmarkTarget:
        raise BenchmarkError()
    if (
        type(target.model_ref) is not str
        or type(target.tokenizer_ref) is not str
        or len(target.model_ref) > 256
        or len(target.tokenizer_ref) > 256
    ):
        raise BenchmarkError()
    validate_hub_ref(target.model_ref)
    validate_hub_ref(target.tokenizer_ref)
    objects: list[object] = [target]
    if type(target.scorer) is Scorer:
        scorer = cast("_ML", target.scorer)
        objects.extend([scorer, scorer._model, scorer._tokenizer])
    if type(target.verifier) is InitializedVerifierClassifier:
        objects.extend([target.verifier, target.verifier.model, target.tokenizer])
    if type(batch) is VerifierBatch:
        if type(batch.model_inputs) is not dict or not 2 <= len(batch.model_inputs) <= 3:
            raise BenchmarkError("benchmark_input_invalid")
        objects.extend(batch.model_inputs.values())
    identities = tuple({id(value) for value in objects if value is not None})
    owner = object()
    with _OWNERS_LOCK:
        if any(key in _OWNERS for key in identities):
            raise BenchmarkError("benchmark_target_unsupported")
        for key in identities:
            _OWNERS[key] = owner
    return identities, owner


def _release(claim: tuple[tuple[int, ...], object]) -> None:
    with _OWNERS_LOCK:
        for key in claim[0]:
            if _OWNERS.get(key) is claim[1]:
                del _OWNERS[key]


def _reason(error: Exception) -> BenchmarkReason:
    if type(error) is BenchmarkError and len(error.args) == 1 and type(error.args[0]) is str:
        return cast("BenchmarkReason", error.args[0])
    if isinstance(error, ValidationError) and error.args in (
        ("input_too_long: overflow_policy=error refuses scoring",),
        ("verifier_overflow",),
    ):
        return "benchmark_overflow"
    return "benchmark_failed"


async def _invoke(operation: _Operation) -> tuple[object, BenchmarkReason | None]:
    value: object = None
    reason: BenchmarkReason | None = None
    try:
        value = await _drain(operation.call(), operation.stopped)
    except asyncio.CancelledError:
        raise
    except ImportError:
        raise ImportError(
            "Install 'strata-forge[serving]' or 'strata-forge[finetuning]' for classifier benchmarks."
        ) from None
    except Exception as error:
        reason = _reason(error)
    return value, reason


async def _measure(
    plan: ClassifierBenchmarkPlanV1, operation: _Operation, clock: _Clock, start: int
) -> ClassifierBenchmarkResult:
    workload = await _drain(asyncio.to_thread(operation.fingerprint), operation.stopped)
    warmup: list[int] = []
    samples: list[int] = []
    attempted = 0
    failed_phase: Literal["warmup", "measured"] | None = None
    failed_index: int | None = None
    failed_duration: int | None = None
    window_start: int | None = None
    window_end: int | None = None
    status: Literal["complete", "failed", "deadline"] = "complete"
    reason: BenchmarkReason | None = None
    try:
        for phase, count in (("warmup", plan.warmup_calls), ("measured", plan.measured_calls)):
            for index in range(count):
                before = clock.now()
                if _duration(start, before) >= plan.max_elapsed_ms * 1_000_000:
                    status, reason = "deadline", "benchmark_deadline"
                    break
                if phase == "measured":
                    attempted += 1
                    if window_start is None:
                        window_start = before
                failed_phase, failed_index = cast("_ML", phase), index
                value, call_error = await _invoke(operation)
                after = clock.now()
                failed_duration = _duration(before, after)
                if phase == "measured":
                    window_end = after
                if call_error is not None:
                    status, reason = "failed", call_error
                    break
                operation.validate(value)
                if phase == "warmup":
                    warmup.append(failed_duration)
                else:
                    samples.append(failed_duration)
                failed_phase = failed_index = failed_duration = None
                checked = clock.now()
                if phase == "measured":
                    window_end = checked
                if _duration(start, checked) > plan.max_elapsed_ms * 1_000_000:
                    status, reason = "deadline", "benchmark_deadline"
                    break
            if status != "complete":
                break
        if (
            status == "complete"
            and await _drain(asyncio.to_thread(operation.fingerprint), operation.stopped)
            != workload
        ):
            raise BenchmarkError("benchmark_input_invalid")
        end = clock.now()
        elapsed = _duration(start, end)
        if status == "complete" and elapsed > plan.max_elapsed_ms * 1_000_000:
            status, reason = "deadline", "benchmark_deadline"
    except asyncio.CancelledError:
        raise
    except Exception as error:
        status, reason = "failed", _reason(error)
        try:
            end = clock.now()
        except BenchmarkError:
            reason = "benchmark_timer_invalid"
            end = clock.last if clock.last is not None else start
        elapsed = max(0, end - start)
    window = (
        None if window_start is None or window_end is None else _duration(window_start, window_end)
    )
    statistics = (
        _statistics(
            tuple(samples), cast("int", window), len(samples) * operation.descriptor.rows_per_call
        )
        if status == "complete"
        else None
    )
    return ClassifierBenchmarkResult(
        status=status,
        reason=reason,
        plan=plan,
        plan_sha256=plan.digest(),
        target=operation.descriptor,
        target_sha256=operation.descriptor.digest(),
        workload_sha256=workload,
        environment=operation.environment,
        warmup_completed=len(warmup),
        measured_attempted=attempted,
        measured_completed=len(samples),
        measured_rows=len(samples) * operation.descriptor.rows_per_call,
        warmup_durations_ns=tuple(warmup),
        measured_durations_ns=tuple(samples),
        failed_phase=failed_phase,
        failed_call_index=failed_index,
        failed_call_duration_ns=failed_duration,
        measurement_window_ns=window,
        total_elapsed_ns=elapsed,
        statistics=statistics,
    )


async def _benchmark(
    plan: ClassifierBenchmarkPlanV1,
    target: ResolvedBenchmarkTarget,
    rows: object,
    examples: object,
    batch: object,
) -> ClassifierBenchmarkResult:
    claim = None
    result = None
    code = "benchmark_invalid"
    missing = False
    try:
        plan = _plan(plan)
        clock = _Clock()
        start = clock.now()
        claim = _claim(target, batch)
        operation = await _drain(
            asyncio.to_thread(_prepare_checked, plan, target, rows, examples, batch)
        )
        result = await _measure(plan, operation, clock, start)
    except asyncio.CancelledError:
        raise
    except ImportError:
        missing = True
    except Exception as error:
        code = str(error) if type(error) is BenchmarkError else "benchmark_invalid"
    finally:
        if claim is not None:
            _release(claim)
    if missing:
        raise ImportError(
            "Install 'strata-forge[serving]' or 'strata-forge[finetuning]' for classifier benchmarks."
        )
    if result is None:
        raise BenchmarkError(code)
    return result


async def benchmark_scorer(
    plan: ClassifierBenchmarkPlanV1,
    *,
    target: ResolvedBenchmarkTarget,
    rows: tuple[ScoringInput, ...],
) -> ClassifierBenchmarkResult:
    """Measure whole accepted scorer calls; no loading, fallback, truncation or hidden warmups."""
    if type(plan) is not ClassifierBenchmarkPlanV1 or plan.boundary != "hf_scoring_sdk":
        raise BenchmarkError()
    return await _benchmark(plan, target, rows, None, None)


async def benchmark_verifier(
    plan: ClassifierBenchmarkPlanV1,
    *,
    target: ResolvedBenchmarkTarget,
    examples: tuple[VerifierExample, ...] | None = None,
    batch: VerifierBatch | None = None,
) -> ClassifierBenchmarkResult:
    """Measure either one fixed encoded verifier batch or local encoding plus scoring."""
    if type(plan) is not ClassifierBenchmarkPlanV1 or plan.boundary not in (
        "verifier_preencoded_sdk",
        "verifier_encode_score_sdk",
    ):
        raise BenchmarkError()
    return await _benchmark(plan, target, None, examples, batch)
