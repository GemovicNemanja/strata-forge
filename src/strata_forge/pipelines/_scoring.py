"""Private exact-scoring primitives used by inert pipeline dispatch.

Inputs are already-rendered text, never executable templates. Heavy dependencies are lazy,
and model loading accepts only explicit Hub credentials, safetensors and built-in model code.
The caller owns result persistence, lifecycle events and scrubbing upstream exceptions.
"""

from __future__ import annotations

import asyncio
import math
import re
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from strata_forge.core.errors import ValidationError
from strata_forge.pipelines._classifier_checkpoint import probe_forge_classifier
from strata_forge.pipelines._common import REPO_ID_RE
from strata_forge.pipelines._population import MAX_POPULATION, PopulationData
from strata_forge.training._load_guard import (
    LoadingAudit,
    ModelLoadingError,
    PaddingPolicy,
    _guard_loading_result,
)
from strata_forge.training.progress import ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager

type ScoringEngine = Literal["hf_classify", "hf_label_logits"]
type OverflowPolicy = Literal["legacy", "error"]
type HubCredential = str | Literal[False]


class ScoringInput(BaseModel):
    """One rendered prompt, optionally paired with a classifier hypothesis."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    text: str
    text_pair: str | None = None


class ScoredRow(BaseModel):
    """A score or an explicit length refusal, indexed in the caller's input order."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    index: int
    probabilities: dict[str, float] = Field(default_factory=dict)
    pred: str | None = None
    mass_outside: float | None = None
    overflow: bool = False
    truncated: bool = False
    error: str | None = None

    def columns(self) -> dict[str, str | float | None]:
        """Return the probability-bearing columns a result writer can persist."""
        result: dict[str, str | float | None] = {
            f"p_{label}": probability for label, probability in self.probabilities.items()
        }
        result.update(pred=self.pred, error=self.error)
        if self.mass_outside is not None:
            result["mass_outside"] = self.mass_outside
        return result


class ScoringResult(BaseModel):
    """Ordered rows and honest counts, including refused overlength inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    rows: list[ScoredRow]
    overflow_rows: int
    truncated_rows: int
    failed_rows: int


class ScoringItemOutcomeV2(PopulationData):
    """Native token evidence with an unchanged legacy score payload when one was produced."""

    index: int = Field(ge=0)
    outcome: Literal["scored", "invalid", "overflow"]
    token_count: int = Field(ge=0)
    model_limit: int = Field(ge=1)
    model_sent: bool
    result: ScoredRow | None

    @model_validator(mode="after")
    def _evidence(self) -> ScoringItemOutcomeV2:
        if self.outcome == "overflow":
            if self.token_count <= self.model_limit or self.model_sent or self.result is not None:
                raise ValueError("scoring_outcome_invalid")
        elif (
            self.token_count > self.model_limit
            or self.result is None
            or self.result.index != self.index
            or self.result.overflow
            or self.result.truncated
            or (self.outcome == "scored") != (self.result.error is None)
            or self.model_sent != (self.outcome == "scored")
        ):
            raise ValueError("scoring_outcome_invalid")
        return self


class ScoringOutcomesV2(PopulationData):
    rows: tuple[ScoringItemOutcomeV2, ...]
    overflow_rows: int = Field(ge=0)
    failed_rows: int = Field(ge=0)
    scored_rows: int = Field(ge=0)

    @model_validator(mode="after")
    def _partition(self) -> ScoringOutcomesV2:
        if (
            [row.index for row in self.rows] != list(range(len(self.rows)))
            or self.overflow_rows != sum(row.outcome == "overflow" for row in self.rows)
            or self.failed_rows != sum(row.outcome == "invalid" for row in self.rows)
            or self.scored_rows != sum(row.outcome == "scored" for row in self.rows)
            or len(self.rows) != self.overflow_rows + self.failed_rows + self.scored_rows
        ):
            raise ValueError("scoring_outcome_invalid")
        return self


class _Tensor(Protocol):
    @property
    def shape(self) -> Sequence[int]: ...
    def __getitem__(self, key: object) -> _Tensor: ...
    def to(self, device: str) -> _Tensor: ...
    def float(self) -> _Tensor: ...
    def detach(self) -> _Tensor: ...
    def cpu(self) -> _Tensor: ...
    def tolist(self) -> object: ...


class _Tokenizer(Protocol):
    padding_side: str
    pad_token_id: int | None
    eos_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...
    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...
    def __call__(
        self, text: list[str], text_pair: list[str] | None = None, **kwargs: object
    ) -> Mapping[str, _Tensor]: ...


class _ModelConfig(Protocol):
    id2label: Mapping[int | str, str]
    pad_token_id: int | None


class _ModelOutput(Protocol):
    logits: _Tensor


class _Model(Protocol):
    config: _ModelConfig

    def eval(self) -> object: ...
    def to(self, device: str) -> object: ...
    def __call__(self, **kwargs: _Tensor) -> _ModelOutput: ...


class _Factory(Protocol):
    def from_pretrained(self, model_id: str, **kwargs: object) -> object: ...


class _Transformers(Protocol):
    AutoTokenizer: _Factory
    AutoModelForSequenceClassification: _Factory
    AutoModelForCausalLM: _Factory


class _Torch(Protocol):
    cuda: _Cuda

    def inference_mode(self) -> AbstractContextManager[object]: ...


class _Cuda(Protocol):
    def is_available(self) -> bool: ...


def _label_names(
    id2label: Mapping[int | str, str], label_alias: Mapping[str, str] | None
) -> list[str]:
    by_id: dict[int, str] = {}
    for key, value in id2label.items():
        if (
            isinstance(key, bool)
            or not str(key).isdigit()
            or not isinstance(cast("object", value), str)
        ):
            raise ValidationError("id2label must map contiguous integer class ids to names")
        index = int(key)
        if index in by_id:
            raise ValidationError("id2label repeats a class id")
        by_id[index] = value
    if not by_id or set(by_id) != set(range(len(by_id))):
        raise ValidationError("id2label must map contiguous integer class ids to names")
    names = [by_id[index] for index in range(len(by_id))]
    aliases = dict(label_alias or {})
    if aliases.keys() - set(names):
        raise ValidationError("label_alias contains a name absent from id2label")
    for name in names:
        if re.fullmatch(r"LABEL_\d+", name, re.IGNORECASE) and name not in aliases:
            raise ValidationError("generic id2label names require an explicit label_alias")
    return _validate_labels([aliases.get(name, name) for name in names])


def _validate_labels(names: list[str]) -> list[str]:
    if any(not isinstance(cast("object", name), str) for name in names):
        raise ValidationError("labels must be strings")
    if any(not name.strip() or len(name) > 64 for name in names):
        raise ValidationError("labels must contain 1 to 64 characters")
    if not names or len(set(names)) != len(names):
        raise ValidationError("labels must be nonempty and unique")
    return names


def _verbalizer_ids(tokenizer: _Tokenizer, label_tokens: Mapping[str, str]) -> list[int]:
    _validate_labels(list(label_tokens))
    ids: list[int] = []
    for text in label_tokens.values():
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValidationError("each label verbalizer must encode to exactly one token")
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        raise ValidationError("label verbalizers must use distinct token ids")
    return ids


def _softmax(values: Sequence[float]) -> list[float]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValidationError("model logits must be finite and nonempty")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    denominator = math.fsum(weights)
    return [weight / denominator for weight in weights]


def _label_probabilities(logits: Sequence[float], ids: Sequence[int]) -> tuple[list[float], float]:
    if not logits or any(not math.isfinite(value) for value in logits):
        raise ValidationError("model logits must be finite and nonempty")
    if any(index < 0 or index >= len(logits) for index in ids):
        raise ValidationError("a verbalizer token id is outside the model vocabulary")
    selected = [logits[index] for index in ids]
    probabilities = _softmax(selected)
    # Sum the complement directly: 1 - label_mass loses tiny outside probabilities.
    maximum = max(logits)
    weights = [math.exp(value - maximum) for value in logits]
    selected_ids = set(ids)
    outside = math.fsum(value for index, value in enumerate(weights) if index not in selected_ids)
    return probabilities, outside / math.fsum(weights)


def _vector(tensor: _Tensor) -> list[float]:
    raw = tensor.float().detach().cpu().tolist()
    if not isinstance(raw, list):
        raise ValidationError("model logits have an unexpected shape")
    values = cast("list[object]", raw)
    if any(isinstance(value, bool) or not isinstance(value, int | float) for value in values):
        raise ValidationError("model logits have an unexpected shape")
    return [float(cast("int | float", value)) for value in values]


def _check_untruncated_features(
    features: Mapping[str, _Tensor], batch: Sequence[tuple[int, int, bool]], max_length: int
) -> None:
    ids = features.get("input_ids")
    mask = features.get("attention_mask")
    if (
        ids is None
        or mask is None
        or len(ids.shape) != 2
        or ids.shape[0] != len(batch)
        or ids.shape[1] > max_length
        or tuple(mask.shape) != tuple(ids.shape)
    ):
        raise ValidationError("tokenization_mismatch: untruncated input shape required")
    width = ids.shape[1]
    for index, (length, _, _) in enumerate(batch):
        if length > width or _vector(mask[index]) != [1.0] * length + [0.0] * (width - length):
            raise ValidationError("tokenization_mismatch: complete right-padded input required")


class Scorer:
    """A private loaded model with synchronous, length-batched forward passes."""

    def __init__(
        self,
        *,
        engine: ScoringEngine,
        model: _Model,
        tokenizer: _Tokenizer,
        torch: _Torch,
        labels: list[str],
        token_ids: list[int],
        max_length: int,
        device: str,
        loading_audit: LoadingAudit,
        padding_policy: PaddingPolicy,
        classifier_single_sequence: bool = False,
        classifier_pair_allowed: bool = True,
    ) -> None:
        self._engine = engine
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self._labels = labels
        self._token_ids = token_ids
        self._max_length = max_length
        self._classifier_single_sequence = classifier_single_sequence
        self._classifier_pair_allowed = classifier_pair_allowed
        self._device = device
        self.loading_audit = loading_audit
        self.padding_policy = padding_policy

    @property
    def labels(self) -> tuple[str, ...]:
        """The complete output label order, including labels absent from refused rows."""
        return tuple(self._labels)

    async def score(
        self,
        rows: Sequence[ScoringInput],
        *,
        batch_size: int = 8,
        on_progress: Callable[[ProgressEvent], None] | None = None,
        overflow_policy: OverflowPolicy = "legacy",
    ) -> ScoringResult:
        """Score off-loop; error overflow policy refuses the call before any model forward."""
        if type(overflow_policy) is not str or overflow_policy not in ("legacy", "error"):
            raise ValidationError("unsupported overflow_policy")
        if batch_size < 1:
            raise ValidationError("batch_size must be positive")
        return await asyncio.to_thread(
            self._score, list(rows), batch_size, on_progress, overflow_policy
        )

    async def score_with_outcomes(
        self,
        rows: Sequence[ScoringInput],
        *,
        overflow: Literal["drop"],
        batch_size: int = 8,
        on_progress: Callable[[ProgressEvent], None] | None = None,
    ) -> ScoringOutcomesV2:
        """Compute native drop evidence; every retained forward uses complete uncut inputs."""
        if type(overflow) is not str or overflow != "drop":
            raise ValidationError("unsupported_overflow_policy")
        if type(batch_size) is not int or not 1 <= batch_size <= 256:
            raise ValidationError("scoring_batch_size_invalid")
        if len(rows) > MAX_POPULATION:
            raise ValidationError("population_limit_exceeded")
        active = True

        def progress(event: ProgressEvent) -> None:
            if active and on_progress is not None:
                on_progress(event)

        try:
            return await asyncio.to_thread(self._score_outcomes, list(rows), batch_size, progress)
        finally:
            # Cancellation cannot stop an existing forward, but its late callback cannot
            # report completion for this abandoned attempt or a replacement caller.
            active = False

    def _input_lengths(self, row: ScoringInput) -> tuple[int, int, int, int]:
        tokenizer = self._tokenizer
        first_length = len(tokenizer.encode(row.text, add_special_tokens=False))
        pair_length = (
            len(tokenizer.encode(row.text_pair, add_special_tokens=False))
            if row.text_pair is not None
            else 0
        )
        if (
            self._engine == "hf_classify"
            and row.text_pair is None
            and not self._classifier_single_sequence
        ):
            raise ValidationError("hf_classify requires text_pair for every row")
        if (
            self._engine == "hf_classify"
            and row.text_pair is not None
            and not self._classifier_pair_allowed
        ):
            raise ValidationError("Forge decoder classifiers require a single rendered input")
        if self._engine == "hf_label_logits" and row.text_pair is not None:
            raise ValidationError("hf_label_logits accepts a single rendered prompt")
        specials = tokenizer.num_special_tokens_to_add(pair=row.text_pair is not None)
        length = first_length + pair_length + specials
        return first_length, pair_length, specials, length

    def _score_outcomes(
        self,
        rows: list[ScoringInput],
        batch_size: int,
        on_progress: Callable[[ProgressEvent], None] | None,
    ) -> ScoringOutcomesV2:
        if self._engine == "hf_classify" and len({row.text_pair is None for row in rows}) > 1:
            raise ValidationError("mixed classifier input shapes are unsupported")
        lengths = [self._input_lengths(row)[3] for row in rows]
        accepted = [index for index, length in enumerate(lengths) if 0 < length <= self._max_length]
        empty = [index for index, length in enumerate(lengths) if length == 0]
        dropped = sum(length > self._max_length for length in lengths)

        def progress(event: ProgressEvent) -> None:
            if on_progress is not None:
                complete = event.step or 0
                failed = event.metrics.get("failed", 0.0) if event.metrics else 0.0
                on_progress(
                    ProgressEvent(
                        kind="step",
                        stage="run",
                        step=dropped + len(empty) + complete,
                        total_steps=len(rows),
                        metrics={
                            "rows_overflow_dropped": float(dropped),
                            "rows_scored_success": complete - failed,
                            "rows_failed_or_invalid": failed + len(empty),
                        },
                    )
                )

        scored = self._score([rows[index] for index in accepted], batch_size, progress, "error")
        by_original = {
            accepted[row.index]: row.model_copy(update={"index": accepted[row.index]})
            for row in scored.rows
        }
        by_original.update(
            {
                index: ScoredRow(index=index, error="empty_prompt: scoring needs an attended token")
                for index in empty
            }
        )
        outcomes = tuple(
            ScoringItemOutcomeV2(
                index=index,
                outcome="overflow"
                if index not in by_original
                else "invalid"
                if by_original[index].error
                else "scored",
                token_count=length,
                model_limit=self._max_length,
                model_sent=index in by_original and by_original[index].error is None,
                result=by_original.get(index),
            )
            for index, length in enumerate(lengths)
        )
        result = ScoringOutcomesV2(
            rows=outcomes,
            overflow_rows=dropped,
            failed_rows=scored.failed_rows + len(empty),
            scored_rows=len(accepted) - scored.failed_rows,
        )
        if on_progress is not None:
            on_progress(
                ProgressEvent(
                    kind="step",
                    stage="run",
                    step=len(rows),
                    total_steps=len(rows),
                    metrics={
                        "rows_overflow_dropped": float(dropped),
                        "rows_scored_success": float(result.scored_rows),
                        "rows_failed_or_invalid": float(result.failed_rows),
                    },
                )
            )
        return result

    def _score(
        self,
        rows: list[ScoringInput],
        batch_size: int,
        on_progress: Callable[[ProgressEvent], None] | None,
        overflow_policy: OverflowPolicy,
    ) -> ScoringResult:
        tokenizer = self._tokenizer
        results: dict[int, ScoredRow] = {}
        pending: list[tuple[int, int, bool]] = []
        overflow_count = 0
        truncated_count = 0
        failed_count = 0
        if self._engine == "hf_classify" and len({row.text_pair is None for row in rows}) > 1:
            raise ValidationError("mixed classifier input shapes are unsupported")
        for index, row in enumerate(rows):
            first_length, pair_length, specials, length = self._input_lengths(row)
            overflow = length > self._max_length
            if overflow and overflow_policy == "error":
                raise ValidationError("input_too_long: overflow_policy=error refuses scoring")
            error: str | None = None
            if self._engine == "hf_classify" and row.text_pair is None:
                if overflow:
                    error = "prompt_too_long: rendered classifier input cannot be truncated"
                elif length == 0:
                    error = "empty_prompt: classifier needs an attended position"
            elif self._engine == "hf_classify":
                # only_first cannot remove the entire first sequence in every HF tokenizer.
                if pair_length + specials > self._max_length or (
                    first_length > 0 and pair_length + specials >= self._max_length
                ):
                    error = "hypothesis_too_long: hypothesis cannot fit without truncation"
            elif overflow:
                error = "prompt_too_long: decoder scoring never truncates the prompt"
            elif length == 0:
                error = "empty_prompt: decoder scoring needs a label position"
            if error is not None:
                results[index] = ScoredRow(index=index, overflow=overflow, error=error)
                overflow_count += overflow
                failed_count += 1
            else:
                pending.append((min(length, self._max_length), index, overflow))

        pending.sort()
        self._progress(
            len(results), len(rows), overflow_count, truncated_count, failed_count, on_progress
        )
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            inputs = [rows[index].text for _, index, _ in batch]
            pairs = (
                [cast("str", rows[index].text_pair) for _, index, _ in batch]
                if self._engine == "hf_classify" and rows[batch[0][1]].text_pair is not None
                else None
            )
            tokenize_kwargs: dict[str, object] = {
                "padding": True,
                "truncation": "only_first"
                if pairs is not None and overflow_policy == "legacy"
                else False,
                "return_tensors": "pt",
                "return_attention_mask": True,
            }
            if pairs is not None and overflow_policy == "legacy":
                tokenize_kwargs["max_length"] = self._max_length
            features = tokenizer(inputs, pairs, **tokenize_kwargs)
            if overflow_policy == "error":
                _check_untruncated_features(features, batch, self._max_length)
            with self._torch.inference_mode():
                output = self._model(
                    **{key: value.to(self._device) for key, value in features.items()}
                )
            expected_rank = 2 if self._engine == "hf_classify" else 3
            if len(output.logits.shape) != expected_rank or output.logits.shape[0] != len(batch):
                raise ValidationError("model logits have an unexpected shape")
            if self._engine == "hf_label_logits" and output.logits.shape[1] < max(
                length for length, _, _ in batch
            ):
                raise ValidationError("model logits omit a prompt's label position")
            for batch_index, (length, index, overflow) in enumerate(batch):
                mass_outside: float | None = None
                if self._engine == "hf_classify":
                    values = _vector(output.logits[batch_index])
                    if len(values) != len(self._labels):
                        raise ValidationError("model logits do not match id2label")
                    probabilities = _softmax(values)
                else:
                    # Right padding keeps each prompt's positions unchanged. Read its last
                    # real token, never the padding position of a longer batch neighbour.
                    values = _vector(output.logits[batch_index, length - 1, :])
                    probabilities, mass_outside = _label_probabilities(values, self._token_ids)
                results[index] = ScoredRow(
                    index=index,
                    probabilities=dict(zip(self._labels, probabilities, strict=True)),
                    pred=self._labels[
                        max(range(len(probabilities)), key=probabilities.__getitem__)
                    ],
                    mass_outside=mass_outside,
                    overflow=overflow,
                    truncated=overflow,
                )
                overflow_count += overflow
                truncated_count += overflow
            self._progress(
                len(results), len(rows), overflow_count, truncated_count, failed_count, on_progress
            )
        ordered = [results[index] for index in range(len(rows))]
        return ScoringResult(
            rows=ordered,
            overflow_rows=overflow_count,
            truncated_rows=truncated_count,
            failed_rows=failed_count,
        )

    @staticmethod
    def _progress(
        completed: int,
        total: int,
        overflow: int,
        truncated: int,
        failed: int,
        sink: Callable[[ProgressEvent], None] | None,
    ) -> None:
        if sink is not None and completed:
            sink(
                ProgressEvent(
                    kind="step",
                    stage="run",
                    step=completed,
                    total_steps=total,
                    metrics={
                        "overflow_rows": float(overflow),
                        "truncated_rows": float(truncated),
                        "failed": float(failed),
                        "succeeded": float(completed - failed),
                    },
                )
            )


async def load_scorer(
    model_id: str,
    *,
    engine: ScoringEngine,
    token: HubCredential,
    max_length: int,
    revision: str | None = None,
    label_alias: Mapping[str, str] | None = None,
    label_tokens: Mapping[str, str] | None = None,
    device: str = "cpu",
    dtype: str | None = None,
) -> Scorer:
    """Load built-in HF model/tokenizer code with explicit authentication and safe weights."""
    if not REPO_ID_RE.fullmatch(model_id):
        raise ValidationError("model_id must be a Hub repo id, not a local path")
    if token is not False and (not isinstance(cast("object", token), str) or not token):
        raise ValidationError("token must be an explicit credential or False")
    if max_length < 1:
        raise ValidationError("max_length must be positive")
    if dtype not in (None, "auto", "float32", "float16", "bfloat16"):
        raise ValidationError("unsupported scoring dtype")
    if engine not in ("hf_classify", "hf_label_logits"):
        raise ValidationError("unsupported scoring engine")
    if engine == "hf_classify" and label_tokens is not None:
        raise ValidationError("hf_classify uses id2label, not label_tokens")
    if engine == "hf_label_logits" and (not label_tokens or label_alias is not None):
        raise ValidationError("hf_label_logits requires label_tokens and does not use label_alias")
    if label_alias:
        _validate_labels(list(label_alias))
        _validate_labels(list(label_alias.values()))
    if label_tokens:
        _validate_labels(list(label_tokens))
    return await asyncio.to_thread(
        _load_scorer,
        model_id,
        engine,
        token,
        max_length,
        revision,
        dict(label_alias) if label_alias is not None else None,
        dict(label_tokens) if label_tokens is not None else None,
        device,
        dtype,
    )


def _load_scorer(
    model_id: str,
    engine: ScoringEngine,
    token: HubCredential,
    max_length: int,
    revision: str | None,
    label_alias: Mapping[str, str] | None,
    label_tokens: Mapping[str, str] | None,
    device: str,
    dtype: str | None,
) -> Scorer:
    # Transformers prefers an existing relative directory over the requested Hub revision.
    # Refuse that ambiguity rather than label local bytes with a resolved Hub commit.
    if Path(model_id).exists():
        raise ValidationError("model_id collides with a local path; Hub-only scoring is required")
    try:
        transformers = cast("_Transformers", import_module("transformers"))
        torch = cast("_Torch", import_module("torch"))
    except ImportError as exc:
        raise ImportError(
            "HF scoring requires torch and transformers; install 'strata-forge[finetuning]'."
        ) from exc
    forge_checkpoint = (
        probe_forge_classifier(model_id, revision=revision, token=token)
        if engine == "hf_classify"
        and revision is not None
        and re.fullmatch(r"[a-f0-9]{40}", revision)
        else None
    )
    tokenizer = cast(
        "_Tokenizer",
        forge_checkpoint.tokenizer
        if forge_checkpoint is not None
        else transformers.AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            token=token,
            trust_remote_code=False,
        ),
    )
    tokenizer.padding_side = "right"
    pad_source: Literal["tokenizer_pad", "eos_fallback"] = "tokenizer_pad"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValidationError("batched scoring requires a pad token or an EOS token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        pad_source = "eos_fallback"
    pad_id = cast("object", tokenizer.pad_token_id)
    if isinstance(pad_id, bool) or not isinstance(pad_id, int) or pad_id < 0:
        raise ValidationError("batched scoring requires a nonnegative integer pad token id")
    ids = _verbalizer_ids(tokenizer, label_tokens) if label_tokens is not None else []
    factory = (
        transformers.AutoModelForSequenceClassification
        if engine == "hf_classify"
        else transformers.AutoModelForCausalLM
    )
    kwargs: dict[str, object] = {
        "revision": revision,
        "token": token,
        "use_safetensors": True,
        "trust_remote_code": False,
        "output_loading_info": True,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    if forge_checkpoint is None:
        try:
            loaded = factory.from_pretrained(model_id, **kwargs)
        except OSError, RuntimeError, ValueError:
            raise ModelLoadingError(LoadingAudit.refusal("pretrained_load_failed")) from None
        checked_model, loading_audit = _guard_loading_result(loaded)
    else:
        checked_model = forge_checkpoint.model
        if dtype not in (None, "auto"):
            forge_checkpoint.model.to(dtype=getattr(torch, dtype))
        # The fixed SDK loader already refused every structural diagnostic. This is a
        # successful complete-load audit, never an exemption for initialized training heads.
        loading_audit = LoadingAudit(status="accepted")
    model = cast("_Model", checked_model)
    original_model_pad = model.config.pad_token_id
    # Decoder classifier heads pool using config.pad_token_id independently of the
    # tokenizer's attention mask. Both must describe the padding actually supplied.
    try:
        model.config.pad_token_id = pad_id
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValidationError("model config cannot use the tokenizer pad token id") from exc
    if model.config.pad_token_id != pad_id:
        raise ValidationError("model config must preserve the tokenizer pad token id")
    labels = (
        _label_names(model.config.id2label, label_alias)
        if engine == "hf_classify"
        else list(label_tokens or {})
    )
    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(resolved_device)
    model.eval()
    return Scorer(
        engine=engine,
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        labels=labels,
        token_ids=ids,
        max_length=max_length,
        device=resolved_device,
        loading_audit=loading_audit,
        classifier_single_sequence=forge_checkpoint is not None,
        classifier_pair_allowed=forge_checkpoint is None
        or forge_checkpoint.model.config.model_type == "bert",
        padding_policy=PaddingPolicy(
            source=pad_source,
            pad_token_id=pad_id,
            original_model_pad_token_id=original_model_pad,
        ),
    )
