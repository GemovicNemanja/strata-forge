"""Sequence-classification fine-tuning: a premise/hypothesis classifier on ``transformers.Trainer``.

:class:`SeqClsConfig` declares the run; :class:`SeqClsRunner` executes it from a local checkpoint
directory and a dataset whose columns are the ``nli_pair`` roles
(:mod:`strata_forge.training.dataset_format`). The objective is
:func:`strata_forge.training.classifier_loss.classifier_loss`; heads, the load guard and the
decoder LM-head initialisation are :mod:`strata_forge.training.classifier_heads`.

What the runner checks before it spends a step, in order, each a named refusal:

1. **Labels and rendering.** Every row's label resolves through the declared label order, alias
   map and (for integer labels) index map; every row renders and tokenises with no truncation, and
   a row longer than ``max_length`` refuses the run (*overflow preflight*), with the count and the
   first row indices. The rendering's token overhead is recorded.
2. **The load guard.** Only the declared head keys may be missing (see ``head_init``).
3. **The step-0 check** (``decoder_lm_head``): on the first training rows, the classifier's logits
   must equal the base model's own LM-head logits for the verbalizer tokens, within
   ``step0_tolerance``. The receipt records the largest difference.
4. **The memory preflight.** One full optimizer step at the split's longest real batch (the
   ``per_device_batch_size`` longest rows), measured, then undone exactly; a peak above
   ``memory_fraction`` of the device, or an out-of-memory error, refuses the run with the measured
   peak.

Gradient accumulation is refused (a declared batch is the loss's denominator); activation
checkpointing is a declared flag (``torch.utils.checkpoint`` with ``use_reentrant=False`` and
``preserve_rng_state=True``), off by default. Mixed precision is bf16 autocast over float32 master
weights. Training batches are grouped by length.

The saved artifact is a stock ``AutoModelForSequenceClassification`` checkpoint (safetensors) with
its tokenizer, the auxiliary heads in :data:`~strata_forge.training.classifier_heads.AUX_HEADS_FILENAME`,
and :data:`VERIFIER_CONFIG_FILENAME`, the rendering an evaluator must reproduce. Intermediate
checkpoints go to ``output_dir``, which is never the artifact directory.

Heavy imports (``torch``, ``transformers``, ``datasets``) happen inside the runner.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import math
import re
import time
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from strata_forge.training.classifier_heads import (
    AUX_HEADS_FILENAME,
    DECODER_LM_HEAD,
    ENCODER_NEW_HEAD,
    HeadLoadError,
    LoadReceipt,
    aux_outputs,
    build_aux_heads,
    check_load,
    init_decoder_head,
    load_aux_heads,
    save_aux_heads,
)
from strata_forge.training.classifier_loss import LOSS_FORM, LossWeights, classifier_loss
from strata_forge.training.dataset_format import DatasetFormatError
from strata_forge.training.loading import (
    LoadRefusedError,
    check_loading_info,
    model_load_kwargs,
    tokenizer_load_kwargs,
)
from strata_forge.training.progress import attach as _attach_progress
from strata_forge.training.progress import coerce_int, numeric_metrics
from strata_forge.training.verifier_render import (
    RenderedRow,
    Rendering,
    encode_rows,
    fill_template,
    marker_tokens,
    render_premises,
    render_row,
)

if TYPE_CHECKING:
    from pathlib import Path

    from strata_forge.training.peft import LoRAConfig, QLoRAConfig

__all__ = [
    "CALIBRATION_FILENAME",
    "MAX_SEQ_CLS_ROWS",
    "VERIFIER_CONFIG_FILENAME",
    "ArtifactCheckError",
    "CudaMemoryProbe",
    "MemoryProbe",
    "PreflightError",
    "PreparedRun",
    "SeqClsConfig",
    "SeqClsResult",
    "SeqClsRunner",
    "Step0Error",
    "render_premises",
]

#: The rendering and label contract an evaluator reads back, beside the weights.
VERIFIER_CONFIG_FILENAME = "strata_verifier.json"
#: The largest split a run trains on. The split is memory-mapped Arrow, never a Python list, so
#: the bound is the supported population rather than a memory guard.
MAX_SEQ_CLS_ROWS = 600_000

_LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_]+)\}")
_ROLES_PRIMARY = ("premises", "hypothesis", "label")
_TWO_WAY = "2way_entail_vs_not"
_FULL = "3way"
_MAX_REPORTED_ROWS = 5


class PreflightError(RuntimeError):
    """A run refused before training. ``receipt`` holds the measured facts for the manifest."""

    def __init__(self, message: str, receipt: Mapping[str, Any], *, check: str = "") -> None:
        super().__init__(message)
        self.receipt = dict(receipt)
        #: Which check refused: ``overflow``, ``load``, ``step0``, ``memory`` or ``artifact``.
        self.check = check


class Step0Error(PreflightError):
    """The decoder head does not reproduce the base model's zero-shot label logits at step 0."""


class ArtifactCheckError(PreflightError):
    """The saved artifact does not reload to the classifier that was trained."""


#: The temperature fit and its record, beside the weights.
CALIBRATION_FILENAME = "calibration.json"
_HEAD_PREFIXES = ("classifier.", "score.", "head.")
_CLASSIFIER_BIAS_REASON = (
    "the stock decoder classifier has no bias, and the head starts from the LM head's output "
    "projection, whose bias is nonzero; classifier_bias=true carries it, and gives head.dense a "
    "bias that is held at exactly zero"
)


class SeqClsConfig(BaseModel):
    """A sequence-classification run. ``model_id``, ``output_dir`` and ``progress_jsonl`` are set by
    whoever drives the runner; everything else is the run's declaration.

    Attributes:
        labels: The label order: logit ``k`` is ``labels[k]``.
        label_aliases: Dataset spelling -> label, for three-way rows.
        label_index_map: Integer label -> label. Required for integer labels, because index orders
            differ between datasets.
        two_way_positive_label: The label whose probability a two-way row's positive is; the
            negative is the sum over every other label. No default: a split with two-way rows
            needs it declared, with ``two_way_labels``.
        two_way_labels: How a two-way row spells its positive and its negative, e.g.
            ``("entailed", "not_entailed")``. A two-way row's label must be one of the two.
        head_init: ``new`` (encoder, fresh classifier), ``decoder_lm_head`` (decoder, classifier
            from its own LM head) or ``existing`` (a checkpoint that already has this classifier).
        verbalizers: Label -> its single-token verbalizer, for ``decoder_lm_head``.
        premise_format: ``ids`` renders each premise as ``"{pid}: {text}"`` (the support head
            needs it); ``plain`` renders the text alone.
        decoder_template: How a decoder sees a row; ``{premises}`` and ``{hypothesis}`` once each.
            The answer position is the final token. Encoders always use the tokenizer's pair
            encoding.
        max_length: Token cap; a longer row refuses the run, it is never truncated.
        gradient_checkpointing: Activation checkpointing (non-reentrant, RNG state preserved).
        brier_weight / flag_loss_weight / support_loss_weight: The objective's coefficients.
        memory_fraction: The share of device memory the preflight peak may reach.
        step0_rows / step0_tolerance: The decoder step-0 check's sample and its absolute bound on
            any label logit's difference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    output_dir: str
    progress_jsonl: str | None = None
    labels: tuple[str, ...]
    label_aliases: dict[str, str] = Field(default_factory=dict[str, str])
    label_index_map: dict[str, str] | None = None
    two_way_positive_label: str | None = None
    two_way_labels: tuple[str, str] | None = None
    head_init: Literal["new", "decoder_lm_head", "existing"] = "new"
    verbalizers: dict[str, str] | None = None
    premise_format: Literal["ids", "plain"] = "ids"
    decoder_template: str = "{premises}\n{hypothesis}"
    max_length: int = Field(default=1024, ge=8, le=32768)
    per_device_batch_size: int = Field(default=32, ge=1, le=4096)
    eval_batch_size: int | None = Field(default=None, ge=1, le=4096)
    num_epochs: float = Field(default=1.0, gt=0)
    max_steps: int | None = Field(default=None, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    lr_scheduler: Literal["linear", "cosine", "constant", "constant_with_warmup"] = "linear"
    warmup_ratio: float = Field(default=0.0, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    precision: Literal["fp32", "bf16"] = "bf16"
    gradient_checkpointing: bool = False
    group_by_length: bool = True
    brier_weight: float = Field(default=0.0, ge=0.0, le=100.0)
    flag_loss_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    support_loss_weight: float = Field(default=1.0, ge=0.0, le=100.0)
    logging_steps: int = Field(default=10, ge=1)
    eval_steps: int | None = Field(default=None, ge=1)
    save_steps: int = Field(default=0, ge=0)
    save_total_limit: int = Field(default=2, ge=1, le=100)
    seed: int = 42
    memory_fraction: float = Field(default=0.92, gt=0.0, le=1.0)
    step0_rows: int = Field(default=8, ge=1, le=64)
    step0_tolerance: float = Field(default=1e-4, gt=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _refuse_accumulation(cls, data: Any) -> Any:
        if isinstance(data, Mapping) and "gradient_accumulation_steps" in data:
            msg = (
                "gradient_accumulation_steps is refused for seq_cls: the objective's means are "
                "over the declared batch, and accumulating would change their denominators. Set "
                "per_device_batch_size to the batch the run means"
            )
            raise ValueError(msg)
        return data  # pyright: ignore[reportUnknownVariableType] - raw input, validated next

    @field_validator("labels")
    @classmethod
    def _valid_labels(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not 2 <= len(value) <= 16:
            msg = "labels must name between 2 and 16 labels"
            raise ValueError(msg)
        if len(set(value)) != len(value):
            msg = "labels must be unique"
            raise ValueError(msg)
        bad = [label for label in value if not _LABEL_RE.fullmatch(label)]
        if bad:
            msg = f"labels must match [A-Za-z0-9_.-]{{1,64}}: {bad}"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _consistent(self) -> SeqClsConfig:
        labels = set(self.labels)
        if len(self.label_aliases) > 256:
            msg = "label_aliases has more than 256 entries"
            raise ValueError(msg)
        stray = sorted({v for v in self.label_aliases.values() if v not in labels})
        if stray:
            msg = f"label_aliases map to labels that are not declared: {stray}"
            raise ValueError(msg)
        if self.label_index_map is not None:
            if not all(k.lstrip("-").isdigit() for k in self.label_index_map):
                msg = "label_index_map keys must be integers"
                raise ValueError(msg)
            unknown = sorted({v for v in self.label_index_map.values() if v not in labels})
            if unknown:
                msg = f"label_index_map maps to labels that are not declared: {unknown}"
                raise ValueError(msg)
        if (self.two_way_positive_label is None) != (self.two_way_labels is None):
            msg = "two_way_positive_label and two_way_labels are declared together, or not at all"
            raise ValueError(msg)
        if self.two_way_positive_label is not None and self.two_way_positive_label not in labels:
            msg = "two_way_positive_label must be one of labels"
            raise ValueError(msg)
        if self.two_way_labels is not None:
            spelled = self.two_way_labels
            if spelled[0] == spelled[1] or not all(_LABEL_RE.fullmatch(x) for x in spelled):
                msg = "two_way_labels are two distinct [A-Za-z0-9_.-]{1,64} spellings"
                raise ValueError(msg)
        if self.head_init == "decoder_lm_head":
            if self.verbalizers is None or set(self.verbalizers) != labels:
                msg = "decoder_lm_head needs a verbalizer for every label, and only for labels"
                raise ValueError(msg)
        elif self.verbalizers is not None:
            msg = "verbalizers apply only to head_init 'decoder_lm_head'"
            raise ValueError(msg)
        found = _PLACEHOLDER_RE.findall(self.decoder_template)
        if sorted(found) != ["hypothesis", "premises"]:
            msg = "decoder_template must contain {premises} and {hypothesis} exactly once each"
            raise ValueError(msg)
        return self

    @property
    def positive_index(self) -> int:
        """The logit a two-way row's positive reads; unused (0) when no two-way row is declared."""
        if self.two_way_positive_label is None:
            return 0
        return self.labels.index(self.two_way_positive_label)

    @property
    def loss_weights(self) -> LossWeights:
        return LossWeights(
            brier_weight=self.brier_weight,
            flag_weight=self.flag_loss_weight,
            support_weight=self.support_loss_weight,
        )

    @property
    def from_lm_head(self) -> bool:
        """Whether the classifier is initialised from the base's LM head (and step 0 checked)."""
        return self.head_init == "decoder_lm_head"


@dataclass
class SeqClsResult:
    """What a finished run reports: where the artifact is, its numbers, and every receipt."""

    output_dir: str
    metrics: dict[str, float]
    steps: int | None
    receipts: dict[str, Any]


class MemoryProbe(Protocol):
    """Measures device memory for the preflight. ``None`` means the device cannot say."""

    def reset(self) -> None: ...
    def peak_bytes(self) -> int | None: ...
    def total_bytes(self) -> int | None: ...


class CudaMemoryProbe:
    """The CUDA allocator's own peak, on the current device; ``None`` off a GPU."""

    def __init__(self, torch_mod: Any) -> None:
        self._torch = torch_mod

    def _cuda(self) -> bool:
        return bool(self._torch.cuda.is_available())

    def reset(self) -> None:
        if self._cuda():
            self._torch.cuda.synchronize()
            self._torch.cuda.empty_cache()
            self._torch.cuda.reset_peak_memory_stats()

    def peak_bytes(self) -> int | None:
        if not self._cuda():
            return None
        self._torch.cuda.synchronize()
        return int(self._torch.cuda.max_memory_allocated())

    def total_bytes(self) -> int | None:
        if not self._cuda():
            return None
        return int(self._torch.cuda.get_device_properties(0).total_memory)


# ----------------------------------------------------------------------------- rendering


@dataclass
class _Encoder:
    """Turns role rows into model inputs and targets. One instance per run; pure Python."""

    config: SeqClsConfig
    tokenizer: Any
    roles: frozenset[str]
    rendering: Rendering

    def _label(self, raw: Any, row: int) -> str:
        if isinstance(raw, bool) or raw is None:
            msg = f"row {row}: label must be a string or an integer"
            raise DatasetFormatError(msg)
        if isinstance(raw, int):
            mapping = self.config.label_index_map
            if mapping is None or str(raw) not in mapping:
                msg = (
                    f"row {row}: an integer label has no label_index_map entry; integer "
                    "labels need an explicit index map"
                )
                raise DatasetFormatError(msg)
            return mapping[str(raw)]
        if not isinstance(raw, str):
            msg = f"row {row}: label must be a string or an integer"
            raise DatasetFormatError(msg)
        if raw in self.config.labels:
            return raw
        if raw in self.config.label_aliases:
            return self.config.label_aliases[raw]
        msg = f"row {row}: label is neither a declared label nor an alias"
        raise DatasetFormatError(msg)

    def _two_way(self, raw: Any, row: int) -> float:
        spelled = self.config.two_way_labels
        if spelled is None:
            msg = (
                f"row {row}: a two-way row needs two_way_positive_label and two_way_labels "
                "declared; there is no default"
            )
            raise DatasetFormatError(msg)
        if raw == spelled[0]:
            return 1.0
        if raw == spelled[1]:
            return 0.0
        msg = f"row {row}: a two-way label must be one of the declared two_way_labels"
        raise DatasetFormatError(msg)

    def _targets(self, batch: Mapping[str, list[Any]], i: int, row: int) -> dict[str, Any]:
        labels = self.config.labels
        space: Any = batch["label_space"][i] if "label_space" in self.roles else _FULL
        if space not in (_FULL, _TWO_WAY):
            msg = f"row {row}: label_space must be {_FULL!r} or {_TWO_WAY!r}"
            raise DatasetFormatError(msg)
        target = [0.0] * len(labels)
        if space == _TWO_WAY:
            return {
                "target": target,
                "two_way": 1,
                "two_way_target": self._two_way(batch["label"][i], row),
            }
        canonical = self._label(batch["label"][i], row)
        dist: Any = batch["label_dist"][i] if "label_dist" in self.roles else None
        if dist is None:
            target[labels.index(canonical)] = 1.0
        else:
            shaped = (
                isinstance(dist, Sequence)
                and not isinstance(dist, str | bytes)
                and len(dist) == len(labels)  # pyright: ignore[reportUnknownArgumentType]
                and all(
                    isinstance(x, int | float) and not isinstance(x, bool)
                    for x in dist  # pyright: ignore[reportUnknownVariableType]
                )
            )
            if not shaped:
                msg = f"row {row}: label_dist must hold {len(labels)} numbers"
                raise DatasetFormatError(msg)
            values = [float(x) for x in dist]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            if (
                not all(math.isfinite(v) for v in values)
                or min(values) < 0
                or abs(sum(values) - 1.0) > 1e-6
            ):
                msg = f"row {row}: label_dist must be finite, non-negative and sum to 1"
                raise DatasetFormatError(msg)
            target = values
        return {"target": target, "two_way": 0, "two_way_target": 0.0}

    def _flag(self, batch: Mapping[str, list[Any]], i: int, row: int) -> float:
        if "inconsistent_premises" not in self.roles:
            return -1.0
        value: Any = batch["inconsistent_premises"][i]
        if value is None:
            return -1.0
        if isinstance(value, bool) or value in (0, 1):
            return float(value)
        msg = f"row {row}: inconsistent_premises must be true, false or null"
        raise DatasetFormatError(msg)

    def encode(self, batch: Mapping[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        """``datasets.Dataset.map(batched=True, with_indices=True)`` body."""
        out: dict[str, list[Any]] = {
            key: []
            for key in (
                "input_ids",
                "length",
                "target",
                "two_way",
                "two_way_target",
                "flag",
                "support_positions",
                "support_target",
                "support_row",
            )
        }
        rendered: list[RenderedRow] = []
        for i in range(len(indices)):
            try:
                rendered.append(
                    render_row(self.rendering, batch["premises"][i], batch["hypothesis"][i])
                )
            except DatasetFormatError as exc:
                msg = f"row {indices[i]}: {exc}"
                raise DatasetFormatError(msg) from exc
        encoded: Any = encode_rows(self.tokenizer, rendered, return_offsets_mapping=True)
        for i, row in enumerate(indices):
            ids: list[int] = list(encoded["input_ids"][i])
            offsets: list[tuple[int, int]] = [tuple(o) for o in encoded["offset_mapping"][i]]  # pyright: ignore[reportAssignmentType] - tokenizer output
            sequence_ids: list[int | None] = encoded.sequence_ids(i)
            out["input_ids"].append(ids)
            out["length"].append(len(ids))
            for key, value in self._targets(batch, i, row).items():
                out[key].append(value)
            out["flag"].append(self._flag(batch, i, row))
            positions, support_target, support_row = self._support(
                batch, i, row, rendered[i].markers, offsets, sequence_ids
            )
            out["support_positions"].append(positions)
            out["support_target"].append(support_target)
            out["support_row"].append(support_row)
        return out

    def _support(
        self,
        batch: Mapping[str, list[Any]],
        i: int,
        row: int,
        markers: list[tuple[str, int]],
        offsets: list[tuple[int, int]],
        sequence_ids: list[int | None],
    ) -> tuple[list[int], list[float], int]:
        if "support_set" not in self.roles:
            return [], [], 0
        value: Any = batch["support_set"][i]
        if value is None:
            return [], [], 0
        if not isinstance(value, Sequence) or isinstance(value, str):
            msg = f"row {row}: support_set must be a list of premise ids or null"
            raise DatasetFormatError(msg)
        if not markers:
            msg = f"row {row}: support_set needs premises with ids (premise_format 'ids', a list)"
            raise DatasetFormatError(msg)
        wanted = {str(x) for x in value}  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        known = {pid for pid, _ in markers}
        if not wanted <= known:
            msg = f"row {row}: support_set names a premise id the row does not have"
            raise DatasetFormatError(msg)
        positions = marker_tokens(markers, offsets, sequence_ids)
        if positions is None:
            msg = f"row {row}: no token covers a premise id's first character"
            raise DatasetFormatError(msg)
        targets = [1.0 if pid in wanted else 0.0 for pid, _ in markers]
        return positions, targets, 1


# ----------------------------------------------------------------------------- runner


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _modules() -> tuple[Any, Any]:
    try:
        torch_mod: Any = __import__("torch")
        transformers_mod: Any = __import__("transformers")
    except ImportError as exc:
        msg = (
            "The [finetuning] extra is required for SeqClsRunner. "
            "Install it with: pip install 'strata-forge[finetuning]'."
        )
        raise ImportError(msg) from exc
    return torch_mod, transformers_mod


def _collate(pad_id: int, torch: Any) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        width = max(len(f["input_ids"]) for f in features)
        slots = max([len(f["support_positions"]) for f in features] + [1])
        ids = [list(f["input_ids"]) + [pad_id] * (width - len(f["input_ids"])) for f in features]
        mask = [[1] * len(f["input_ids"]) + [0] * (width - len(f["input_ids"])) for f in features]
        positions: list[list[int]] = []
        targets: list[list[float]] = []
        valid: list[list[int]] = []
        for f in features:
            n = len(f["support_positions"])
            positions.append(list(f["support_positions"]) + [0] * (slots - n))
            targets.append(list(f["support_target"]) + [0.0] * (slots - n))
            valid.append([1] * n + [0] * (slots - n))
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "target": torch.tensor([f["target"] for f in features], dtype=torch.float32),
            "two_way": torch.tensor([f["two_way"] for f in features], dtype=torch.bool),
            "two_way_target": torch.tensor(
                [f["two_way_target"] for f in features], dtype=torch.float32
            ),
            "flag": torch.tensor([f["flag"] for f in features], dtype=torch.float32),
            "support_positions": torch.tensor(positions, dtype=torch.long),
            "support_target": torch.tensor(targets, dtype=torch.float32),
            "support_valid": torch.tensor(valid, dtype=torch.bool),
            "support_row": torch.tensor([f["support_row"] for f in features], dtype=torch.bool),
        }

    return collate


def _classes(torch: Any, transformers: Any) -> tuple[Any, Any]:
    """The module wrapper and the Trainer subclass, built once torch is importable."""
    nn: Any = torch.nn

    class SeqClsModule(nn.Module):  # pyright: ignore[reportUntypedBaseClass] - torch is lazy
        """The classifier plus its auxiliary heads, as one module the Trainer optimises."""

        def __init__(self, classifier: Any, aux: Any, *, last_token: bool) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType] - torch is lazy
            self.classifier: Any = classifier
            self.aux: Any = aux
            self._last_token = last_token
            self._hidden: list[Any] = []
            classifier.base_model.register_forward_hook(self._capture)

        def _capture(self, _module: Any, _args: Any, output: Any) -> None:
            self._hidden.append(output[0])

        def forward(
            self,
            input_ids: Any,
            attention_mask: Any,
            support_positions: Any = None,
            support_valid: Any = None,
            **_ignored: Any,
        ) -> dict[str, Any]:
            self._hidden.clear()
            out: Any = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
            result: dict[str, Any] = {"logits": out.logits}
            if self.aux.flag is None and self.aux.support is None:
                return result
            result.update(
                aux_outputs(
                    torch,
                    self.aux,
                    self._hidden[-1],
                    attention_mask,
                    last_token=self._last_token,
                    support_positions=support_positions,
                )
            )
            return result

    class SeqClsTrainer(transformers.Trainer):  # pyright: ignore[reportUntypedBaseClass] - transformers is lazy
        """``Trainer`` with the classifier objective, length grouping and loss parts in logs."""

        def __init__(
            self, *args: Any, seq_cls: SeqClsConfig, positive_index: int, **kwargs: Any
        ) -> None:
            super().__init__(*args, **kwargs)  # pyright: ignore[reportUnknownMemberType]
            self._seq_cls = seq_cls
            self._positive_index = positive_index
            self._parts: dict[str, float] = {}
            self._parts_count = 0

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch  # the objective's means are over this batch, by contract
            out: dict[str, Any] = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                support_positions=inputs["support_positions"],
                support_valid=inputs["support_valid"],
            )
            terms = classifier_loss(
                out["logits"],
                target=inputs["target"],
                two_way=inputs["two_way"],
                two_way_target=inputs["two_way_target"],
                positive_index=self._positive_index,
                weights=self._seq_cls.loss_weights,
                flag_logits=out.get("flag_logits"),
                flag_target=inputs["flag"] if "flag_logits" in out else None,
                support_logits=out.get("support_logits"),
                support_target=inputs["support_target"],
                support_valid=inputs["support_valid"],
                support_row=inputs["support_row"],
            )
            if model.training:
                # Summed over the logging window and averaged when logged, like `loss` itself.
                for key, value in (
                    ("loss_primary", terms.primary),
                    ("loss_flag", terms.flag),
                    ("loss_support", terms.support),
                ):
                    self._parts[key] = self._parts.get(key, 0.0) + float(value)
                self._parts_count += 1
            if return_outputs:
                return terms.total, {"logits": out["logits"]}
            return terms.total

        def reset_loss_parts(self) -> None:
            """Forget loss parts accumulated outside training (the memory preflight's step)."""
            self._parts, self._parts_count = {}, 0

        def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
            if "loss" in logs and self._parts_count:
                count = self._parts_count
                logs = {**logs, **{k: v / count for k, v in self._parts.items()}}
                self._parts, self._parts_count = {}, 0
            super().log(logs, *args, **kwargs)  # pyright: ignore[reportUnknownMemberType]

        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            dataset: Any = train_dataset if train_dataset is not None else self.train_dataset
            if not self._seq_cls.group_by_length:
                return super()._get_train_sampler(train_dataset)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
            pt_utils: Any = __import__("transformers.trainer_pt_utils", fromlist=["_"])
            return pt_utils.LengthGroupedSampler(
                self.args.train_batch_size, dataset=dataset, lengths=list(dataset["length"])
            )

    return SeqClsModule, SeqClsTrainer


class SeqClsRunner:
    """Run one sequence-classification fine-tune from a local checkpoint directory.

    Args:
        config: The run's :class:`SeqClsConfig`.
        peft_config: Refused when set: the method trains full weights.
        memory_probe: How the preflight measures memory (default: the CUDA allocator).
    """

    def __init__(
        self,
        config: SeqClsConfig,
        *,
        peft_config: LoRAConfig | QLoRAConfig | None = None,
        memory_probe: MemoryProbe | None = None,
    ) -> None:
        if peft_config is not None:
            msg = "seq_cls trains full weights; adapter must be 'none'"
            raise ValueError(msg)
        self._config = config
        self._probe = memory_probe
        # Whether the base is a decoder (template rendering, last-token pooling), read from the
        # checkpoint by `load_tokenizer`.
        self._decoder: bool | None = None

    @property
    def config(self) -> SeqClsConfig:
        return self._config

    # ------------------------------------------------------------- preparation

    def load_tokenizer(self, checkpoint: Path) -> Any:
        _, transformers = _modules()
        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            str(checkpoint), **tokenizer_load_kwargs(False)
        )
        if not getattr(tokenizer, "is_fast", False):
            msg = "seq_cls needs a fast tokenizer (it locates premise markers by character offset)"
            raise PreflightError(msg, {})
        if tokenizer.pad_token_id is None:
            msg = "the tokenizer defines no padding token"
            raise PreflightError(msg, {})
        self._decoder = self._is_decoder(checkpoint)
        return tokenizer

    @property
    def decoder(self) -> bool:
        if self._decoder is None:
            msg = "load_tokenizer must run first: it reads the checkpoint's architecture"
            raise RuntimeError(msg)
        return self._decoder

    @property
    def rendering(self) -> Rendering:
        """This run's rendering, shared with the scorer through ``strata_verifier.json``."""
        cfg = self._config
        return Rendering(
            encoding="template" if self.decoder else "pair",
            premise_format=cfg.premise_format,
            decoder_template=cfg.decoder_template if self.decoder else None,
            max_length=cfg.max_length,
        )

    def _is_decoder(self, checkpoint: Path) -> bool:
        """The base's kind, from its own config; an ``existing`` decoder keeps its rendering.

        A run continuing a classifier must render and pool rows exactly as the run that trained
        it did, so the saved rendering must equal the declared one.
        """
        cfg = self._config
        try:
            config: Any = json.loads((checkpoint / "config.json").read_text())
        except (OSError, ValueError) as exc:
            msg = "the checkpoint has no readable config.json"
            raise PreflightError(msg, {}) from exc
        model_type = (
            str(cast("dict[str, Any]", config).get("model_type", ""))
            if isinstance(config, dict)
            else ""
        )
        if cfg.head_init == "decoder_lm_head":
            return True
        if cfg.head_init == "new":
            return False
        if model_type not in DECODER_LM_HEAD and model_type not in ENCODER_NEW_HEAD:
            msg = (
                f"head_init 'existing' has no declared architecture for model type "
                f"{model_type!r}; supported: "
                f"{', '.join(sorted({*DECODER_LM_HEAD, *ENCODER_NEW_HEAD}))}"
            )
            raise PreflightError(msg, {})
        decoder = model_type in DECODER_LM_HEAD
        saved = _read_verifier(checkpoint)
        rendering = saved.get("input") if saved else None
        if isinstance(rendering, dict):
            declared = {
                "premise_format": cfg.premise_format,
                "decoder_template": cfg.decoder_template if decoder else None,
            }
            stored = cast("dict[str, Any]", rendering)
            changed = sorted(k for k, v in declared.items() if stored.get(k) != v)
            if changed:
                msg = (
                    f"the existing classifier was trained with a different {', '.join(changed)}; "
                    "a continued run renders rows exactly as the first one did"
                )
                raise PreflightError(msg, {})
        return decoder

    def verbalizer_ids(self, tokenizer: Any) -> dict[str, int]:
        """Each label's verbalizer as its single token id; a multi-token or shared one refuses."""
        verbalizers = self._config.verbalizers or {}
        ids: dict[str, int] = {}
        for label in self._config.labels:
            tokens: list[int] = list(
                tokenizer(verbalizers[label], add_special_tokens=False)["input_ids"]
            )
            if len(tokens) != 1:
                msg = (
                    f"the verbalizer for {label!r} is {len(tokens)} tokens; it must be exactly one"
                )
                raise PreflightError(msg, {})
            ids[label] = tokens[0]
        if len(set(ids.values())) != len(ids):
            msg = "two labels share a verbalizer token"
            raise PreflightError(msg, {})
        return ids

    def encode(self, dataset: Any, tokenizer: Any, *, split: str) -> tuple[Any, dict[str, Any]]:
        """Tokenise every row and apply the overflow preflight. Returns the encoded split and its
        receipt (row count, longest row, the rendering's token overhead)."""
        roles = frozenset(dataset.column_names)
        missing = [r for r in _ROLES_PRIMARY if r not in roles]
        if missing:
            msg = f"{split}: the split has no {missing} column"
            raise PreflightError(msg, {})
        if "support_set" in roles and self._config.premise_format != "ids":
            msg = "the support head needs premise_format 'ids'"
            raise PreflightError(msg, {})
        rows = len(dataset)
        if rows == 0 or rows > MAX_SEQ_CLS_ROWS:
            msg = f"{split}: {rows} rows; a seq_cls split holds 1 to {MAX_SEQ_CLS_ROWS} rows"
            raise PreflightError(msg, {"rows": rows})
        encoder = _Encoder(self._config, tokenizer, roles, self.rendering)
        try:
            encoded: Any = dataset.map(
                encoder.encode,
                batched=True,
                with_indices=True,
                remove_columns=list(dataset.column_names),
                desc=f"Tokenising {split}",
            )
        except DatasetFormatError as exc:
            raise PreflightError(f"{split}: {exc}", {"rows": rows}) from exc
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            # A cell of an unexpected type; the library's own message can quote the cell, and
            # this one reaches a browser, so it names only the split.
            msg = f"{split}: a row could not be encoded (a cell has an unexpected type)"
            raise PreflightError(msg, {"rows": rows}) from exc
        lengths: list[int] = list(encoded["length"])
        over = [i for i, n in enumerate(lengths) if n > self._config.max_length]
        receipt: dict[str, Any] = {
            "rows": rows,
            "max_length": self._config.max_length,
            "longest": max(lengths),
            "rows_over_max_length": len(over),
            # Per-row token counts, special tokens and template literals included.
            "length_distribution": {
                "min": min(lengths),
                "p50": _nearest_rank(lengths, 0.50),
                "p95": _nearest_rank(lengths, 0.95),
                "max": max(lengths),
            },
            "template_token_overhead": self._overhead(tokenizer),
        }
        if over:
            msg = (
                f"{split}: {len(over)} row(s) exceed max_length {self._config.max_length} "
                f"(longest {max(lengths)} tokens; first rows {over[:_MAX_REPORTED_ROWS]}); "
                "rows are never truncated"
            )
            raise PreflightError(msg, receipt)
        if self.decoder:
            ends_on_pad = sum(
                1 for ids in encoded["input_ids"] if ids and ids[-1] == tokenizer.pad_token_id
            )
            if ends_on_pad:
                msg = (
                    f"{split}: {ends_on_pad} row(s) end on the padding token, so the answer "
                    "position cannot be found"
                )
                raise PreflightError(msg, receipt)
        return encoded, receipt

    def _overhead(self, tokenizer: Any) -> int:
        """Tokens the rendering adds around the fields: special tokens plus template literals."""
        if self.decoder:
            literal, _ = fill_template(self._config.decoder_template, "", "")
            literal_tokens = len(tokenizer(literal, add_special_tokens=False)["input_ids"])
            return int(tokenizer.num_special_tokens_to_add(pair=False)) + literal_tokens
        return int(tokenizer.num_special_tokens_to_add(pair=True))

    def load_model(
        self, checkpoint: Path, tokenizer: Any, *, roles: frozenset[str]
    ) -> tuple[Any, LoadReceipt, list[str]]:
        """Load the classifier under the load guard, initialise its heads, attach aux heads."""
        torch, transformers = _modules()
        cfg = self._config
        labels = list(cfg.labels)
        if cfg.head_init == "existing":
            self._check_existing_labels(checkpoint)
        model, info = transformers.AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint),
            num_labels=len(labels),
            id2label=dict(enumerate(labels)),
            label2id={label: i for i, label in enumerate(labels)},
            output_loading_info=True,
            dtype=torch.float32,
            **model_load_kwargs(False),
        )
        try:
            receipt = check_load(model, info, cfg.head_init)
            if cfg.from_lm_head:
                ids = self.verbalizer_ids(tokenizer)
                init_decoder_head(
                    model, checkpoint, receipt, verbalizer_ids=[ids[k] for k in labels]
                )
                receipt.verbalizer_token_ids = ids
        except HeadLoadError as exc:
            raise PreflightError(str(exc), {}) from exc
        model.config.pad_token_id = tokenizer.pad_token_id
        # Every head reads one pooled vector, the first token for an encoder; a config that mean
        # pools would give the primary head a different vector from the auxiliary heads.
        pooling: Any = getattr(model.config, "classifier_pooling", None)
        if pooling is not None and pooling != "cls":
            model.config.classifier_pooling = "cls"
            receipt.pooling_changed_from = str(pooling)
        frozen = self._frozen_parameters(checkpoint, receipt)
        params = dict(model.named_parameters())
        unknown = [name for name in frozen if name not in params]
        if unknown:
            msg = f"{VERIFIER_CONFIG_FILENAME} lists frozen parameters the model does not have: {unknown}"
            raise PreflightError(msg, {})
        for name in frozen:
            params[name].requires_grad_(False)
        aux = build_aux_heads(
            int(model.config.hidden_size),
            flag="inconsistent_premises" in roles,
            support="support_set" in roles,
            std=float(getattr(model.config, "initializer_range", 0.02)),
        )
        if cfg.head_init == "existing":
            try:
                load_aux_heads(aux, checkpoint)
            except HeadLoadError as exc:
                raise PreflightError(str(exc), {}) from exc
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False, "preserve_rng_state": True}
            )
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        module_cls, _ = _classes(torch, transformers)
        return module_cls(model, aux, last_token=self.decoder), receipt, frozen

    def _check_existing_labels(self, checkpoint: Path) -> None:
        try:
            config: Any = json.loads((checkpoint / "config.json").read_text())
        except (OSError, ValueError) as exc:
            msg = "the checkpoint has no readable config.json"
            raise PreflightError(msg, {}) from exc
        id2label: Any = (
            cast("dict[str, Any]", config).get("id2label") if isinstance(config, dict) else None
        )
        found = (
            [str(id2label[str(i)]) for i in range(len(id2label))]  # pyright: ignore[reportUnknownArgumentType]
            if isinstance(id2label, dict) and all(str(i) in id2label for i in range(len(id2label)))  # pyright: ignore[reportUnknownArgumentType]
            else None
        )
        if found != list(self._config.labels):
            msg = (
                f"the existing classifier's label order is {found}, the run declares "
                f"{list(self._config.labels)}"
            )
            raise PreflightError(msg, {})

    def _frozen_parameters(self, checkpoint: Path, receipt: LoadReceipt) -> list[str]:
        frozen = [p.split(" ")[0] for p in receipt.added_parameters if "frozen" in p]
        if self._config.head_init == "existing":
            listed: Any = (_read_verifier(checkpoint) or {}).get("frozen_parameters")
            if isinstance(listed, list):
                frozen += [str(p) for p in listed]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        return sorted(set(frozen))

    # ------------------------------------------------------------- checks

    def step0_check(
        self, module: Any, checkpoint: Path, rows: Any, receipt: LoadReceipt
    ) -> dict[str, Any]:
        """Compare the classifier with the base model's LM-head label logits on real rows."""
        torch, transformers = _modules()
        cfg = self._config
        ids = [receipt.verbalizer_token_ids[label] for label in cfg.labels]
        lm: Any = transformers.AutoModelForCausalLM.from_pretrained(
            str(checkpoint), dtype=torch.float32, **model_load_kwargs(False)
        )
        device: Any = next(module.parameters()).device
        lm.to(device)
        lm.eval()
        module.eval()
        logit_diff = 0.0
        prob_diff = 0.0
        argmax_agree = 0
        sample = min(cfg.step0_rows, len(rows))
        # Both sides in float32 with TF32 off, so the comparison is not of two rounding modes.
        with _exact_fp32(torch), torch.no_grad():
            for i in range(sample):
                input_ids: Any = torch.tensor([rows[i]["input_ids"]], device=device)
                mask: Any = torch.ones_like(input_ids)
                zero_shot: Any = lm(input_ids=input_ids, attention_mask=mask).logits[0, -1, ids]
                head: Any = module(input_ids=input_ids, attention_mask=mask)["logits"][0]
                logit_diff = max(logit_diff, float((zero_shot - head).abs().max()))
                prob_diff = max(
                    prob_diff,
                    float((zero_shot.softmax(-1) - head.softmax(-1)).abs().max()),
                )
                argmax_agree += int(int(zero_shot.argmax()) == int(head.argmax()))
        del lm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        module.train()
        result = {
            "rows": sample,
            "max_abs_logit_diff": logit_diff,
            "max_abs_prob_diff": prob_diff,
            "tolerance": cfg.step0_tolerance,
            "argmax_agree": argmax_agree,
            "precision": "float32, tf32 off",
            "passed": logit_diff <= cfg.step0_tolerance and argmax_agree == sample,
            "verbalizer_token_ids": dict(receipt.verbalizer_token_ids),
        }
        if not result["passed"]:
            msg = (
                f"step-0 check failed: the classifier's label logits differ from the base model's "
                f"zero-shot logits by {logit_diff:.3g} (tolerance {cfg.step0_tolerance:g}), and "
                f"the predicted label agrees on {argmax_agree} of {sample} rows"
            )
            raise Step0Error(msg, result)
        return result

    def memory_preflight(
        self, trainer: Any, encoded: Any, evaluation: Any = None
    ) -> dict[str, Any]:
        """One optimizer step at the longest real batch, measured and then undone exactly.

        With an eval split, the in-run evaluation is measured too: a forward pass at the eval
        batch size over the eval split's longest rows, with the optimizer state still allocated,
        as it is when the trainer evaluates mid-run. The budget applies to the larger peak.
        """
        torch, _ = _modules()
        cfg = self._config
        probe = self._probe or CudaMemoryProbe(torch)
        lengths: list[int] = list(encoded["length"])
        batch_rows = min(cfg.per_device_batch_size, len(lengths))
        longest = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)[:batch_rows]
        features = [encoded[i] for i in longest]
        model: Any = trainer.model
        device: Any = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in trainer.data_collator(features).items()}
        snapshot = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()}
        receipt: dict[str, Any] = {
            "batch_rows": batch_rows,
            "batch_max_length": max(lengths[i] for i in longest),
            "gradient_checkpointing": cfg.gradient_checkpointing,
            "precision": cfg.precision,
            "memory_fraction": cfg.memory_fraction,
            "device": str(device),
            "total_bytes": probe.total_bytes(),
        }
        autocast_device = "cuda" if device.type == "cuda" else "cpu"
        model.train()
        probe.reset()
        try:
            with torch.autocast(
                autocast_device, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"
            ):
                loss: Any = trainer.compute_loss(model, batch)
            loss.backward()
            optimizer: Any = trainer.create_optimizer()
            optimizer.step()
            receipt["peak_bytes"] = probe.peak_bytes()
            if evaluation is not None and len(evaluation) > 0:
                eval_lengths: list[int] = list(evaluation["length"])
                eval_rows = min(cfg.eval_batch_size or cfg.per_device_batch_size, len(eval_lengths))
                eval_longest = sorted(
                    range(len(eval_lengths)), key=lambda i: eval_lengths[i], reverse=True
                )[:eval_rows]
                eval_batch = {
                    k: v.to(device)
                    for k, v in trainer.data_collator([evaluation[i] for i in eval_longest]).items()
                }
                receipt["eval_batch_rows"] = eval_rows
                receipt["eval_batch_max_length"] = max(eval_lengths[i] for i in eval_longest)
                model.zero_grad(set_to_none=True)
                model.eval()
                probe.reset()
                with (
                    torch.no_grad(),
                    torch.autocast(
                        autocast_device, dtype=torch.bfloat16, enabled=cfg.precision == "bf16"
                    ),
                ):
                    trainer.compute_loss(model, eval_batch)
                receipt["eval_peak_bytes"] = probe.peak_bytes()
                model.train()
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            # cuBLAS / cuDNN allocation failures surface as a RuntimeError, not as the OOM type.
            if not isinstance(exc, torch.OutOfMemoryError) and not _is_allocation_failure(exc):
                raise
            receipt["peak_bytes"] = probe.peak_bytes()
            receipt["out_of_memory"] = True
            msg = f"memory preflight ran out of memory at the longest batch ({_gib(receipt)})"
            raise PreflightError(msg, receipt) from exc
        finally:
            trainer.optimizer = None
            trainer.reset_loss_parts()
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(snapshot[name])
            del snapshot
            if device.type == "cuda":
                torch.cuda.empty_cache()
        measured = [
            b for b in (receipt["peak_bytes"], receipt.get("eval_peak_bytes")) if b is not None
        ]
        peak = max(measured) if measured else None
        total = receipt["total_bytes"]
        receipt["measured_peak_bytes"] = peak
        receipt["budget_bytes"] = int(total * cfg.memory_fraction) if total else None
        if (
            peak is not None
            and receipt["budget_bytes"] is not None
            and peak > receipt["budget_bytes"]
        ):
            which = "eval" if peak == receipt.get("eval_peak_bytes") else "training"
            msg = (
                f"memory preflight {which} peak {_gib({**receipt, 'peak_bytes': peak})} exceeds "
                f"{cfg.memory_fraction:.0%} of the device"
            )
            raise PreflightError(msg, receipt)
        return receipt

    # ------------------------------------------------------------- the run

    def build_trainer(self, module: Any, train: Any, evaluation: Any, tokenizer: Any) -> Any:
        torch, transformers = _modules()
        cfg = self._config
        _, trainer_cls = _classes(torch, transformers)
        has_eval = evaluation is not None
        args: Any = transformers.TrainingArguments(
            output_dir=cfg.output_dir,
            per_device_train_batch_size=cfg.per_device_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size or cfg.per_device_batch_size,
            num_train_epochs=cfg.num_epochs,
            max_steps=cfg.max_steps or -1,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler,
            warmup_steps=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            max_grad_norm=cfg.max_grad_norm,
            logging_steps=cfg.logging_steps,
            eval_strategy=("steps" if cfg.eval_steps else "epoch") if has_eval else "no",
            eval_steps=cfg.eval_steps,
            save_strategy="steps" if cfg.save_steps > 0 else "no",
            save_steps=cfg.save_steps if cfg.save_steps > 0 else 500,
            save_total_limit=cfg.save_total_limit,
            seed=cfg.seed,
            bf16=cfg.precision == "bf16",
            gradient_accumulation_steps=1,
            gradient_checkpointing=False,  # enabled on the classifier itself (see load_model)
            report_to="none",
            remove_unused_columns=False,
            prediction_loss_only=True,
            label_names=["target"],
            dataloader_num_workers=0,
            use_cpu=not torch.cuda.is_available(),
        )
        trainer: Any = trainer_cls(
            model=module,
            args=args,
            train_dataset=train,
            eval_dataset=evaluation,
            data_collator=_collate(int(tokenizer.pad_token_id), torch),
            seq_cls=cfg,
            positive_index=cfg.positive_index,
        )
        _attach_progress(trainer, cfg.progress_jsonl)
        return trainer

    def save(
        self,
        module: Any,
        tokenizer: Any,
        artifact_dir: Path,
        *,
        public_model_id: str,
        receipt: LoadReceipt,
        frozen: list[str],
    ) -> dict[str, Any]:
        """Write the stock checkpoint, tokenizer, aux heads and rendering config."""
        cfg = self._config
        artifact_dir.mkdir(parents=True, exist_ok=True)
        classifier: Any = module.classifier
        _require_zero(classifier, _zero_parameters(receipt), when="at save")
        classifier.config._name_or_path = public_model_id
        classifier.save_pretrained(str(artifact_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(artifact_dir))
        aux_path = save_aux_heads(module.aux, artifact_dir)
        outputs = ["primary"]
        if module.aux.flag is not None:
            outputs.append("flag")
        if module.aux.support is not None:
            outputs.append("support")
        verifier: dict[str, Any] = {
            "schema": "strata.verifier/v1",
            "labels": list(cfg.labels),
            "label_aliases": dict(cfg.label_aliases),
            "label_index_map": cfg.label_index_map,
            "two_way_positive_label": cfg.two_way_positive_label,
            "two_way_labels": list(cfg.two_way_labels) if cfg.two_way_labels else None,
            "head_init": cfg.head_init,
            "model_type": receipt.model_type,
            "pooling": "last_token" if self.decoder else "first_token",
            "classifier_pooling": getattr(classifier.config, "classifier_pooling", None),
            "input": {
                **self.rendering.to_verifier(),
                "decoder_template_sha256": _sha256(cfg.decoder_template) if self.decoder else None,
                "overflow": "error",
            },
            "verbalizers": dict(cfg.verbalizers) if cfg.verbalizers else None,
            "verbalizer_token_ids": dict(receipt.verbalizer_token_ids) or None,
            "outputs": outputs,
            "aux_heads_file": AUX_HEADS_FILENAME if aux_path else None,
            "frozen_parameters": frozen,
        }
        (artifact_dir / VERIFIER_CONFIG_FILENAME).write_text(
            json.dumps(verifier, indent=2, sort_keys=True) + "\n"
        )
        return verifier

    def artifact_checks(
        self,
        prepared: PreparedRun,
        artifact_dir: Path,
        calibration_source: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reload the saved artifact the way a scorer does, check it, and calibrate it.

        The reload is stock ``AutoModelForSequenceClassification`` from the artifact's own
        safetensors, no remote code, float32 with TF32 off, under the same complete-load check a
        scorer applies (:func:`~strata_forge.training.loading.check_loading_info`). Its logits on
        a fixed set of rows (the first ``step0_rows`` of the eval split, else of the training
        split) must equal the trained classifier's in-memory float32 logits within
        ``step0_tolerance`` with the same predicted label on every row, and every parameter held
        at zero must still be exactly zero. With a calibration split, a temperature is then fitted
        on the reloaded model's logits over its three-way rows and written to
        :data:`CALIBRATION_FILENAME`, before anything else reads the artifact.
        """
        torch, transformers = _modules()
        cfg = self._config
        module: Any = prepared.module
        device: Any = next(module.parameters()).device
        model: Any
        try:
            model, info = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(artifact_dir),
                local_files_only=True,
                output_loading_info=True,
                dtype=torch.float32,
                **model_load_kwargs(False),
            )
            check_loading_info(cast("Mapping[str, Any]", info), head_prefixes=_HEAD_PREFIXES)
        except (LoadRefusedError, OSError, ValueError) as exc:
            msg = f"the saved classifier does not reload cleanly: {exc}"
            raise ArtifactCheckError(msg, {}) from exc
        zeros = _zero_parameters(prepared.load_receipt)
        _require_zero(model, zeros, when="after reload")
        model.config.pad_token_id = prepared.tokenizer.pad_token_id
        model.to(device)
        model.eval()
        module.eval()
        rows = prepared.parity_rows
        logit_diff = 0.0
        agree = 0
        with _exact_fp32(torch), torch.no_grad():
            for row in rows:
                ids: Any = torch.tensor([row["input_ids"]], device=device)
                mask: Any = torch.ones_like(ids)
                # The classifier itself, not the module the trainer wrapped: mixed-precision
                # training wraps that module's forward in autocast, and this compare is fp32.
                trained: Any = (
                    module.classifier(input_ids=ids, attention_mask=mask).logits[0].float()
                )
                reloaded: Any = model(input_ids=ids, attention_mask=mask).logits[0].float()
                logit_diff = max(logit_diff, float((trained - reloaded).abs().max()))
                agree += int(int(trained.argmax()) == int(reloaded.argmax()))
        parity: dict[str, Any] = {
            "rows": len(rows),
            "rows_from": prepared.parity_from,
            "max_abs_logit_diff": logit_diff,
            "tolerance": cfg.step0_tolerance,
            "argmax_agree": agree,
            "precision": "float32, tf32 off",
            "zero_parameters": zeros,
            "passed": logit_diff <= cfg.step0_tolerance and agree == len(rows),
        }
        if not parity["passed"]:
            msg = (
                f"the saved classifier's logits differ from the trained one's by {logit_diff:.3g} "
                f"(tolerance {cfg.step0_tolerance:g}), and the predicted label agrees on {agree} of "
                f"{len(rows)} rows"
            )
            raise ArtifactCheckError(msg, parity)
        out: dict[str, Any] = {
            "reload_parity": parity,
            "scoring_parity": self._scoring_parity(prepared, artifact_dir),
            "calibration": None,
        }
        if prepared.calibration is not None:
            out["calibration"] = self._calibrate(
                model, prepared, artifact_dir, calibration_source or {}
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def _scoring_parity(self, prepared: PreparedRun, artifact_dir: Path) -> dict[str, Any]:
        """Score the first training rows through the scorer and compare with the trainer.

        The scorer (``hf_classify`` with ``artifact: strata_verifier``) loads the saved artifact
        and renders the RAW rows from its ``strata_verifier.json``; the trainer's in-memory
        classifier reads the same rows as it encoded them. Both in float32 with TF32 off: the
        logits must agree within ``step0_tolerance`` with the same predicted label on every row,
        or the artifact is refused before it is pushed.
        """
        torch, _ = _modules()
        cfg = self._config
        # Imported here: the scoring pipeline imports this package's heads and rendering.
        scoring: Any = importlib.import_module("strata_forge.pipelines._scoring")
        raw = prepared.scoring_rows
        encoded = prepared.scoring_encoded
        module: Any = prepared.module
        device: Any = next(module.parameters()).device
        with _exact_fp32(torch):
            try:
                scorer: Any = scoring.load_verifier(
                    artifact_dir,
                    label_alias=None,
                    max_length=None,
                    dtype="float32",
                    apply_calibration=False,
                )
                rendered = scorer.render(
                    [row["premises"] for row in raw], [row["hypothesis"] for row in raw]
                )
                scored: list[Any] = scorer.score(rendered)
            except scoring.RunError as exc:
                msg = f"the saved classifier could not be scored as a verifier: {exc}"
                raise ArtifactCheckError(msg, {}) from exc
            trained: list[list[float]] = []
            module.eval()
            with torch.no_grad():
                for row in encoded:
                    ids: Any = torch.tensor([row["input_ids"]], device=device)
                    logits: Any = module.classifier(
                        input_ids=ids, attention_mask=torch.ones_like(ids)
                    ).logits[0]
                    trained.append(cast("list[float]", logits.float().cpu().tolist()))
        diff = 0.0
        agree = 0
        for mine, theirs in zip(trained, scored, strict=True):
            diff = max(diff, max(abs(a - b) for a, b in zip(mine, theirs.logits, strict=True)))
            agree += int(mine.index(max(mine)) == theirs.logits.index(max(theirs.logits)))
        receipt: dict[str, Any] = {
            "rows": len(raw),
            "rows_from": "train",
            "path": "hf_classify artifact strata_verifier",
            "max_abs_logit_diff": diff,
            "tolerance": cfg.step0_tolerance,
            "argmax_agree": agree,
            "precision": "float32, tf32 off",
            "passed": diff <= cfg.step0_tolerance and agree == len(raw),
        }
        del scorer
        if not receipt["passed"]:
            msg = (
                f"scored as a verifier, the saved classifier's logits differ from the trained "
                f"one's by {diff:.3g} (tolerance {cfg.step0_tolerance:g}), and the predicted label "
                f"agrees on {agree} of {len(raw)} rows"
            )
            raise ArtifactCheckError(msg, receipt)
        return receipt

    def _calibrate(
        self,
        model: Any,
        prepared: PreparedRun,
        artifact_dir: Path,
        source: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Fit one temperature on the calibration split's three-way rows; write the record."""
        torch, _ = _modules()
        cfg = self._config
        try:
            fitter: Any = importlib.import_module("strata_forge.metrics.calibration_fit")
        except ImportError as exc:
            msg = "this engine has no temperature fit (strata_forge.metrics.calibration_fit)"
            raise ArtifactCheckError(msg, {}) from exc
        encoded: Any = prepared.calibration
        device: Any = next(model.parameters()).device
        collate = prepared.trainer.data_collator
        labels = list(cfg.labels)
        picked = [
            i
            for i, (two_way, target) in enumerate(
                zip(encoded["two_way"], encoded["target"], strict=True)
            )
            if not two_way and max(target) == 1.0
        ]
        if not picked:
            msg = "the calibration split has no three-way row with a single label to fit on"
            raise ArtifactCheckError(msg, {"rows": len(encoded)})
        digest = hashlib.sha256()
        logits: list[list[float]] = []
        gold: list[str] = []
        size = cfg.eval_batch_size or cfg.per_device_batch_size
        with _exact_fp32(torch), torch.no_grad():
            for start in range(0, len(picked), size):
                features = [encoded[i] for i in picked[start : start + size]]
                batch = collate(features)
                out: Any = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                ).logits.float()
                logits.extend(cast("list[list[float]]", out.cpu().tolist()))
                for feature in features:
                    label = labels[list(feature["target"]).index(1.0)]
                    gold.append(label)
                    line = json.dumps([list(feature["input_ids"]), label], separators=(",", ":"))
                    digest.update(line.encode() + b"\n")
        fit: Any = fitter.fit_temperature(logits, gold, labels=labels)
        record: dict[str, Any] = {
            "schema": "strata.calibration/v1",
            # The ordered output labels the temperature applies to: an evaluator applies the
            # record from this file alone, without reading strata_verifier.json.
            "labels": labels,
            "fit": cast("dict[str, Any]", fit.model_dump(mode="json")),
            "split": dict(source),
            "rows": len(picked),
            "excluded_rows": len(encoded) - len(picked),
            # sha256 over one JSON line [input_ids, label] per fitted row, in split order.
            "rows_sha256": digest.hexdigest(),
            "logits_from": "the saved artifact, float32, tf32 off",
        }
        (artifact_dir / CALIBRATION_FILENAME).write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        return record

    def prepare(
        self,
        *,
        checkpoint: Path,
        train_dataset: Any,
        eval_dataset: Any = None,
        calibration_dataset: Any = None,
        on_receipt: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> PreparedRun:
        """Every check before step 0, in order; returns the trainer ready to start.

        ``calibration_dataset`` is the declared in-distribution dev split the temperature is
        fitted on after training; it is encoded and length-checked here like the others.

        ``on_receipt(name, receipt)`` is called as each check passes (``overflow``, ``load``,
        ``step0``, ``memory``), so a caller can report it before the run moves on. Blocking.
        """
        torch, transformers = _modules()
        cfg = self._config
        report = on_receipt or _ignore_receipt
        visible = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if visible > 1:
            msg = (
                f"seq_cls trains on one GPU and this machine exposes {visible}; the batch would "
                "be split across them and change the objective's denominators. Set "
                "CUDA_VISIBLE_DEVICES to one device"
            )
            raise PreflightError(msg, {"visible_gpus": visible})
        timing: dict[str, float] = {}
        receipts: dict[str, Any] = {}
        mark = time.monotonic()

        def _stamp(name: str) -> None:
            nonlocal mark
            now = time.monotonic()
            timing[name] = round(now - mark, 3)
            mark = now

        transformers.set_seed(cfg.seed)
        tokenizer = self.load_tokenizer(checkpoint)
        roles = frozenset(train_dataset.column_names)
        with _check_stage("overflow"):
            train, receipts["overflow"] = self.encode(train_dataset, tokenizer, split="train")
        evaluation = None
        if eval_dataset is not None:
            if frozenset(eval_dataset.column_names) != roles:
                msg = "the eval split must map the same roles as the training split"
                raise PreflightError(msg, {})
            with _check_stage("overflow"):
                evaluation, receipts["overflow_eval"] = self.encode(
                    eval_dataset, tokenizer, split="eval"
                )
        calibration = None
        if calibration_dataset is not None:
            with _check_stage("overflow"):
                calibration, receipts["overflow_calibration"] = self.encode(
                    calibration_dataset, tokenizer, split="calibration"
                )
        report("overflow", receipts["overflow"])
        _stamp("encode_s")

        with _check_stage("load"):
            module, load_receipt, frozen = self.load_model(checkpoint, tokenizer, roles=roles)
        receipts["load"] = {**asdict(load_receipt), "frozen_parameters": frozen}
        receipts["load"]["classifier_bias"] = (
            {"value": True, "reason": _CLASSIFIER_BIAS_REASON}
            if "classifier.bias" in load_receipt.added_parameters
            else None
        )
        receipts["loss"] = {
            "form": LOSS_FORM,
            "brier_weight": cfg.brier_weight,
            "flag_weight": cfg.flag_loss_weight,
            "support_weight": cfg.support_loss_weight,
        }
        report("load", receipts["load"])
        if visible:
            module.to("cuda")
        if cfg.from_lm_head:
            with _check_stage("step0"):
                receipts["step0"] = self.step0_check(module, checkpoint, train, load_receipt)
            report("step0", receipts["step0"])
        _stamp("load_s")

        trainer = self.build_trainer(module, train, evaluation, tokenizer)
        with _check_stage("memory"):
            receipts["memory"] = self.memory_preflight(trainer, train, evaluation)
        report("memory", receipts["memory"])
        # The preflight consumed random state; start the run from the declared seed regardless.
        transformers.set_seed(cfg.seed)
        _stamp("preflight_s")
        receipts["timing"] = timing
        parity_source = evaluation if evaluation is not None else train
        return PreparedRun(
            trainer=trainer,
            module=module,
            tokenizer=tokenizer,
            load_receipt=load_receipt,
            frozen=frozen,
            receipts=receipts,
            train_rows=len(train),
            eval_rows=len(evaluation) if evaluation is not None else 0,
            on_gpu=bool(visible),
            parity_rows=[parity_source[i] for i in range(min(cfg.step0_rows, len(parity_source)))],
            parity_from="eval" if evaluation is not None else "train",
            calibration=calibration,
            scoring_rows=[
                {
                    "premises": train_dataset[i]["premises"],
                    "hypothesis": train_dataset[i]["hypothesis"],
                }
                for i in range(min(cfg.step0_rows, len(train_dataset)))
            ],
            scoring_encoded=[train[i] for i in range(min(cfg.step0_rows, len(train)))],
        )

    def finish(
        self,
        prepared: PreparedRun,
        *,
        artifact_dir: Path,
        public_model_id: str | None = None,
        calibration_source: Mapping[str, Any] | None = None,
    ) -> SeqClsResult:
        """Train a prepared run, write its artifact, then check and calibrate it. Blocking.

        ``calibration_source`` names the calibration split (repo, config, split, commit) in the
        calibration record.
        """
        started = time.monotonic()
        output: Any = prepared.trainer.train()
        trained = time.monotonic()
        receipts = dict(prepared.receipts)
        receipts["verifier"] = self.save(
            prepared.module,
            prepared.tokenizer,
            artifact_dir,
            public_model_id=public_model_id or self._config.model_id,
            receipt=prepared.load_receipt,
            frozen=prepared.frozen,
        )
        saved = time.monotonic()
        with _check_stage("artifact"):
            receipts.update(self.artifact_checks(prepared, artifact_dir, calibration_source))
        timing = dict(receipts.get("timing") or {})
        timing["train_s"] = round(trained - started, 3)
        timing["save_s"] = round(saved - trained, 3)
        timing["artifact_checks_s"] = round(time.monotonic() - saved, 3)
        receipts["timing"] = timing
        return SeqClsResult(
            output_dir=str(artifact_dir),
            metrics=numeric_metrics(output),
            steps=coerce_int(getattr(output, "global_step", None)),
            receipts=receipts,
        )

    def run(
        self,
        *,
        checkpoint: Path,
        artifact_dir: Path,
        train_dataset: Any,
        eval_dataset: Any = None,
        calibration_dataset: Any = None,
        calibration_source: Mapping[str, Any] | None = None,
        public_model_id: str | None = None,
        on_receipt: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> SeqClsResult:
        """:meth:`prepare` then :meth:`finish`. Blocking; call it in a thread."""
        prepared = self.prepare(
            checkpoint=checkpoint,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            calibration_dataset=calibration_dataset,
            on_receipt=on_receipt,
        )
        return self.finish(
            prepared,
            artifact_dir=artifact_dir,
            public_model_id=public_model_id,
            calibration_source=calibration_source,
        )


@dataclass
class PreparedRun:
    """A run that passed every check and has a trainer ready to start."""

    trainer: Any
    module: Any
    tokenizer: Any
    load_receipt: LoadReceipt
    frozen: list[str]
    receipts: dict[str, Any]
    train_rows: int
    eval_rows: int
    on_gpu: bool
    parity_rows: list[dict[str, Any]]
    parity_from: str
    calibration: Any = None
    # The first training rows, raw and as the trainer encoded them, for the scoring-path check.
    scoring_rows: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    scoring_encoded: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


@contextlib.contextmanager
def _exact_fp32(torch: Any) -> Generator[None]:
    """TF32 off for the block (matmul and cuDNN), restored after; a check compares exact fp32."""
    backends: Any = getattr(torch, "backends", None)
    switches = [
        getattr(getattr(backends, "cuda", None), "matmul", None),
        getattr(backends, "cudnn", None),
    ]
    saved = [getattr(x, "allow_tf32", None) for x in switches]
    for switch in switches:
        if switch is not None and hasattr(switch, "allow_tf32"):
            switch.allow_tf32 = False
    try:
        yield
    finally:
        for switch, value in zip(switches, saved, strict=True):
            if switch is not None and isinstance(value, bool):
                switch.allow_tf32 = value


def _zero_parameters(receipt: LoadReceipt) -> list[str]:
    """The parameters this method added as exactly zero (and frozen there)."""
    return [p.split(" ")[0] for p in receipt.added_parameters if "zero" in p]


def _require_zero(model: Any, names: list[str], *, when: str) -> None:
    params = dict(model.named_parameters())
    for name in names:
        param: Any = params.get(name)
        if param is None or bool((param.detach() != 0).any()):
            msg = f"{name} must be exactly zero {when}, and is not"
            raise ArtifactCheckError(msg, {"parameter": name})


def _nearest_rank(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def _is_allocation_failure(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "alloc_failed" in text


def _read_verifier(checkpoint: Path) -> dict[str, Any] | None:
    """The saved rendering config of a checkpoint this method wrote, or ``None``."""
    try:
        saved: Any = json.loads((checkpoint / VERIFIER_CONFIG_FILENAME).read_text())
    except OSError, ValueError:
        return None
    return cast("dict[str, Any]", saved) if isinstance(saved, dict) else None


@contextlib.contextmanager
def _check_stage(name: str) -> Generator[None]:
    """Tag a refusal raised inside the block with the check it came from."""
    try:
        yield
    except PreflightError as exc:
        if not exc.check:
            exc.check = name
        raise


def _ignore_receipt(_name: str, _receipt: dict[str, Any]) -> None:
    return None


def _gib(receipt: Mapping[str, Any]) -> str:
    peak = receipt.get("peak_bytes")
    total = receipt.get("total_bytes")
    peak_s = f"{peak / 2**30:.2f} GiB" if isinstance(peak, int) else "unmeasured"
    total_s = f" of {total / 2**30:.2f} GiB" if isinstance(total, int) else ""
    flag = "on" if receipt.get("gradient_checkpointing") else "off"
    return f"peak {peak_s}{total_s}, activation checkpointing {flag}"
