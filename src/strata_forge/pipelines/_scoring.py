"""Exact sequence-classification scoring for the batch-inference runner (``engine="hf_classify"``).

A scoring run reads a classifier's own head instead of generating text. Each dataset row renders
into a ``(text, text_pair)`` pair — the spec's ``template`` and ``hyperparams.pair_template``, by
the same non-executing ``{name}`` substitution as a chat prompt, with a list-valued column (a
row's premises) joined by ``hyperparams.list_separator`` — and one forward pass of an
``AutoModelForSequenceClassification`` gives a logit per class. The runner persists the RAW
logits (``logit_<label>``), their plain softmax over the included classes (``p_<label>``,
temperature 1: no calibration is applied here) and the argmax label (``pred``).

The rules that make the numbers trustworthy:

- **Labels are declared, by class id.** ``config.id2label`` is read in class-id order, and the
  run's REQUIRED ``label_alias`` must name every one of its classes and nothing else. Each class
  maps to an output label, or to ``null``, which EXCLUDES it: an excluded class gets no column,
  and the softmax and argmax run over the included classes only. Two classes mapping to one label
  are refused; no combine rule is supported. At least two classes stay included.
- **Precision is declared.** ``dtype`` is required (``float32`` or ``bfloat16``). In float32 the
  TF32 matmul and cuDNN paths are switched off, so a GPU computes in true float32.
- **Nothing is truncated.** A pair longer than the model's limit (special tokens included) is
  dropped before the forward pass, counted, and written as a row with ``overflow`` true and no
  score: truncating the premises would score a different question than the one asked.
- **The load is complete.** Weights are safetensors from the run's own snapshot, no remote code,
  and :func:`~strata_forge.training.loading.check_loading_info` refuses a load that left a weight
  at its random initial value.

Heavy dependencies (torch, transformers) are imported inside :func:`load_classifier`. Everything
else here is pure, so it is tested without them.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from strata_forge.pipelines._common import MANIFEST_SCHEMA, RunError, manifest_engine
from strata_forge.training.loading import (
    LoadRefusedError,
    check_loading_info,
    model_load_kwargs,
    tokenizer_load_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from strata_forge.pipelines.inference_runner import RunSpec

__all__ = [
    "BATCH_ORDER",
    "LABEL_RE",
    "SCORING_DTYPES",
    "Classifier",
    "ScoringInputs",
    "batch_plan_record",
    "build_inputs",
    "build_manifest",
    "load_classifier",
    "plan_batches",
    "render_scoring_text",
    "resolve_labels",
    "result_rows",
    "softmax",
    "tally",
]

#: What an output label may be: it becomes part of a column name (``logit_<label>``).
LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
#: The dtypes a scoring run may declare. There is no default: the spec must name one.
SCORING_DTYPES = ("float32", "bfloat16")
#: How :func:`plan_batches` orders rows, as recorded in the manifest.
BATCH_ORDER = "token_length_desc_then_row_index"
# The same placeholder grammar as the chat prompt's `render_template`: a bare `{name}`, nothing else.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
# A rendered side longer than this is over any model's limit without tokenizing it.
_MAX_RENDERED_CHARS = 200_000
# Checkpoint tensors that belong to a classification head across the architectures transformers
# ships (`classifier` for BERT-family and DeBERTa, `score` for decoder heads, `head` for
# ModernBERT's prediction head).
_HEAD_PREFIXES = ("classifier.", "score.", "head.")
# transformers reports "no limit" as a huge sentinel `model_max_length`.
_NO_TOKENIZER_LIMIT = 1_000_000
# Architectures whose learned absolute positions start after the padding index (fairseq's
# convention): position ids run from `pad_token_id + 1`, so `pad_token_id + 1` of the
# `max_position_embeddings` slots are never usable by a token.
_PADDING_OFFSET_POSITIONS = frozenset(
    {"roberta", "xlm-roberta", "camembert", "data2vec-text", "roberta-prelayernorm", "xmod"}
)
_MAX_SLICES = 256
_MAX_SLICE_CHARS = 64
# The slice key a missing value is counted under; a real value spelled the same is refused.
_NULL_SLICE = "(null)"
_MAX_ITEM_ID_CHARS = 256
# Rows tokenized per call when measuring lengths: the fast tokenizer batches internally.
_MEASURE_CHUNK = 512
#: The installed distributions whose versions decide what a scoring run computed.
RECORDED_DISTRIBUTIONS = (
    "transformers",
    "torch",
    "tokenizers",
    "safetensors",
    "datasets",
    "huggingface-hub",
)


def render_scoring_text(
    template: str, row: Mapping[str, Any], column_mapping: Mapping[str, str], *, separator: str
) -> str:
    """Render ``template`` like a chat prompt, joining a list-valued column with ``separator``.

    No length cap here: scoring never truncates, so an over-long rendering is dropped and
    counted by the caller rather than cut.
    """

    def _sub(match: re.Match[str]) -> str:
        column = column_mapping.get(match.group(1))
        if column is None or column not in row:
            return match.group(0)
        value: object = row[column]
        if isinstance(value, list | tuple):
            return separator.join(str(item) for item in cast("Sequence[object]", value))
        return str(value)

    return _PLACEHOLDER_RE.sub(_sub, template)


def resolve_labels(
    id2label: Mapping[Any, Any], label_alias: Mapping[str, str | None]
) -> tuple[list[str], list[str | None]]:
    """The model's class names and each class's output label (``None`` = excluded), by class id.

    Refuses an ``id2label`` that is not a contiguous ``0..n-1`` map of at least two distinct
    names; an alias that does not name exactly the model's classes; an output label that is not a
    :data:`LABEL_RE` token; two classes mapped to one label (many-to-one has no combine rule); and
    an alias that leaves fewer than two classes included.
    """
    by_id: dict[int, str] = {}
    for key, value in id2label.items():
        index = key if isinstance(key, int) and not isinstance(key, bool) else None
        if isinstance(key, str) and key.isdigit():
            index = int(key)
        if index is None or not isinstance(value, str) or index in by_id:
            msg = "the model's id2label must map each integer class id to one name"
            raise RunError(msg)
        by_id[index] = value
    if len(by_id) < 2 or set(by_id) != set(range(len(by_id))):
        msg = "the model's id2label must name classes 0..n-1, at least two of them"
        raise RunError(msg)
    names = [by_id[index] for index in range(len(by_id))]
    if len(set(names)) != len(names):
        msg = f"the model's id2label repeats a class name: {names}"
        raise RunError(msg)
    unknown = sorted(set(label_alias) - set(names))
    unaliased = [name for name in names if name not in label_alias]
    if unknown or unaliased:
        msg = (
            f"label_alias must name exactly the model's classes {names} (a label, or null to "
            f"exclude one); not in the model: {unknown or 'none'}; missing: {unaliased or 'none'}"
        )
        raise RunError(msg)
    outputs = [label_alias[name] for name in names]
    included = [label for label in outputs if label is not None]
    bad = [label for label in included if not LABEL_RE.fullmatch(label)]
    if bad:
        msg = f"labels {bad} cannot name a column: a label is 1-64 letters, digits, '_', '.' or '-'"
        raise RunError(msg)
    if len(set(included)) != len(included):
        msg = (
            f"label_alias maps two classes to one label ({outputs}); combining classes is not "
            "supported, so give each its own label or exclude one with null"
        )
        raise RunError(msg)
    if len(included) < 2:
        msg = "label_alias must leave at least two classes included (not null)"
        raise RunError(msg)
    return names, outputs


@dataclass(frozen=True, slots=True)
class ScoringInputs:
    """Every row's rendered pair, plus its stable id and slice when the spec names them."""

    texts: list[str]
    pairs: list[str]
    item_ids: list[str] | None
    slices: list[str] | None


def build_inputs(spec: RunSpec, rows: Sequence[Mapping[str, Any]]) -> ScoringInputs:
    """Render every row and read its id and slice, refusing what would make them ambiguous."""
    hp = spec.hyperparams
    pair_template = hp.pair_template or ""
    columns = set(rows[0]) if rows else set[str]()
    for what, column in (("id_column", hp.id_column), ("slice_column", hp.slice_column)):
        if column is not None and rows and column not in columns:
            msg = f"{what} {column!r} is not a column of the split; columns: {sorted(columns)}"
            raise RunError(msg)
    texts = [
        render_scoring_text(spec.template, row, spec.column_mapping, separator=hp.list_separator)
        for row in rows
    ]
    pairs = [
        render_scoring_text(pair_template, row, spec.column_mapping, separator=hp.list_separator)
        for row in rows
    ]
    item_ids = _item_ids(rows, hp.id_column) if hp.id_column is not None else None
    slices = _slices(rows, hp.slice_column) if hp.slice_column is not None else None
    return ScoringInputs(texts=texts, pairs=pairs, item_ids=item_ids, slices=slices)


def _item_ids(rows: Sequence[Mapping[str, Any]], column: str) -> list[str]:
    """Each row's id as a string: present, a string or an integer, short, and unique.

    A float is refused rather than stringified: ``1.0`` and ``1`` would be two spellings of what
    a reader joins on as one id.
    """
    ids: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        value: object = row.get(column)
        if isinstance(value, bool) or not isinstance(value, str | int):
            msg = f"id_column {column!r} must hold a string or an integer on every row; row {index} does not"
            raise RunError(msg)
        text = str(value)
        if not text or len(text) > _MAX_ITEM_ID_CHARS:
            msg = f"id_column {column!r}: row {index} has an empty or over-long id"
            raise RunError(msg)
        if text in seen:
            msg = f"id_column {column!r} repeats the id {text[:64]!r}; ids must be unique"
            raise RunError(msg)
        seen.add(text)
        ids.append(text)
    return ids


def _slices(rows: Sequence[Mapping[str, Any]], column: str) -> list[str]:
    """Each row's slice key, refusing what would silently merge two slices into one count.

    A missing value counts under ``(null)``, so a real value spelled ``(null)`` is refused, as is
    one over 64 characters (cutting it could merge two values) and two values of different types
    with one spelling (``1`` and ``"1"``).
    """
    values: list[str] = []
    types: dict[str, str] = {}
    for index, row in enumerate(rows):
        raw: object = row.get(column)
        if raw is None:
            values.append(_NULL_SLICE)
            continue
        text = str(raw)
        if text == _NULL_SLICE:
            msg = f"slice_column {column!r}: row {index} holds {_NULL_SLICE!r}, the key for a missing value"
            raise RunError(msg)
        if len(text) > _MAX_SLICE_CHARS:
            msg = f"slice_column {column!r}: row {index} has a value over {_MAX_SLICE_CHARS} characters"
            raise RunError(msg)
        kind = type(raw).__name__
        if types.setdefault(text, kind) != kind:
            msg = (
                f"slice_column {column!r} holds {text!r} as both {types[text]} and {kind}; "
                "they would be counted as one slice"
            )
            raise RunError(msg)
        values.append(text)
    distinct = len(set(values))
    if distinct > _MAX_SLICES:
        msg = (
            f"slice_column {column!r} has {distinct} distinct values; counts are kept per slice, "
            f"so at most {_MAX_SLICES} are allowed"
        )
        raise RunError(msg)
    return values


def plan_batches(
    lengths: Sequence[int | None], *, max_length: int, batch_size: int
) -> tuple[list[list[int]], list[int]]:
    """Split rows into length-sorted batches and the overflow rows that are never scored.

    A ``None`` length is a rendering over the character cap. Batches run longest first, so a
    batch that does not fit in memory fails at the first step rather than near the end; ties keep
    row order, so the plan is deterministic.
    """
    overflow = [i for i, n in enumerate(lengths) if n is None or n > max_length]
    kept = sorted(
        ((n, i) for i, n in enumerate(lengths) if n is not None and n <= max_length),
        key=lambda item: (-item[0], item[1]),
    )
    batches = [
        [index for _, index in kept[offset : offset + batch_size]]
        for offset in range(0, len(kept), batch_size)
    ]
    return batches, overflow


def softmax(values: Sequence[float]) -> list[float]:
    """The plain (temperature 1) softmax of ``values``, computed stably in double precision."""
    peak = max(values)
    weights = [math.exp(value - peak) for value in values]
    total = math.fsum(weights)
    return [weight / total for weight in weights]


def result_rows(
    lengths: Sequence[int | None],
    logits: Mapping[int, Sequence[float]],
    outputs: Sequence[str | None],
    inputs: ScoringInputs,
) -> list[dict[str, Any]]:
    """One results row per input row, in input order; an unscored row carries no score.

    ``outputs`` is each class's label by class id, ``None`` for an excluded class: its logit is
    dropped before the softmax and the argmax, so both run over the included classes only. Every
    row has every column, so a refused first row cannot drop a column from the file.
    """
    kept = [index for index, label in enumerate(outputs) if label is not None]
    labels = [cast("str", outputs[index]) for index in kept]
    out: list[dict[str, Any]] = []
    for index, length in enumerate(lengths):
        full = logits.get(index)
        values = None if full is None else [full[k] for k in kept]
        row: dict[str, Any] = {"custom_id": f"row-{index}"}
        if inputs.item_ids is not None:
            row["item_id"] = inputs.item_ids[index]
        if inputs.slices is not None:
            row["slice"] = inputs.slices[index]
        row["input_tokens"] = length
        row["overflow"] = values is None
        if values is None:
            row["pred"] = None
            row.update({f"logit_{label}": None for label in labels})
            row.update({f"p_{label}": None for label in labels})
        else:
            # The first maximum on a tie: the lowest class id, deterministically.
            row["pred"] = labels[max(range(len(values)), key=values.__getitem__)]
            row.update(
                {f"logit_{label}": value for label, value in zip(labels, values, strict=True)}
            )
            probabilities = softmax(values)
            row.update(
                {f"p_{label}": value for label, value in zip(labels, probabilities, strict=True)}
            )
        out.append(row)
    return out


def tally(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Row, scored and overflow counts overall and, when rows carry a slice, per slice."""
    counts: dict[str, Any] = {
        "rows": len(rows),
        "scored": sum(1 for row in rows if not row["overflow"]),
        "overflow": sum(1 for row in rows if row["overflow"]),
    }
    if rows and "slice" in rows[0]:
        by_slice: dict[str, dict[str, int]] = {}
        for row in rows:
            entry = by_slice.setdefault(str(row["slice"]), {"rows": 0, "scored": 0, "overflow": 0})
            entry["rows"] += 1
            entry["overflow" if row["overflow"] else "scored"] += 1
        counts["by_slice"] = dict(sorted(by_slice.items()))
    return counts


class Classifier:
    """A loaded sequence classifier: measures pairs and scores batches, synchronously.

    The runner calls both from a worker thread, one batch at a time, so the event loop stays free
    to report progress between batches.
    """

    def __init__(
        self,
        *,
        model: Any,  # a transformers PreTrainedModel; untyped third-party object
        tokenizer: Any,  # a transformers tokenizer; untyped third-party object
        torch: Any,  # the torch module, imported lazily
        names: list[str],
        outputs: list[str | None],
        max_length: int,
        max_length_source: str,
        dtype: str,
        tf32: dict[str, bool | None],
        device: str,
        pad_token: str,
        unexpected_keys: list[str],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self.names = names
        self.outputs = outputs
        self.max_length = max_length
        self.max_length_source = max_length_source
        self.dtype = dtype
        self.tf32 = tf32
        self.device = device
        self.pad_token = pad_token
        self.unexpected_keys = unexpected_keys

    @property
    def labels(self) -> list[str]:
        """The included output labels, in class-id order."""
        return [label for label in self.outputs if label is not None]

    def lengths(self, texts: Sequence[str], pairs: Sequence[str]) -> list[int | None]:
        """Each pair's full token count with special tokens, untruncated; ``None`` past the cap."""
        out: list[int | None] = []
        for offset in range(0, len(texts), _MEASURE_CHUNK):
            chunk_texts = list(texts[offset : offset + _MEASURE_CHUNK])
            chunk_pairs = list(pairs[offset : offset + _MEASURE_CHUNK])
            measurable = [
                i
                for i, (text, pair) in enumerate(zip(chunk_texts, chunk_pairs, strict=True))
                if len(text) <= _MAX_RENDERED_CHARS and len(pair) <= _MAX_RENDERED_CHARS
            ]
            chunk: list[int | None] = [None] * len(chunk_texts)
            if measurable:
                encoded: Any = self._tokenizer(
                    [chunk_texts[i] for i in measurable],
                    [chunk_pairs[i] for i in measurable],
                    truncation=False,
                    add_special_tokens=True,
                )
                for i, ids in zip(measurable, encoded["input_ids"], strict=True):
                    chunk[i] = len(cast("list[int]", ids))
            out.extend(chunk)
        return out

    def logits(self, texts: Sequence[str], pairs: Sequence[str]) -> list[list[float]]:
        """Raw head logits for one batch of pairs (every class), right-padded, never truncated."""
        torch = self._torch
        features: Any = self._tokenizer(
            list(texts),
            list(pairs),
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        if int(features["input_ids"].shape[1]) > self.max_length:
            msg = "a batch exceeded the model's length limit after planning; nothing was scored"
            raise RunError(msg)
        with torch.inference_mode():
            output: Any = self._model(
                **{key: value.to(self.device) for key, value in features.items()}
            )
        raw: Any = output.logits.float().cpu().tolist()
        values = cast("list[list[float]]", raw)
        if len(values) != len(texts) or any(len(row) != len(self.names) for row in values):
            msg = f"the model returned logits of an unexpected shape for {len(self.names)} classes"
            raise RunError(msg)
        if any(not math.isfinite(value) for row in values for value in row):
            msg = "the model returned a non-finite logit"
            raise RunError(msg)
        return [[float(value) for value in row] for row in values]


def load_classifier(
    snapshot: Path,
    *,
    label_alias: Mapping[str, str | None],
    max_length: int | None,
    dtype: str,
) -> Classifier:
    """Load the snapshot's tokenizer and sequence-classification model, refusing a partial load.

    Reads only the local snapshot the runner downloaded (``local_files_only``), so no credential
    is involved and nothing else is fetched. In ``float32`` the TF32 matmul and cuDNN paths are
    switched off first, so a GPU does not silently compute at reduced precision.
    """
    if dtype not in SCORING_DTYPES:
        msg = f"dtype must be one of {', '.join(SCORING_DTYPES)}"
        raise RunError(msg)
    try:
        torch: Any = import_module("torch")
        transformers: Any = import_module("transformers")
    except ImportError as exc:
        msg = "classifier scoring needs torch and transformers: pip install 'strata-forge[serving]'"
        raise RunError(msg) from exc
    tf32 = _tf32_state(torch, disable=dtype == "float32")
    location = str(snapshot)
    try:
        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            location, local_files_only=True, **tokenizer_load_kwargs(False)
        )
    except (OSError, ValueError) as exc:
        msg = f"could not load the model's tokenizer: {exc}"
        raise RunError(msg) from exc
    try:
        model, info = transformers.AutoModelForSequenceClassification.from_pretrained(
            location,
            local_files_only=True,
            output_loading_info=True,
            dtype=getattr(torch, dtype),
            **model_load_kwargs(False),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        msg = f"could not load the model as a sequence classifier: {exc}"
        raise RunError(msg) from exc
    try:
        unexpected = check_loading_info(
            cast("Mapping[str, Any]", info), head_prefixes=_HEAD_PREFIXES
        )
    except LoadRefusedError as exc:
        raise RunError(str(exc)) from exc
    config: Any = model.config
    names, outputs = resolve_labels(cast("Mapping[Any, Any]", config.id2label), label_alias)
    # Before padding is aligned: the position offset is the CHECKPOINT's padding index.
    limit, source = _length_limit(config, tokenizer, max_length)
    pad_token = _align_padding(tokenizer, config)
    device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    model.to(device)
    model.eval()
    return Classifier(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        names=names,
        outputs=outputs,
        max_length=limit,
        max_length_source=source,
        dtype=str(model.dtype).removeprefix("torch."),
        tf32=tf32,
        device=device,
        pad_token=pad_token,
        unexpected_keys=unexpected,
    )


def _tf32_state(torch: Any, *, disable: bool) -> dict[str, bool | None]:
    """Switch TF32 off when ``disable``, and report both switches as they now stand."""
    backends: Any = getattr(torch, "backends", None)
    matmul: Any = getattr(getattr(backends, "cuda", None), "matmul", None)
    cudnn: Any = getattr(backends, "cudnn", None)
    if disable:
        if matmul is not None:
            matmul.allow_tf32 = False
        if cudnn is not None:
            cudnn.allow_tf32 = False
    state: dict[str, bool | None] = {}
    for name, switch in (("matmul", matmul), ("cudnn", cudnn)):
        value: object = getattr(switch, "allow_tf32", None)
        state[name] = value if isinstance(value, bool) else None
    return state


def _align_padding(tokenizer: Any, config: Any) -> str:
    """Right-pad, and make the model's pad id the tokenizer's: decoder heads pool by it.

    A decoder classification head reads the last token that is not ``config.pad_token_id``, so
    the two must agree or it reads a padding position. A tokenizer with no pad token pads with
    its end-of-sequence token, which is recorded.
    """
    tokenizer.padding_side = "right"
    source = "tokenizer"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            msg = "the tokenizer has neither a pad nor an end-of-sequence token to pad batches with"
            raise RunError(msg)
        tokenizer.pad_token = tokenizer.eos_token
        source = "eos"
    config.pad_token_id = tokenizer.pad_token_id
    return source


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _length_limit(config: Any, tokenizer: Any, requested: int | None) -> tuple[int, str]:
    """The token limit a pair must fit: the model's own, or a smaller requested one.

    The model's own is the smallest of its position table (less the slots a padding-offset
    architecture such as RoBERTa never gives a token: ``pad_token_id + 1``) and the tokenizer's
    declared ``model_max_length``.
    """
    limits: list[int] = []
    positions = _positive_int(getattr(config, "max_position_embeddings", None))
    if positions is not None:
        if getattr(config, "model_type", None) in _PADDING_OFFSET_POSITIONS:
            pad: object = getattr(config, "pad_token_id", None)
            offset = (
                pad + 1 if isinstance(pad, int) and not isinstance(pad, bool) and pad >= 0 else 2
            )
            positions -= offset
        if positions > 0:
            limits.append(positions)
    declared = _positive_int(getattr(tokenizer, "model_max_length", None))
    if declared is not None and declared < _NO_TOKENIZER_LIMIT:
        limits.append(declared)
    model_limit = min(limits) if limits else None
    if requested is None:
        if model_limit is None:
            msg = "the model declares no length limit; set hyperparams.max_model_len"
            raise RunError(msg)
        return model_limit, "model"
    if model_limit is not None and requested > model_limit:
        msg = f"max_model_len {requested} exceeds the model's own limit of {model_limit} tokens"
        raise RunError(msg)
    return requested, "requested"


def batch_plan_record(batches: Sequence[Sequence[int]], batch_size: int) -> dict[str, Any]:
    """How the scored rows were batched: the ordering rule, the size, and a digest of the plan.

    In reduced precision a row's logits can depend on which rows share its batch (padding
    changes the kernels' reduction shapes), so the plan is part of what a run computed. The full
    plan is reproducible from the rule and the dataset; the digest lets two runs prove they match.
    """
    return {
        "order": BATCH_ORDER,
        "batch_size": batch_size,
        "batches": len(batches),
        "sha256": _digest(json.dumps([list(batch) for batch in batches], separators=(",", ":"))),
    }


def build_manifest(
    spec: RunSpec,
    classifier: Classifier,
    rows: Sequence[Mapping[str, Any]],
    *,
    model_commit: str | None,
    dataset_commit: str | None,
    batches: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """The run's manifest (``kind`` ``scoring``): pins, the scoring setup and counts, no row text.

    Templates are recorded as digests (the pair template is left out of the hyperparameters for
    the same reason): the control plane stores their text with the spec, and a pushed manifest
    should not republish it beside the scores.
    """
    hp = spec.hyperparams
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": "scoring",
        "engine": manifest_engine(spec.engine_version, RECORDED_DISTRIBUTIONS),
        "model": {
            "id": spec.model_id,
            "requested_revision": spec.model_revision,
            "commit": model_commit,
        },
        "dataset": {
            "id": spec.dataset_id,
            "config": spec.dataset_config,
            "split": spec.split,
            "requested_revision": spec.dataset_commit_sha,
            "commit": dataset_commit,
        },
        "scoring": {
            "engine": hp.engine,
            "head": "sequence_classification",
            "id2label": {str(index): name for index, name in enumerate(classifier.names)},
            "label_alias": hp.label_alias,
            "labels": classifier.labels,
            "excluded": [
                name
                for name, label in zip(classifier.names, classifier.outputs, strict=True)
                if label is None
            ],
            "outputs": ["logit", "p_softmax_t1_over_included", "pred_argmax_over_included"],
            "calibration": None,
            "max_length": classifier.max_length,
            "max_length_source": classifier.max_length_source,
            "overflow_policy": "drop_and_count",
            "truncation": None,
            "padding_side": "right",
            "pad_token": classifier.pad_token,
            "dtype": classifier.dtype,
            "tf32": classifier.tf32,
            "device": classifier.device,
            "batch_plan": batch_plan_record(batches, hp.scoring_batch_size),
            "list_separator": hp.list_separator,
            "id_column": hp.id_column,
            "slice_column": hp.slice_column,
            "unexpected_checkpoint_keys": classifier.unexpected_keys,
        },
        "template_sha256": _digest(spec.template),
        "pair_template_sha256": _digest(hp.pair_template or ""),
        "column_mapping": spec.column_mapping,
        "hyperparams": hp.model_dump(exclude={"pair_template"}),
        "counts": tally(rows),
    }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
