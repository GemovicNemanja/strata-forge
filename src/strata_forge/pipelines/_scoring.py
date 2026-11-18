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

from strata_forge.pipelines._chat_template import render_kwargs
from strata_forge.pipelines._common import MANIFEST_SCHEMA, RunError, manifest_engine
from strata_forge.training.classifier_heads import (
    HeadLoadError,
    aux_outputs,
    build_aux_heads,
    load_aux_heads,
)
from strata_forge.training.dataset_format import DatasetFormatError
from strata_forge.training.loading import (
    LoadRefusedError,
    check_loading_info,
    model_load_kwargs,
    tokenizer_load_kwargs,
)
from strata_forge.training.verifier_render import (
    RenderedRow,
    Rendering,
    encode_rows,
    marker_tokens,
    render_row,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from pydantic import JsonValue

    from strata_forge.pipelines.inference_runner import RunSpec

__all__ = [
    "ARTIFACT_PAIR_TEMPLATE",
    "ARTIFACT_TEMPLATE",
    "BATCH_ORDER",
    "LABEL_RE",
    "SCORING_DTYPES",
    "VERIFIER_ARTIFACT",
    "Classifier",
    "ClassifierScorer",
    "ScoringInputs",
    "VerifierRunScorer",
    "VerifierScorer",
    "VerifierScores",
    "artifact_template_error",
    "batch_plan_record",
    "build_inputs",
    "build_manifest",
    "length_limit",
    "load_classifier",
    "load_verifier",
    "manifest_frame",
    "plan_batches",
    "read_calibration",
    "read_verifier",
    "render_scoring_text",
    "resolve_labels",
    "result_rows",
    "softmax",
    "tally",
    "tf32_state",
    "verifier_result_rows",
]

#: ``hyperparams.artifact`` for a verifier trained by ``seq_cls``: score it as it was trained.
VERIFIER_ARTIFACT = "strata_verifier"
#: The fixed templates of artifact scoring: they only name the two roles.
ARTIFACT_TEMPLATE = "{premises}"
ARTIFACT_PAIR_TEMPLATE = "{hypothesis}"
VERIFIER_FILENAME = "strata_verifier.json"
VERIFIER_SCHEMA = "strata.verifier/v1"
CALIBRATION_FILENAME = "calibration.json"
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

    A list's elements must be scalars (a string or a number): a ``list<struct>`` column would
    otherwise render as Python dict text, and the model would score that text as the premises.
    A boolean is refused too, since ``True`` is not a premise either. No length cap here: scoring
    never truncates, so an over-long rendering is dropped and counted by the caller rather than cut.
    """

    def _sub(match: re.Match[str]) -> str:
        column = column_mapping.get(match.group(1))
        if column is None or column not in row:
            return match.group(0)
        value: object = row[column]
        if isinstance(value, list | tuple):
            items = cast("Sequence[object]", value)
            for item in items:
                if isinstance(item, bool) or not isinstance(item, str | int | float):
                    msg = (
                        f"column {column!r} is a list of {type(item).__name__} values, which would "
                        "be scored as their Python text; stage a list<string> column holding the "
                        "text to score"
                    )
                    raise RunError(msg)
            return separator.join(str(item) for item in items)
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
    """Every row's rendered pair, plus its stable id and slice when the spec names them.

    ``chat_kwargs`` holds each row's rendered chat-template kwargs when the spec declares any.
    """

    texts: list[str]
    pairs: list[str]
    item_ids: list[str] | None
    slices: list[str] | None
    chat_kwargs: list[dict[str, JsonValue]] | None = None
    #: Artifact scoring: each row's raw ``(premises, hypothesis)`` cells, which the verifier's
    #: own rendering turns into text (``texts`` and ``pairs`` are then empty).
    cells: list[tuple[Any, Any]] | None = None


def build_inputs(spec: RunSpec, rows: Sequence[Mapping[str, Any]]) -> ScoringInputs:
    """Render every row and read its id and slice, refusing what would make them ambiguous."""
    hp = spec.hyperparams
    pair_template = hp.pair_template or ""
    columns = set(rows[0]) if rows else set[str]()
    for what, column in (("id_column", hp.id_column), ("slice_column", hp.slice_column)):
        if column is not None and rows and column not in columns:
            msg = f"{what} {column!r} is not a column of the split; columns: {sorted(columns)}"
            raise RunError(msg)
    cells: list[tuple[Any, Any]] | None = None
    if hp.artifact is not None:
        # A trained verifier renders its own rows (`VerifierScorer.render`), from the raw cells
        # of the columns the two fixed templates name.
        texts: list[str] = []
        pairs: list[str] = []
        named = [spec.column_mapping.get("premises"), spec.column_mapping.get("hypothesis")]
        missing = [c for c in named if c is None or (rows and c not in columns)]
        if missing:
            msg = f"artifact scoring needs the premises and hypothesis columns; columns: {sorted(columns)}"
            raise RunError(msg)
        cells = [(row.get(named[0] or ""), row.get(named[1] or "")) for row in rows]
    else:
        texts = [
            render_scoring_text(
                spec.template, row, spec.column_mapping, separator=hp.list_separator
            )
            for row in rows
        ]
        pairs = [
            render_scoring_text(
                pair_template, row, spec.column_mapping, separator=hp.list_separator
            )
            for row in rows
        ]
    item_ids = _item_ids(rows, hp.id_column) if hp.id_column is not None else None
    slices = _slices(rows, hp.slice_column) if hp.slice_column is not None else None
    declared = hp.chat_template_kwargs
    chat_kwargs = (
        [
            render_kwargs(
                declared,
                lambda text, row=row: render_scoring_text(
                    text, row, spec.column_mapping, separator=hp.list_separator
                ),
            )
            for row in rows
        ]
        if declared
        else None
    )
    return ScoringInputs(
        texts=texts,
        pairs=pairs,
        item_ids=item_ids,
        slices=slices,
        chat_kwargs=chat_kwargs,
        cells=cells,
    )


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
    *,
    temperature: float | None = None,
) -> list[dict[str, Any]]:
    """One results row per input row, in input order; an unscored row carries no score.

    ``temperature`` divides the logits before the softmax (``p_*`` only; the ``logit_*`` columns
    stay raw). ``None`` is temperature 1, no calibration.

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
            probabilities = softmax(
                values if temperature is None else [value / temperature for value in values]
            )
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

    @property
    def attention(self) -> str | None:
        """The attention implementation the loaded model runs (``sdpa``, ``eager``, ...)."""
        value: object = getattr(self._model.config, "_attn_implementation", None)
        return value if isinstance(value, str) else None

    def synchronize(self) -> None:
        """Wait for the device's queued work, so a clock read after it has seen that work end."""
        if self.device == "cuda":
            self._torch.cuda.synchronize()

    def timed_forward(
        self,
        texts: Sequence[str],
        pairs: Sequence[str],
        *,
        clock: Callable[[], int],
        model_only: bool,
    ) -> tuple[int, int, int | None]:
        """One request timed end to end, and its forward pass alone when ``model_only``.

        End to end runs from the text pairs in memory, through tokenization, the host-to-device
        copy, the forward pass and a softmax over every class, to the probabilities copied back
        to the host, and stops after the device is synchronized. The model-only interval brackets
        the forward pass with a synchronization on each side. Returns ``(start, end, model)`` in
        the clock's nanoseconds; ``model`` is ``None`` when it was not measured.
        """
        torch = self._torch
        start = clock()
        features: Any = self._tokenizer(
            list(texts), list(pairs), padding=True, truncation=False, return_tensors="pt"
        )
        if int(features["input_ids"].shape[1]) > self.max_length:
            msg = "a batch exceeded the model's length limit after planning; nothing was timed"
            raise RunError(msg)
        moved = {key: value.to(self.device) for key, value in features.items()}
        model_ns: int | None = None
        forward_start = 0
        with torch.inference_mode():
            if model_only:
                self.synchronize()
                forward_start = clock()
            output: Any = self._model(**moved)
            if model_only:
                self.synchronize()
                model_ns = clock() - forward_start
            probabilities: Any = torch.softmax(output.logits.float(), dim=-1).cpu()
        self.synchronize()
        end = clock()
        del probabilities
        return start, end, model_ns


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
    torch, transformers, tf32 = _scoring_modules(dtype)
    tokenizer, model, unexpected = _load_model(transformers, torch, snapshot, dtype)
    config: Any = model.config
    names, outputs = resolve_labels(cast("Mapping[Any, Any]", config.id2label), label_alias)
    # Before padding is aligned: the position offset is the CHECKPOINT's padding index.
    limit, source = length_limit(config, tokenizer, max_length)
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


def _scoring_modules(dtype: str) -> tuple[Any, Any, dict[str, bool | None]]:
    """torch and transformers, with TF32 switched off for a float32 run, and the TF32 state."""
    if dtype not in SCORING_DTYPES:
        msg = f"dtype must be one of {', '.join(SCORING_DTYPES)}"
        raise RunError(msg)
    try:
        torch: Any = import_module("torch")
        transformers: Any = import_module("transformers")
    except ImportError as exc:
        msg = "classifier scoring needs torch and transformers: pip install 'strata-forge[serving]'"
        raise RunError(msg) from exc
    return torch, transformers, tf32_state(torch, disable=dtype == "float32")


def _load_model(
    transformers: Any, torch: Any, snapshot: Path, dtype: str
) -> tuple[Any, Any, list[str]]:
    """The snapshot's tokenizer and sequence classifier, local files only, refusing a partial load."""
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
    return tokenizer, model, unexpected


def tf32_state(torch: Any, *, disable: bool) -> dict[str, bool | None]:
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


def length_limit(config: Any, tokenizer: Any, requested: int | None) -> tuple[int, str]:
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


#: Hyperparameters that hold prompt text, recorded in a manifest as digests only.
_PROMPT_TEXT_KNOBS = frozenset({"pair_template", "assistant_prefill", "chat_template_kwargs"})


def manifest_frame(
    spec: RunSpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    model_commit: str | None,
    dataset_commit: str | None,
    scoring: dict[str, Any],
) -> dict[str, Any]:
    """A scoring run's manifest (``kind`` ``scoring``) around one engine's ``scoring`` block.

    Pins (the commits actually read beside the ones requested), template digests, the mapping,
    the knobs and the counts, with no row text: templates are recorded as digests, and the knobs
    that hold prompt text (the pair template, an assistant prefill, chat-template kwargs) are left
    out of the hyperparameters for the same reason, since the control plane stores their text
    with the spec and a pushed manifest should not republish it. The engine's ``scoring`` block
    carries their digests.
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
        "scoring": scoring,
        "template_sha256": _digest(spec.template),
        "pair_template_sha256": _digest(hp.pair_template or ""),
        "column_mapping": spec.column_mapping,
        "hyperparams": hp.model_dump(exclude=set(_PROMPT_TEXT_KNOBS)),
        "counts": tally(rows),
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
    """A classifier scoring run's manifest: :func:`manifest_frame` with the head's setup.

    A trained verifier (:class:`VerifierScorer`) adds its artifact record and calibration.
    """
    hp = spec.hyperparams
    verifier = classifier if isinstance(classifier, VerifierScorer) else None
    outputs = ["logit", "p_softmax_t1_over_included", "pred_argmax_over_included"]
    calibration: dict[str, Any] | None = None
    artifact: dict[str, Any] | None = None
    if verifier is not None:
        if verifier.temperature is not None:
            outputs[1] = "p_softmax_calibrated_over_included"
        outputs += [f"aux_{head}" for head in verifier.heads]
        record = verifier.calibration
        calibration = {
            "mode": hp.apply_calibration,
            "applied": verifier.temperature is not None,
            "temperature": verifier.temperature,
            "artifact_record": (
                None
                if record is None
                else {
                    "sha256": record["sha256"],
                    "fit": cast("dict[str, Any]", record["record"])["fit"],
                }
            ),
            "raw_logits_kept": True,
        }
        artifact = {
            "kind": hp.artifact,
            "verifier_sha256": verifier.verifier_sha256,
            "rendering": verifier.rendering.to_verifier(),
            "pooling": verifier.verifier.get("pooling"),
            "heads": verifier.heads,
            "support_scores": "list of {pid, logit, p} per premise, in premise order"
            if "support" in verifier.heads
            else None,
        }
    return manifest_frame(
        spec,
        rows,
        model_commit=model_commit,
        dataset_commit=dataset_commit,
        scoring={
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
            "input": "strata_verifier" if verifier is not None else "pair",
            "artifact": artifact,
            "outputs": outputs,
            "calibration": calibration,
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
    )


class ClassifierScorer:
    """A :class:`Classifier` in the shape the runner drives every scoring engine through."""

    def __init__(self, classifier: Classifier) -> None:
        self.classifier = classifier
        self.max_length = classifier.max_length

    def lengths(self, inputs: ScoringInputs) -> list[int | None]:
        return self.classifier.lengths(inputs.texts, inputs.pairs)

    def score(self, inputs: ScoringInputs, batch: Sequence[int]) -> list[list[float]]:
        return self.classifier.logits(
            [inputs.texts[i] for i in batch], [inputs.pairs[i] for i in batch]
        )

    def result_rows(
        self,
        lengths: Sequence[int | None],
        values: Mapping[int, Sequence[float]],
        inputs: ScoringInputs,
    ) -> list[dict[str, Any]]:
        return result_rows(lengths, values, self.classifier.outputs, inputs)

    def manifest(
        self,
        spec: RunSpec,
        rows: Sequence[Mapping[str, Any]],
        *,
        model_commit: str | None,
        dataset_commit: str | None,
        batches: Sequence[Sequence[int]],
    ) -> dict[str, Any]:
        return build_manifest(
            spec,
            self.classifier,
            rows,
            model_commit=model_commit,
            dataset_commit=dataset_commit,
            batches=batches,
        )


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------- trained verifiers


@dataclass(frozen=True, slots=True)
class VerifierScores:
    """One row's scores from a trained verifier: every class's raw logit, and the auxiliary
    heads' raw logits when the artifact has them (``support`` as ``(pid, logit)`` per premise)."""

    logits: list[float]
    flag: float | None
    support: list[tuple[str, float]] | None


def artifact_template_error(template: str, hyperparams: Any) -> str | None:
    """Why a spec's templates do not fit artifact scoring, or ``None`` when they do.

    A trained verifier renders its own input from ``strata_verifier.json``; the run's templates
    only NAME the two roles, so they are fixed: ``template`` is ``{premises}`` and
    ``pair_template`` is ``{hypothesis}``, and ``column_mapping`` maps each to its column.
    """
    if getattr(hyperparams, "artifact", None) is None:
        return None
    if template != ARTIFACT_TEMPLATE or hyperparams.pair_template != ARTIFACT_PAIR_TEMPLATE:
        return (
            f"artifact scoring renders rows from the verifier's own strata_verifier.json, so the "
            f"templates only name the roles: template {ARTIFACT_TEMPLATE!r} and pair_template "
            f"{ARTIFACT_PAIR_TEMPLATE!r}, each mapped to its column"
        )
    return None


def read_verifier(snapshot: Path) -> dict[str, Any]:
    """The artifact's ``strata_verifier.json``, refusing a checkpoint that is not one."""
    path = snapshot / VERIFIER_FILENAME
    if not path.is_file():
        msg = (
            f"the model has no {VERIFIER_FILENAME}, so it is not a verifier trained by seq_cls; "
            "score it in pair mode instead (no hyperparams.artifact)"
        )
        raise RunError(msg)
    try:
        raw: Any = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        msg = f"{VERIFIER_FILENAME} could not be read as JSON"
        raise RunError(msg) from exc
    record = cast("dict[str, Any]", raw) if isinstance(raw, dict) else {}
    if record.get("schema") != VERIFIER_SCHEMA:
        msg = f"{VERIFIER_FILENAME} is not a {VERIFIER_SCHEMA} record"
        raise RunError(msg)
    return record


def read_calibration(snapshot: Path) -> dict[str, Any] | None:
    """The artifact's ``calibration.json`` and its digest, or ``None`` when it has none."""
    path = snapshot / CALIBRATION_FILENAME
    if not path.is_file():
        return None
    text = path.read_text()
    try:
        record: Any = json.loads(text)
        temperature = float(record["fit"]["temperature"])
    except (ValueError, KeyError, TypeError) as exc:
        msg = f"{CALIBRATION_FILENAME} holds no fitted temperature"
        raise RunError(msg) from exc
    if not (math.isfinite(temperature) and temperature > 0):
        msg = f"{CALIBRATION_FILENAME} holds a temperature that is not a positive number"
        raise RunError(msg)
    return {"temperature": temperature, "record": record, "sha256": _digest(text)}


class VerifierScorer(Classifier):
    """A ``seq_cls`` artifact, scored with the rendering it was trained with.

    The rendering, the label order, the pooling and the auxiliary heads all come from the
    artifact (``strata_verifier.json``), through the same functions the trainer calls
    (:mod:`strata_forge.training.verifier_render`,
    :func:`~strata_forge.training.classifier_heads.aux_outputs`).
    """

    def __init__(
        self,
        *,
        rendering: Rendering,
        verifier: dict[str, Any],
        verifier_sha256: str,
        aux: Any,
        hidden: list[Any],
        calibration: dict[str, Any] | None,
        apply_calibration: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.rendering = rendering
        self.verifier = verifier
        self.verifier_sha256 = verifier_sha256
        self._aux = aux
        self._hidden = hidden
        self.calibration = calibration
        self.apply_calibration = apply_calibration
        self.heads = [h for h in ("flag", "support") if getattr(aux, h, None) is not None]
        self._pad_id: Any = getattr(self._tokenizer, "pad_token_id", None)

    @property
    def temperature(self) -> float | None:
        """The temperature applied to ``p_*``: only under ``apply_calibration: auto``."""
        if not self.apply_calibration or self.calibration is None:
            return None
        return float(self.calibration["temperature"])

    def render(self, premises: Sequence[Any], hypotheses: Sequence[Any]) -> list[RenderedRow]:
        """Every row's text; a malformed cell refuses the run, naming the row and never the cell."""
        out: list[RenderedRow] = []
        for index, (cell, hypothesis) in enumerate(zip(premises, hypotheses, strict=True)):
            try:
                out.append(render_row(self.rendering, cell, hypothesis))
            except DatasetFormatError as exc:
                msg = f"row {index}: {exc}"
                raise RunError(msg) from exc
        return out

    def measure(self, rows: Sequence[RenderedRow]) -> list[int | None]:
        """Each rendered row's token count with special tokens; ``None`` past the character cap."""
        out: list[int | None] = []
        ends_on_pad: list[int] = []
        for offset in range(0, len(rows), _MEASURE_CHUNK):
            chunk = list(rows[offset : offset + _MEASURE_CHUNK])
            measurable = [
                i
                for i, row in enumerate(chunk)
                if len(row.text) + len(row.pair or "") <= _MAX_RENDERED_CHARS
            ]
            lengths: list[int | None] = [None] * len(chunk)
            if measurable:
                encoded: Any = encode_rows(self._tokenizer, [chunk[i] for i in measurable])
                for i, ids in zip(measurable, encoded["input_ids"], strict=True):
                    tokens = cast("list[int]", ids)
                    if self.rendering.decoder and tokens and tokens[-1] == self._pad_id:
                        ends_on_pad.append(offset + i)
                    lengths[i] = len(tokens)
            out.extend(lengths)
        if ends_on_pad:
            # The stock decoder head reads the rightmost token that is not the pad id, the
            # auxiliary heads the last one the attention mask keeps; on such a row they would read
            # two positions, so it is refused, as training refused it.
            msg = (
                f"{len(ends_on_pad)} row(s) end on the padding token (first rows "
                f"{ends_on_pad[:5]}), so the answer position cannot be found"
            )
            raise RunError(msg)
        return out

    def score(self, rows: Sequence[RenderedRow]) -> list[VerifierScores]:
        """One batch: raw logits for every class and the auxiliary heads, right-padded."""
        torch = self._torch
        features: Any = encode_rows(
            self._tokenizer,
            rows,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=self._aux.support is not None,
        )
        if int(features["input_ids"].shape[1]) > self.max_length:
            msg = "a batch exceeded the model's length limit after planning; nothing was scored"
            raise RunError(msg)
        ids: Any = features["input_ids"].to(self.device)
        mask: Any = features["attention_mask"].to(self.device)
        support_markers: list[list[int]] = []
        positions: Any = None
        if self._aux.support is not None:
            offsets: Any = features["offset_mapping"].tolist()
            for i, row in enumerate(rows):
                found = marker_tokens(row.markers, offsets[i], features.sequence_ids(i))
                if found is None:
                    msg = "a premise id's first character is covered by no token"
                    raise RunError(msg)
                support_markers.append(found)
            width = max((len(m) for m in support_markers), default=0)
            if width:
                positions = torch.tensor(
                    [m + [0] * (width - len(m)) for m in support_markers], device=self.device
                )
        self._hidden.clear()
        with torch.inference_mode():
            output: Any = self._model(input_ids=ids, attention_mask=mask)
            aux: dict[str, Any] = (
                aux_outputs(
                    torch,
                    self._aux,
                    self._hidden[-1] if self._hidden else None,
                    mask,
                    last_token=self.rendering.decoder,
                    support_positions=positions,
                )
                if self.heads
                else {}
            )
        logits = cast("list[list[float]]", output.logits.float().cpu().tolist())
        flags: Any = aux.get("flag_logits")
        flag_values = (
            cast("list[float]", flags.float().cpu().tolist()) if flags is not None else None
        )
        support: Any = aux.get("support_logits")
        support_values = (
            cast("list[list[float]]", support.float().cpu().tolist())
            if support is not None
            else None
        )
        out: list[VerifierScores] = []
        for i, row in enumerate(rows):
            per_premise: list[tuple[str, float]] | None = None
            if self._aux.support is not None:
                values = support_values[i] if support_values is not None else []
                per_premise = [(pid, float(values[k])) for k, (pid, _) in enumerate(row.markers)]
            scores = VerifierScores(
                logits=[float(v) for v in logits[i]],
                flag=float(flag_values[i]) if flag_values is not None else None,
                support=per_premise,
            )
            numbers = [*scores.logits, *([scores.flag] if scores.flag is not None else [])]
            numbers += [v for _, v in scores.support or []]
            if len(scores.logits) != len(self.names) or not all(math.isfinite(v) for v in numbers):
                msg = "the verifier returned a non-finite or mis-shaped score"
                raise RunError(msg)
            out.append(scores)
        return out


def load_verifier(
    snapshot: Path,
    *,
    label_alias: Mapping[str, str | None] | None,
    max_length: int | None,
    dtype: str,
    apply_calibration: bool,
) -> VerifierScorer:
    """Load a ``seq_cls`` artifact for scoring: its rendering, labels, heads and calibration.

    The label order is the artifact's: its ``id2label`` must name the verifier's labels in order,
    and ``label_alias`` (optional here) defaults to each label as itself. The length limit is the
    artifact's training ``max_length`` unless ``max_model_len`` asks for another. The
    calibration record is read whenever present (and recorded); it changes ``p_*`` only under
    ``apply_calibration``, and the raw logits are always kept.
    """
    verifier = read_verifier(snapshot)
    try:
        rendering = Rendering.from_verifier(verifier)
    except ValueError as exc:
        raise RunError(str(exc)) from exc
    raw_labels: Any = verifier.get("labels")
    if not isinstance(raw_labels, list):
        msg = f"{VERIFIER_FILENAME} has no label list"
        raise RunError(msg)
    labels = [str(x) for x in cast("list[object]", raw_labels)]
    pooling = "last_token" if rendering.decoder else "first_token"
    if verifier.get("pooling") != pooling:
        msg = f"{VERIFIER_FILENAME} pools at {verifier.get('pooling')!r}; its rendering reads {pooling}"
        raise RunError(msg)
    torch, transformers, tf32 = _scoring_modules(dtype)
    tokenizer, model, unexpected = _load_model(transformers, torch, snapshot, dtype)
    config: Any = model.config
    alias: Mapping[str, str | None] = (
        label_alias if label_alias is not None else {name: name for name in labels}
    )
    names, outputs = resolve_labels(cast("Mapping[Any, Any]", config.id2label), alias)
    if names != labels:
        msg = f"the model's id2label {names} is not the verifier's label order {labels}"
        raise RunError(msg)
    if not rendering.decoder and getattr(config, "classifier_pooling", "cls") != "cls":
        msg = "an encoder verifier must pool at its first token (classifier_pooling 'cls')"
        raise RunError(msg)
    if max_length is not None and max_length > rendering.max_length:
        msg = (
            f"max_model_len {max_length} exceeds the {rendering.max_length} tokens the verifier was "
            "trained on; a verifier is scored only on rows it could have been trained on"
        )
        raise RunError(msg)
    limit, source = length_limit(config, tokenizer, max_length or rendering.max_length)
    if max_length is None:
        source = "artifact"
    pad_token = _align_padding(tokenizer, config)
    heads: Any = verifier.get("outputs") or []
    aux = build_aux_heads(
        int(config.hidden_size), flag="flag" in heads, support="support" in heads, std=0.0
    )
    try:
        load_aux_heads(aux, snapshot)
    except HeadLoadError as exc:
        raise RunError(str(exc)) from exc
    hidden: list[Any] = []
    if "flag" in heads or "support" in heads:

        def _capture(_module: Any, _args: Any, output: Any) -> None:
            hidden.append(output[0])

        model.base_model.register_forward_hook(_capture)
    device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    model.to(device)
    aux.to(device=device, dtype=model.dtype)
    model.eval()
    aux.eval()
    return VerifierScorer(
        rendering=rendering,
        verifier=verifier,
        verifier_sha256=_digest((snapshot / VERIFIER_FILENAME).read_text()),
        aux=aux,
        hidden=hidden,
        calibration=read_calibration(snapshot),
        apply_calibration=apply_calibration,
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


def verifier_result_rows(
    lengths: Sequence[int | None],
    scores: Mapping[int, VerifierScores],
    scorer: VerifierScorer,
    inputs: ScoringInputs,
) -> list[dict[str, Any]]:
    """:func:`result_rows` for a verifier, plus its auxiliary heads' columns.

    ``p_*`` is the softmax at the applied temperature (1 unless calibration is applied); the
    ``logit_*`` columns are always the raw logits. With a flag head: ``logit_inconsistent`` and
    ``p_inconsistent`` (its sigmoid). With a support head: ``support_scores``, a list of
    ``{pid, logit, p}`` per premise in premise order. An unscored row carries ``None`` in each.
    """
    rows = result_rows(
        lengths,
        {i: s.logits for i, s in scores.items()},
        scorer.outputs,
        inputs,
        temperature=scorer.temperature,
    )
    for index, row in enumerate(rows):
        entry = scores.get(index)
        if "flag" in scorer.heads:
            flag = entry.flag if entry is not None else None
            row["logit_inconsistent"] = flag
            row["p_inconsistent"] = None if flag is None else _sigmoid(flag)
        if "support" in scorer.heads:
            row["support_scores"] = (
                None
                if entry is None or entry.support is None
                else [{"pid": pid, "logit": v, "p": _sigmoid(v)} for pid, v in entry.support]
            )
    return rows


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp = math.exp(value)
    return exp / (1.0 + exp)


class VerifierRunScorer:
    """A :class:`VerifierScorer` in the shape the runner drives every scoring engine through.

    The rows are rendered from the raw ``premises`` and ``hypothesis`` cells
    (``ScoringInputs.cells``) by the artifact's own rendering, once, on the first measure; each
    batch's auxiliary scores are kept beside the logits the frame collects, for the result rows.
    """

    def __init__(self, scorer: VerifierScorer) -> None:
        self.scorer = scorer
        self.max_length = scorer.max_length
        self._rendered: list[RenderedRow] | None = None
        self._scores: dict[int, VerifierScores] = {}

    def _rows(self, inputs: ScoringInputs) -> list[RenderedRow]:
        if self._rendered is None:
            cells = inputs.cells or []
            self._rendered = self.scorer.render([c[0] for c in cells], [c[1] for c in cells])
        return self._rendered

    def lengths(self, inputs: ScoringInputs) -> list[int | None]:
        return self.scorer.measure(self._rows(inputs))

    def score(self, inputs: ScoringInputs, batch: Sequence[int]) -> list[list[float]]:
        rows = self._rows(inputs)
        values = self.scorer.score([rows[i] for i in batch])
        self._scores.update(zip(batch, values, strict=True))
        return [v.logits for v in values]

    def result_rows(
        self,
        lengths: Sequence[int | None],
        values: Mapping[int, Sequence[float]],
        inputs: ScoringInputs,
    ) -> list[dict[str, Any]]:
        del values  # the same logits, held with their auxiliary scores
        return verifier_result_rows(lengths, self._scores, self.scorer, inputs)

    def manifest(
        self,
        spec: RunSpec,
        rows: Sequence[Mapping[str, Any]],
        *,
        model_commit: str | None,
        dataset_commit: str | None,
        batches: Sequence[Sequence[int]],
    ) -> dict[str, Any]:
        return build_manifest(
            spec,
            self.scorer,
            rows,
            model_commit=model_commit,
            dataset_commit=dataset_commit,
            batches=batches,
        )
