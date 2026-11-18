"""Exact label-token read-outs from a causal LM after its chat template (``engine="hf_label_logits"``).

A label-logits run asks a generative model a question and reads, instead of a sampled answer,
the next-token logits of a few DECLARED continuations — one token per label (``" True"``,
``" False"``, ``" Unknown"``; a judge's ``" yes"`` / ``" no"``). Each row renders into one user
message (the spec's ``template``), the model's own chat template turns it into a prompt with the
assistant's generation header (``add_generation_prompt=True``) and the run's declared
``chat_template_kwargs``, and an optional ``assistant_prefill`` string is appended after the
header (a judge that answers ``<score> yes``, read after ``<score>``). One forward pass reads the
logits at the prompt's last position.

The rules that make the numbers trustworthy:

- **Each label is one distinct token.** A verbalizer that tokenizes to several tokens, or two
  verbalizers that share a token, are refused at load. The token is checked twice: on its own,
  and as the actual continuation of the first row's prompt, so a tokenizer that merges it with
  the prefill's last characters is caught before any GPU time is spent. The resolved ids are
  recorded.
- **Nothing is hidden.** Every row carries the RAW logit of each declared token, their softmax
  over the declared tokens only (``p_<label>``, temperature 1, no calibration), the argmax, and
  ``mass_outside``: the probability the full output distribution puts on every OTHER token, so a
  reader sees how much of the model's answer the declared labels did not cover.
- **The prompt is recorded, not guessed.** The manifest holds the sha256 of the chat template the
  tokenizer used, of the canonical kwargs and of the prefill, the prefill's token ids (checked to
  be the same tokens on every row) and the label ids.
- **Nothing is truncated.** A prompt over the model's limit is dropped and counted, exactly as the
  classifier engine does.
- **The precision is declared.** ``dtype`` is required (``float32`` or ``bfloat16``), as for the
  classifier; in ``float32`` the TF32 matmul and cuDNN paths are switched off first, and the
  manifest records both switches and the batch plan.

Heavy dependencies (torch, transformers) are imported inside :func:`load_label_reader`.
"""

from __future__ import annotations

import inspect
import math
from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

from strata_forge.pipelines._chat_template import (
    kwargs_digest,
    template_digest,
    unread_kwargs,
)
from strata_forge.pipelines._common import RunError
from strata_forge.pipelines._scoring import (
    SCORING_DTYPES,
    batch_plan_record,
    length_limit,
    manifest_frame,
    result_rows,
    tf32_state,
)
from strata_forge.training.loading import (
    LoadRefusedError,
    check_loading_info,
    model_load_kwargs,
    tokenizer_load_kwargs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pydantic import JsonValue

    from strata_forge.pipelines._scoring import ScoringInputs
    from strata_forge.pipelines.inference_runner import LabelToken, RunSpec

__all__ = ["LabelReader", "load_label_reader", "read_out"]

# A rendered prompt longer than this is over any model's limit without tokenizing it.
_MAX_RENDERED_CHARS = 200_000
_MEASURE_CHUNK = 256


def read_out(declared: Sequence[float], log_normalizer: float) -> float:
    """The full-vocabulary probability outside the declared tokens.

    ``declared`` are the declared tokens' logits and ``log_normalizer`` is the log-sum-exp over
    the WHOLE output vocabulary. Computed as ``-expm1(lse(declared) - lse(all))``, so a mass near
    zero keeps its precision instead of cancelling in ``1 - sum``; clamped at zero, since rounding
    can put the declared share a hair above one.
    """
    peak = max(declared)
    inside = peak + math.log(math.fsum(math.exp(value - peak) for value in declared))
    return max(0.0, -math.expm1(inside - log_normalizer))


class LabelReader:
    """A loaded causal LM that renders, measures and reads rows, synchronously.

    The runner calls it from a worker thread, one batch at a time.
    """

    def __init__(
        self,
        *,
        model: Any,  # a transformers PreTrainedModel; untyped third-party object
        tokenizer: Any,  # a transformers tokenizer; untyped third-party object
        torch: Any,  # the torch module, imported lazily
        labels: list[str],
        tokens: list[str],
        token_ids: list[int],
        token_pieces: list[str],
        prefill: str,
        chat_template: str,
        vocab_size: int,
        max_length: int,
        max_length_source: str,
        dtype: str,
        tf32: dict[str, bool | None],
        device: str,
        unexpected_keys: list[str],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self.labels = labels
        self.tokens = tokens
        self.token_ids = token_ids
        self.token_pieces = token_pieces
        self.prefill = prefill
        self.chat_template = chat_template
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.max_length_source = max_length_source
        self.dtype = dtype
        self.tf32 = tf32
        self.device = device
        self.unexpected_keys = unexpected_keys
        #: The prefill's tokens as they follow the generation header; ``None`` until measured.
        self.prefill_ids: list[int] | None = None
        pad: object = tokenizer.pad_token_id
        if not isinstance(pad, int):
            pad = tokenizer.eos_token_id
        # Any id works: padded positions are masked, and right padding never shifts a real one.
        self._pad_id = pad if isinstance(pad, int) else 0

    def _render(self, text: str, kwargs: Mapping[str, JsonValue]) -> str:
        try:
            rendered: Any = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
                **kwargs,
            )
        except Exception as exc:
            # A template may raise on purpose (`raise_exception`) when a kwarg is wrong; that is
            # the spec's problem, the same for every row, so the run stops here.
            msg = f"the model's chat template refused the prompt: {exc}"
            raise RunError(msg) from exc
        return cast("str", rendered)

    def _ids(self, texts: list[str]) -> list[list[int]]:
        encoded: Any = self._tokenizer(texts, add_special_tokens=False)
        return cast("list[list[int]]", encoded["input_ids"])

    def _prompts(self, inputs: ScoringInputs, rows: Sequence[int]) -> list[str | None]:
        """Each row's rendered prompt, without the prefill; ``None`` past the character cap."""
        kwargs = inputs.chat_kwargs
        out: list[str | None] = []
        for index in rows:
            rendered = self._render(inputs.texts[index], kwargs[index] if kwargs else {})
            out.append(
                rendered if len(rendered) + len(self.prefill) <= _MAX_RENDERED_CHARS else None
            )
        return out

    def _encode(self, prompts: Sequence[str]) -> list[list[int]]:
        """Token ids of each prompt followed by the prefill, checking the prefill's tokens.

        The prompt and the prefill are tokenized as one text, as the model reads them. The
        prompt's own tokens must be a prefix of that, and what follows them must be the same
        prefill tokens on every row; otherwise the prefill is not one token sequence the record
        could name.
        """
        full = self._ids([prompt + self.prefill for prompt in prompts])
        if not self.prefill:
            return full
        heads = self._ids(list(prompts))
        for head, ids in zip(heads, full, strict=True):
            tail = ids[len(head) :]
            if ids[: len(head)] != head or not tail:
                msg = (
                    "the assistant prefill merges with the end of the rendered prompt when "
                    "tokenized, so it is not a token sequence of its own; change the prefill"
                )
                raise RunError(msg)
            if self.prefill_ids is None:
                self.prefill_ids = tail
            elif tail != self.prefill_ids:
                msg = "the assistant prefill tokenizes differently on different rows"
                raise RunError(msg)
        return full

    def _check_continuations(self, prompt: str, ids: list[int]) -> None:
        """Each label token must be the single next token after the prompt and prefill."""
        texts = [prompt + self.prefill + token for token in self.tokens]
        for label, token_id, extended in zip(
            self.labels, self.token_ids, self._ids(texts), strict=True
        ):
            if extended != [*ids, token_id]:
                msg = (
                    f"the token for label {label!r} is not a single-token continuation of the "
                    f"prompt: after the prompt it tokenizes as {extended[len(ids) :][:8]}, not "
                    f"[{token_id}]"
                )
                raise RunError(msg)

    def lengths(self, inputs: ScoringInputs) -> list[int | None]:
        """Each row's prompt length in tokens, prefill included; ``None`` past the character cap."""
        out: list[int | None] = []
        checked = False
        for offset in range(0, len(inputs.texts), _MEASURE_CHUNK):
            rows = range(offset, min(offset + _MEASURE_CHUNK, len(inputs.texts)))
            prompts = self._prompts(inputs, rows)
            measurable = [prompt for prompt in prompts if prompt is not None]
            encoded = iter(self._encode(measurable)) if measurable else iter(())
            for prompt in prompts:
                if prompt is None:
                    out.append(None)
                    continue
                ids = next(encoded)
                if not checked:
                    self._check_continuations(prompt, ids)
                    checked = True
                out.append(len(ids))
        return out

    def score(self, inputs: ScoringInputs, batch: Sequence[int]) -> list[list[float]]:
        """For each row: the declared tokens' raw logits, then the full-vocabulary log-sum-exp."""
        torch = self._torch
        prompts = [prompt for prompt in self._prompts(inputs, batch) if prompt is not None]
        if len(prompts) != len(batch):
            msg = "a batch held a row over the length limit after planning; nothing was scored"
            raise RunError(msg)
        encoded = self._encode(prompts)
        lengths = [len(ids) for ids in encoded]
        if max(lengths) > self.max_length:
            msg = "a batch exceeded the model's length limit after planning; nothing was scored"
            raise RunError(msg)
        width = max(lengths)
        input_ids = torch.full((len(encoded), width), self._pad_id, dtype=torch.long)
        mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[row, : len(ids)] = 1
        # Right padding: a real token's position and attention never depend on the padding after
        # it. Only the last real position of each row is projected onto the vocabulary.
        positions = sorted({length - 1 for length in lengths})
        column = {position: index for index, position in enumerate(positions)}
        keep = torch.tensor(positions, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output: Any = self._model(
                input_ids=input_ids.to(self.device),
                attention_mask=mask.to(self.device),
                logits_to_keep=keep,
                use_cache=False,
            )
            logits: Any = output.logits
            if int(logits.shape[-1]) != self.vocab_size:
                msg = f"the model returned logits over {int(logits.shape[-1])} tokens, not {self.vocab_size}"
                raise RunError(msg)
            rows = torch.arange(len(encoded), device=logits.device)
            cols = torch.tensor([column[length - 1] for length in lengths], device=logits.device)
            last = logits[rows, cols].double()
            normalizer = torch.logsumexp(last, dim=-1)
            declared = last[:, torch.tensor(self.token_ids, device=logits.device)]
            values = torch.cat([declared, normalizer.unsqueeze(-1)], dim=-1).cpu().tolist()
        out = cast("list[list[float]]", values)
        if any(not math.isfinite(value) for row in out for value in row):
            msg = "the model returned a non-finite logit"
            raise RunError(msg)
        return [[float(value) for value in row] for row in out]

    def result_rows(
        self,
        lengths: Sequence[int | None],
        values: Mapping[int, Sequence[float]],
        inputs: ScoringInputs,
    ) -> list[dict[str, Any]]:
        """The classifier's row shape over the declared tokens, plus ``mass_outside``."""
        declared = {index: list(row[:-1]) for index, row in values.items()}
        rows = result_rows(lengths, declared, self.labels, inputs)
        for index, row in enumerate(rows):
            read = values.get(index)
            row["mass_outside"] = None if read is None else read_out(read[:-1], read[-1])
        return rows

    def manifest(
        self,
        spec: RunSpec,
        rows: Sequence[Mapping[str, Any]],
        *,
        model_commit: str | None,
        dataset_commit: str | None,
        batches: Sequence[Sequence[int]],
    ) -> dict[str, Any]:
        """The run's manifest (``kind`` ``scoring``): the prompt's digests, the ids, the counts."""
        hp = spec.hyperparams
        return manifest_frame(
            spec,
            rows,
            model_commit=model_commit,
            dataset_commit=dataset_commit,
            scoring={
                "engine": hp.engine,
                "head": "causal_lm_next_token",
                "labels": self.labels,
                "label_tokens": [
                    {"label": label, "token": token, "token_id": token_id, "piece": piece}
                    for label, token, token_id, piece in zip(
                        self.labels, self.tokens, self.token_ids, self.token_pieces, strict=True
                    )
                ],
                "outputs": ["logit", "p_softmax_over_labels_t1", "mass_outside", "pred_argmax"],
                "calibration": None,
                "add_generation_prompt": True,
                "chat_template_sha256": template_digest(self.chat_template),
                "chat_template_kwargs_sha256": kwargs_digest(hp.chat_template_kwargs),
                "assistant_prefill_sha256": template_digest(self.prefill) if self.prefill else None,
                "assistant_prefill_token_ids": self.prefill_ids,
                "vocab_size": self.vocab_size,
                "max_length": self.max_length,
                "max_length_source": self.max_length_source,
                "overflow_policy": "drop_and_count",
                "truncation": None,
                "padding_side": "right",
                "dtype": self.dtype,
                "tf32": self.tf32,
                "device": self.device,
                "batch_plan": batch_plan_record(batches, hp.scoring_batch_size),
                "list_separator": hp.list_separator,
                "id_column": hp.id_column,
                "slice_column": hp.slice_column,
                "unexpected_checkpoint_keys": self.unexpected_keys,
            },
        )


def _resolve_tokens(
    tokenizer: Any, label_tokens: Sequence[LabelToken]
) -> tuple[list[int], list[str]]:
    """Each label token's single id and its vocabulary piece, refusing anything else."""
    ids: list[int] = []
    for entry in label_tokens:
        encoded: Any = tokenizer(entry.token, add_special_tokens=False)["input_ids"]
        found = cast("list[int]", encoded)
        if len(found) != 1:
            msg = (
                f"the token for label {entry.label!r} ({entry.token!r}) is {len(found)} tokens "
                f"({found[:8]}), not one; declare a verbalizer that is a single token"
            )
            raise RunError(msg)
        ids.append(found[0])
    if len(set(ids)) != len(ids):
        msg = f"two labels share a token id ({ids}); each label needs a distinct token"
        raise RunError(msg)
    pieces: Any = tokenizer.convert_ids_to_tokens(ids)
    return ids, [str(piece) for piece in cast("list[object]", pieces)]


def load_label_reader(
    snapshot: Path,
    *,
    label_tokens: Sequence[LabelToken],
    prefill: str | None,
    chat_template_kwargs: Mapping[str, Any] | None,
    max_length: int | None,
    dtype: str,
) -> LabelReader:
    """Load the snapshot's tokenizer and causal LM, refusing a partial load or an unusable setup.

    Reads only the local snapshot the runner downloaded (``local_files_only``), with no remote
    code and safetensors weights only.
    """
    if dtype not in SCORING_DTYPES:
        msg = f"dtype must be one of {', '.join(SCORING_DTYPES)}"
        raise RunError(msg)
    try:
        torch: Any = import_module("torch")
        transformers: Any = import_module("transformers")
    except ImportError as exc:
        msg = (
            "label-logit scoring needs torch and transformers: pip install 'strata-forge[serving]'"
        )
        raise RunError(msg) from exc
    tf32 = tf32_state(torch, disable=dtype == "float32")
    location = str(snapshot)
    try:
        tokenizer: Any = transformers.AutoTokenizer.from_pretrained(
            location, local_files_only=True, **tokenizer_load_kwargs(False)
        )
    except (OSError, ValueError) as exc:
        msg = f"could not load the model's tokenizer: {exc}"
        raise RunError(msg) from exc
    try:
        template = cast("str", tokenizer.get_chat_template())
    except ValueError as exc:
        msg = f"the model has no usable chat template: {exc}"
        raise RunError(msg) from exc
    unread = unread_kwargs(chat_template_kwargs, template)
    if unread:
        msg = (
            f"the model's chat template never reads {unread}, so setting them would change "
            "nothing the model sees; remove them from chat_template_kwargs"
        )
        raise RunError(msg)
    token_ids, pieces = _resolve_tokens(tokenizer, label_tokens)
    try:
        model, info = transformers.AutoModelForCausalLM.from_pretrained(
            location,
            local_files_only=True,
            output_loading_info=True,
            dtype=getattr(torch, dtype),
            **model_load_kwargs(False),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        msg = f"could not load the model as a causal language model: {exc}"
        raise RunError(msg) from exc
    try:
        # No head of its own to excuse: every weight of a causal LM must come from the checkpoint.
        unexpected = check_loading_info(cast("Mapping[str, Any]", info), head_prefixes=())
    except LoadRefusedError as exc:
        raise RunError(str(exc)) from exc
    if "logits_to_keep" not in inspect.signature(model.forward).parameters:
        msg = (
            f"{type(model).__name__} cannot project selected positions only (no logits_to_keep), "
            "so a batch would materialise logits for every position; it is not supported"
        )
        raise RunError(msg)
    embeddings: Any = model.get_output_embeddings()
    vocab = int(embeddings.weight.shape[0])
    out_of_range = [token_id for token_id in token_ids if token_id >= vocab]
    if out_of_range:
        msg = f"label token ids {out_of_range} are outside the model's {vocab}-token output layer"
        raise RunError(msg)
    limit, source = length_limit(model.config, tokenizer, max_length)
    device = "cuda" if bool(torch.cuda.is_available()) else "cpu"
    model.to(device)
    model.eval()
    return LabelReader(
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        labels=[entry.label for entry in label_tokens],
        tokens=[entry.token for entry in label_tokens],
        token_ids=token_ids,
        token_pieces=pieces,
        prefill=prefill or "",
        chat_template=template,
        vocab_size=vocab,
        max_length=limit,
        max_length_source=source,
        dtype=str(model.dtype).removeprefix("torch."),
        tf32=tf32,
        device=device,
        unexpected_keys=unexpected,
    )
