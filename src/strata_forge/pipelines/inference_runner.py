"""VM-side batch-inference runner: STRATA_RUN_CONFIG (inert JSON) -> vLLM -> HF results.

Launched on the user's own VM by the control plane (``python -m
strata_forge.pipelines.inference_runner``). It reads an INERT run spec from the
``STRATA_RUN_CONFIG`` env var, loads a Hugging Face dataset split, renders one prompt
per row by simple ``{column}`` substitution (NO code execution), serves the model with
vLLM on the same VM, runs a concurrency-bounded batch through an openai-compatible
``LLMClient``, and then EITHER pushes ``{custom_id, output, ...}`` results to the user's HF
dataset repo (when a write token + an output repo are supplied) OR keeps them as a parquet
file on the VM (outside the per-run workdir, for the user to retrieve over SSH). Live
progress is appended to the file named by ``FORGE_PROGRESS_PATH`` (the orchestrator tails it).

``hyperparams.engine = "hf_classify"`` swaps the middle of that for exact classifier scoring: each
row renders into a text pair, a sequence-classification head is read with no model server, and
the raw logits are persisted (see :mod:`strata_forge.pipelines._scoring`).
``hyperparams.engine = "hf_label_logits"`` reads a causal LM instead: each row renders into a user
message, the model's chat template (with the run's declared kwargs) and an optional assistant
prefill make the prompt, and the logits of declared single-token labels at its last position are
persisted (see :mod:`strata_forge.pipelines._label_logits`). The frame around both — the spec, the
filtered model download, the private output repo, the drop-and-count overflow and the one-commit
push — is the same.

Security boundary (the VM is where untrusted-but-allow-listed config meets real
credentials + the network):

  - The config is parsed as DATA only: ``json`` + Pydantic ``extra="forbid"`` — never
    ``eval``/``pickle``/``yaml.unsafe_load``. ``template``/``column_mapping``/
    ``hyperparams`` are never executed.
  - Template rendering is a bounded, NON-executing ``{name}`` substitution (a regex, NOT
    ``str.format`` and NOT a template engine) — only placeholders that map to a real
    column are replaced; anything else stays literal.
  - The Hugging Face token arrives in a private secrets file the backend wrote beside the job
    (named by ``FORGE_SECRETS_FILE``, read and deleted before anything else runs — see
    :func:`~strata_forge.pipelines._common.load_secrets`), never in ``STRATA_RUN_CONFIG`` or
    the environment. It is passed EXPLICITLY to every Hub read and write — the split, the model
    download, the push — and a run that received none sends no credential rather than the
    VM's ambient ``HF_TOKEN`` or cached login (a repo the VM's Hub cache already holds still
    loads from that cache). It is scrubbed from any surfaced error.
  - The runner downloads the model ITSELF, before the model server starts, and only its
    servable files (:data:`SNAPSHOT_PATTERNS`: safetensors weights, configs, tokenizer and
    chat-template files — never a pickle checkpoint or a ``*.py`` file). The server then loads
    that local directory with ``HF_HUB_OFFLINE=1``, so it never holds the token and never
    fetches anything of its own, and with ``--load-format safetensors``, so a pickle checkpoint
    an earlier download left in the shared cache directory is never what it loads.
  - The model server gets an allow-listed environment
    (:func:`~strata_forge.pipelines._common.model_server_environ`), not the runner's: it is
    third-party code, and it has no use for the spec, the secrets file's path or a token.
  - Progress + result rows carry no secret: events hold step counts + float metrics + a
    repo id; result rows are ``{custom_id, output, reasoning, error, finish_reason, input_tokens,
    output_tokens}`` (model text and its accounting only), and the ``manifest.json`` beside them
    records the resolved model commit, the engine version and counts, never row text. The ``error``
    column, phase messages and the run's own error all go through the one run redactor: the
    column is pushed to the Hub with the results, an exception's text is not under this
    module's control, and neither is a phrase from ``serving_endpoint``'s public phase hook.

The exit code is the run's VERDICT, and the control plane reads it as such. Individual row
failures are collected rather than fatal (a few filtered rows must not discard thousands of good
generations) and reported as succeeded/failed counts, but a run that produced no usable row at
all exits nonzero: there is no reading under which it did its job, and the results file it
leaves behind holds only errors.

Long provisioning steps (downloading the split, installing and starting the model server,
writing and uploading results) have nothing to count, so each reports itself with a
``ProgressEvent(kind="phase")``. Without them a run is a single indeterminate wait between
``start`` and the first ``step`` — which is where a model server that never comes up spends
its entire timeout, invisibly.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import itertools
import json
import re
import shlex
import sys
import time
from collections import Counter, deque
from pathlib import Path
from statistics import median
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    SecretStr,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from strata_forge.compute import LocalBackend
from strata_forge.compute.batch import BatchInferenceRunner
from strata_forge.compute.serving import build_vllm_task, serving_endpoint
from strata_forge.llm import LLMClient, UserMessage
from strata_forge.llm.providers.config import OpenAICompatConfig
from strata_forge.llm.providers.openai_compat import (
    UNAUTHENTICATED_API_KEY,
    OpenAICompatProvider,
)
from strata_forge.pipelines._chat_template import kwargs_placeholders, kwargs_problem
from strata_forge.pipelines._common import (
    MANIFEST_FILENAME,
    MANIFEST_SCHEMA,
    RunError,
    RunSecrets,
    emit,
    load_config,
    manifest_engine,
    model_server_environ,
    phase_sink,
    require_commit_sha,
    results_dir,
    run_redactor,
    runner_main,
    snapshot_commit,
    ticking_phase,
    validate_repo_id,
    write_manifest,
)
from strata_forge.pipelines._label_logits import load_label_reader
from strata_forge.pipelines._scoring import (
    LABEL_RE,
    SCORING_DTYPES,
    ClassifierScorer,
    build_inputs,
    load_classifier,
    plan_batches,
    tally,
)
from strata_forge.storage import HFHubClient
from strata_forge.training.hardware import GpuSampler
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from strata_forge.core.redact import Redactor
    from strata_forge.llm.registry import ProviderName
    from strata_forge.pipelines._scoring import ScoringInputs

__all__ = [
    "SNAPSHOT_IGNORED",
    "SNAPSHOT_PATTERNS",
    "LabelToken",
    "RunSpec",
    "main",
    "render_template",
    "template_placeholders",
]

# vLLM serves on loopback only — the runner talks to it via localhost, and a multi-tenant
# / network-reachable VM must not expose the model server.
_SERVE_HOST = "127.0.0.1"
_SERVE_PORT = 8000
# A bare ``{name}`` placeholder only — no attribute/index access, no format mini-language.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
# Cap a rendered prompt so a pathological row can't blow up memory / the request.
_MAX_RENDERED_CHARS = 200_000
_RESULTS_FILENAME = "results.parquet"
_MANIFEST_KIND = "inference"
#: The installed distributions whose versions decide what a run computed. The model server and
#: its stack are installed fresh on the VM per run within a version range, so the exact versions
#: are only knowable here, after installation.
_RECORDED_DISTRIBUTIONS = ("vllm", "transformers", "torch", "datasets", "huggingface-hub")
# vLLM prints its resolved engine configuration once at start-up, ``dtype=torch.bfloat16`` among
# it. Only the head of each log is read: the line comes before any request is served.
_SERVED_DTYPE_RE = re.compile(r"\bdtype=torch\.([a-z0-9_]+)")
_SERVE_LOG_HEAD_BYTES = 2 * 1024 * 1024
_SERVE_LOGS = ("serve.stdout.log", "serve.stderr.log")
# Where results land on the VM when they are not pushed — see `_local_results_dir`.
_RESULTS_DIR_NAME = "strata-inference-results"
# How much of one row's error is quoted as the sample when EVERY row failed. Enough to name a
# provider/status/class, short enough that the run's message stays a message.
_MAX_SAMPLE_ERROR_CHARS = 500
# How many recent row latencies the p50 is taken over. Bounds both the memory and the sort.
_LATENCY_WINDOW = 1000
#: The files the runner downloads for the model server: safetensors weights, every JSON (configs,
#: a sharded checkpoint's index, tokenizer and generation settings), and the tokenizer and
#: chat-template formats that are not JSON.
#: ``tokenizer.model*`` rather than ``tokenizer*``: the versioned SentencePiece files some repos
#: ship (``tokenizer.model.v3``) need a prefix match, and every other tokenizer format is named by
#: its extension above, so the prefix need not admit whatever else a repo calls ``tokenizer.*``.
SNAPSHOT_PATTERNS = (
    "*.safetensors",
    "*.json",
    "tokenizer.model*",
    "*.model",
    "*.tiktoken",
    "*.jinja",
    "*.txt",
)
#: Refused even when a pattern above admits the name: pickle checkpoints and the array formats
#: that can carry one, which can run code when loaded; a repo's own Python, compiled modules and
#: shell scripts; and the ``original/`` checkpoints some repos carry beside the converted ones.
SNAPSHOT_IGNORED = (
    "*.py",
    "*.pyc",
    "*.so",
    "*.sh",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.pkl",
    "*.pickle",
    "*.ckpt",
    "*.joblib",
    "*.npy",
    "*.npz",
    "*.msgpack",
    "*.h5",
    "*.onnx",
    "original/*",
)
#: How the model server is told to read the weights. vLLM's default (``auto``) falls back to a
#: pickle checkpoint when the directory root holds no safetensors file, and the directory a
#: snapshot download returns is the SHARED cache folder for that commit: it also holds whatever an
#: earlier, unfiltered download of the same commit left there. Naming the format means the
#: download's filter is not the only thing standing between a ``*.bin`` and the server.
_SERVE_LOAD_FORMAT = ("--load-format", "safetensors")
#: The ``model_type`` a gpt-oss checkpoint declares. The model server renders and parses it in the
#: harmony format, whose analysis channel reaches the response's content unless a reasoning parser
#: splits it out.
_HARMONY_MODEL_TYPE = "gpt_oss"
_HARMONY_REASONING_PARSER = "openai_gptoss"
#: The ``model_type`` values the ``qwen3`` reasoning parser is for.
_QWEN3_MODEL_TYPES = frozenset({"qwen3", "qwen3_moe"})
# The harmony system message states a "Current date", which the model server takes from this
# variable and otherwise from its own clock, per request. Unpinned, the same prompts differ across
# days, and a long run can cross midnight.
_HARMONY_DATE_ENV = "VLLM_SYSTEM_START_DATE"
_ISO_DATE_RE = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
# The pair template is bounded like the prompt template the control plane accepts.
_MAX_PAIR_TEMPLATE = 8_192
# An assistant prefill is a short declared string (a judge's "<score>"), not a second prompt.
_MAX_PREFILL = 1_024
_MAX_LABEL_TOKEN = 64
# The knobs each scoring engine reads on top of the ones every scoring engine reads, and the ones
# every engine reads. A knob outside an engine's set is refused rather than ignored.
_CLASSIFY_FIELDS = frozenset({"pair_template", "label_alias"})
_LABEL_LOGITS_FIELDS = frozenset({"label_tokens", "assistant_prefill", "chat_template_kwargs"})
_SCORING_COMMON_FIELDS = frozenset(
    {"id_column", "slice_column", "list_separator", "scoring_batch_size"}
)
_SHARED_FIELDS = frozenset({"row_limit", "max_model_len", "dtype"})
_ENGINE_FIELDS: dict[str, frozenset[str]] = {
    "hf_classify": _CLASSIFY_FIELDS | _SCORING_COMMON_FIELDS | _SHARED_FIELDS,
    "hf_label_logits": _LABEL_LOGITS_FIELDS | _SCORING_COMMON_FIELDS | _SHARED_FIELDS,
}
# Knobs only a scoring engine reads: a chat run refuses them.
_SCORING_FIELDS = _CLASSIFY_FIELDS | _LABEL_LOGITS_FIELDS | _SCORING_COMMON_FIELDS


class LabelToken(BaseModel):
    """One declared label of a label-logits run and the single token that stands for it."""

    model_config = ConfigDict(extra="forbid")

    # The results column suffix (``logit_<label>``): the classifier engine's label grammar.
    label: str
    # The token's TEXT, leading space included where the vocabulary has one (``" True"``).
    token: str = Field(min_length=1, max_length=_MAX_LABEL_TOKEN)

    @field_validator("label")
    @classmethod
    def _label_names_a_column(cls, value: str) -> str:
        if not LABEL_RE.fullmatch(value):
            msg = "a label is 1-64 of A-Z a-z 0-9 _ . - and starts with a letter or digit"
            raise ValueError(msg)
        return value


class Hyperparams(BaseModel):
    """Inert tuning knobs (sampling + serving + bounds). Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = None
    # Sent to the model server with every request. A per-request seed makes each row's sampling
    # reproducible on the same engine and hardware; it does not make two engines agree.
    # Strict: JSON ``true`` is not a seed. Bounded to int64, the model server's own range, so an
    # out-of-range value is refused at spec load rather than by every request after start-up.
    seed: StrictInt | None = Field(default=None, ge=0, le=2**63 - 1)
    top_k: StrictInt | None = Field(default=None, ge=1)
    concurrency: int = Field(default=8, ge=1, le=256)
    progress_chunk: int = Field(default=200, ge=1)
    row_limit: int | None = Field(default=None, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    max_model_len: int | None = Field(default=None, ge=1)
    dtype: str | None = None
    # The share of GPU memory the model server may claim, and its per-step token budget. vLLM's
    # defaults run gpt-oss-120b out of memory on a single 80 GB GPU; its recipe lowers the budget
    # or raises the share. ``None`` keeps the server's default.
    gpu_memory_utilization: StrictFloat | None = Field(default=None, gt=0, le=1)
    max_num_batched_tokens: StrictInt | None = Field(default=None, ge=1)
    # The model server's reasoning parser: it moves the model's reasoning out of the answer into
    # the response's own reasoning field, which the runner stores in the ``reasoning`` column.
    # An allow-list because it becomes a command-line argument. Checked against the checkpoint's
    # ``config.json`` before anything large is downloaded (see `_effective_hyperparams`): unset on
    # a gpt-oss checkpoint it defaults to ``openai_gptoss``, and a parser for another
    # architecture is refused.
    reasoning_parser: Literal["openai_gptoss", "qwen3"] | None = None
    # gpt-oss's harmony reasoning effort, sent with every request.
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    # The date the harmony system message states, ``YYYY-MM-DD``. Unset, a gpt-oss run pins the
    # serving machine's date at start-up. Either way the manifest records the date used.
    harmony_date: str | None = None
    wait_timeout_s: float = Field(default=1800.0, gt=0)
    # What the run does with each rendered row: generate a reply through the model server
    # (`vllm_chat`), read a sequence-classification head's logits for the pair
    # (`template`, `pair_template`) with no model server (`hf_classify`, see `_scoring`), or read
    # a causal LM's next-token logits for declared label tokens after its chat template
    # (`hf_label_logits`, see `_label_logits`).
    engine: Literal["vllm_chat", "hf_classify", "hf_label_logits"] = "vllm_chat"
    pair_template: str | None = Field(default=None, min_length=1, max_length=_MAX_PAIR_TEMPLATE)
    # Required for scoring: every model class -> its output label, or null to exclude it.
    label_alias: dict[str, str | None] | None = Field(default=None, min_length=2, max_length=64)
    id_column: str | None = Field(default=None, min_length=1, max_length=255)
    slice_column: str | None = Field(default=None, min_length=1, max_length=255)
    list_separator: str = Field(default="\n", max_length=16)
    scoring_batch_size: int = Field(default=16, ge=1, le=1024)
    # `hf_label_logits`: the labels, in order, each with the one token whose logit it reads.
    label_tokens: list[LabelToken] | None = Field(default=None, min_length=2, max_length=64)
    # Text appended after the chat template's assistant header, before the label position. Not a
    # template: no placeholder in it is substituted.
    assistant_prefill: str | None = Field(default=None, min_length=1, max_length=_MAX_PREFILL)
    # Variables the model's chat template reads (`enable_thinking`, `documents`, a judge's
    # config). A string anywhere in a value may hold `{name}` placeholders, rendered per row.
    chat_template_kwargs: dict[str, JsonValue] | None = None

    @model_validator(mode="after")
    def _fields_match_the_engine(self) -> Hyperparams:
        """Refuse a knob the chosen engine would ignore, so a spec means what it says."""
        given = self.model_fields_set - {"engine"}
        if self.engine == "vllm_chat":
            stray = sorted(given & _SCORING_FIELDS)
            if stray:
                raise ValueError(_owners(stray))
            return self
        accepted = _ENGINE_FIELDS[self.engine]
        foreign = sorted(given & (_SCORING_FIELDS - accepted))
        if foreign:
            msg = f"engine {self.engine!r} takes no {', '.join(foreign)}: " + _owners(foreign)
            raise ValueError(msg)
        stray = sorted(given - accepted - _SCORING_FIELDS)
        if stray:
            msg = f"engine {self.engine!r} generates no text and takes no {', '.join(stray)}"
            raise ValueError(msg)
        if self.dtype not in SCORING_DTYPES:
            msg = f"engine {self.engine!r} needs dtype, one of {', '.join(SCORING_DTYPES)}"
            raise ValueError(msg)
        if self.engine == "hf_label_logits":
            self._check_label_logits()
            return self
        if self.pair_template is None:
            msg = "engine 'hf_classify' needs pair_template, the second text of each pair"
            raise ValueError(msg)
        if self.label_alias is None:
            msg = (
                "engine 'hf_classify' needs label_alias: every model class mapped to an output "
                "label, or to null to exclude it"
            )
            raise ValueError(msg)
        included = [v for v in self.label_alias.values() if v is not None]
        bad = [
            k
            for k, v in self.label_alias.items()
            if not (0 < len(k) <= 256 and (v is None or LABEL_RE.fullmatch(v)))
        ]
        if bad or len(set(included)) != len(included) or len(included) < 2:
            msg = (
                "label_alias maps each model class name to a distinct label (1-64 of A-Z a-z "
                "0-9 _ . -) or to null to exclude it, leaving at least two classes included; "
                "two classes cannot share a label"
            )
            raise ValueError(msg)
        return self

    def _check_label_logits(self) -> None:
        if self.label_tokens is None:
            msg = "engine 'hf_label_logits' needs label_tokens, the labels and the token of each"
            raise ValueError(msg)
        labels = [entry.label for entry in self.label_tokens]
        tokens = [entry.token for entry in self.label_tokens]
        if len(set(labels)) != len(labels) or len(set(tokens)) != len(tokens):
            msg = "label_tokens needs a distinct label and a distinct token for every entry"
            raise ValueError(msg)
        if self.chat_template_kwargs is not None:
            problem = kwargs_problem(self.chat_template_kwargs)
            if problem:
                raise ValueError(problem)

    @field_validator("harmony_date")
    @classmethod
    def _harmony_date_is_a_calendar_date(cls, value: str | None) -> str | None:
        # The pattern first: `date.fromisoformat` also takes `20261118` and week dates, and the
        # value is recorded as given, so only the one spelling is accepted.
        if value is not None:
            try:
                if not _ISO_DATE_RE.fullmatch(value):
                    raise ValueError
                datetime.date.fromisoformat(value)
            except ValueError:
                msg = "harmony_date must be a calendar date written YYYY-MM-DD"
                raise ValueError(msg) from None
        return value

    @model_validator(mode="after")
    def _harmony_settings_need_the_harmony_parser(self) -> Hyperparams:
        # Only gpt-oss reads an effort level or states a date. The model server would accept an
        # effort for any other model and ignore it (or read it as "think"), and a run that records
        # a setting its model never saw is a false record. An unset parser may still turn out to be
        # gpt-oss's, so that is decided against the checkpoint (`_effective_hyperparams`).
        if self.reasoning_parser not in (None, _HARMONY_REASONING_PARSER):
            for name in ("reasoning_effort", "harmony_date"):
                if getattr(self, name) is not None:
                    msg = (
                        f"{name} applies only to gpt-oss models, served with "
                        f"reasoning_parser={_HARMONY_REASONING_PARSER!r}"
                    )
                    raise ValueError(msg)
        return self


def _owners(fields: list[str]) -> str:
    """Which engines read each of ``fields``, as one sentence per group of fields."""
    groups: dict[tuple[str, ...], list[str]] = {}
    for field in fields:
        engines = tuple(e for e, accepted in _ENGINE_FIELDS.items() if field in accepted)
        groups.setdefault(engines, []).append(field)
    return "; ".join(
        f"{', '.join(names)} apply only to engine {' or '.join(repr(e) for e in engines)}"
        for engines, names in groups.items()
    )


class RunSpec(BaseModel):
    """The inert run spec the server delivers in ``STRATA_RUN_CONFIG`` (no secrets)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    # The engine the control plane validated this spec against: ``"<version>"`` or
    # ``"<version>+<commit>"``. ``load_config`` refuses any other installed engine (``None`` makes
    # no claim). The control plane sets it, never a user: the same string decides what the setup
    # step installs, so a spec cannot pick an engine the control plane did not validate against.
    engine_version: str | None = None
    model_id: str
    # The model's full Hub commit id. Set, it is the only revision the run downloads; unset, the
    # run takes the repo's default branch as it is at download time, and records what that was.
    model_revision: str | None = None
    dataset_id: str
    dataset_commit_sha: str | None = None
    dataset_config: str | None = None
    split: str
    column_mapping: dict[str, str] = Field(default_factory=dict)
    template: str
    hyperparams: Hyperparams = Field(default_factory=Hyperparams)
    # Where to push results. The server populates it ONLY when pushing (a write token is present);
    # when absent the runner keeps results on the VM instead of pushing. The VM never guesses ownership.
    output_repo_id: str | None = None
    progress_path: str | None = None
    # The control plane's run id — names the on-VM results dir when NOT pushing to the Hub.
    run_id: str | None = None


def load_spec(*, writer: JsonlProgressWriter | None = None) -> RunSpec:
    spec = load_config(RunSpec, writer=writer)
    validate_repo_id(spec.model_id, "model")
    validate_repo_id(spec.dataset_id, "dataset")
    require_commit_sha(spec.model_revision, "model_revision")
    if spec.output_repo_id is not None:
        validate_repo_id(spec.output_repo_id, "output repo")
    _validate_template_coverage(spec)
    return spec


def _validate_template_coverage(spec: RunSpec) -> None:
    """Reject a template whose ``{placeholder}`` has no entry in ``column_mapping``.

    An unmapped placeholder renders LITERALLY (see ``render_template``), which is the right
    behaviour for the primitive but a catastrophe for a run: every row gets the byte-identical,
    row-independent prompt, the model answers it N times, every row SUCCEEDS, and the run reports
    `succeeded` with N copies of an answer to the literal text ``{question}``. Nothing downstream
    can notice — the results file records the output and its accounting per row, so neither the rendered
    prompt nor the source row is in it.

    The one check that existed validated the mapping's VALUES against the split's columns, which
    passes in exactly this case: with ``{"q": "question"}`` against a template of ``{question}``,
    the column ``question`` really does exist. It is the KEYS that fail to cover the template, and
    nobody was looking at them.

    That mistake is easy to make and expensive to discover, so it fails here — before the dataset
    download, before the GPU, within seconds of launch — naming both what is missing and what is
    available.

    The trade-off is deliberate: a template can no longer carry a LITERAL ``{word}`` that is meant
    to survive to the model. That reading is rare, and it is not worth the run this protects.
    """
    unmapped = sorted(
        template_placeholders(spec.template, spec.hyperparams) - set(spec.column_mapping)
    )
    if not unmapped:
        return
    known = sorted(spec.column_mapping) or ["(none)"]
    msg = (
        f"template placeholders have no column_mapping entry: {unmapped}. "
        f"Mapped placeholders: {known}. "
        "A placeholder is named WITHOUT braces on the left of the mapping "
        '(template "{question}" needs the entry "question" -> the column name); '
        "an unmapped one would be sent to the model literally, identically for every row."
    )
    raise RunError(msg)


def template_placeholders(template: str, hyperparams: Hyperparams) -> set[str]:
    """Every ``{name}`` a run renders per row: in the template, the pair template and the string
    values of the chat-template kwargs. A control plane checks the mapping covers them with this.
    """
    texts = (template, hyperparams.pair_template or "")
    found = {m.group(1) for text in texts for m in _PLACEHOLDER_RE.finditer(text)}
    return found | kwargs_placeholders(hyperparams.chat_template_kwargs)


def render_template(template: str, row: dict[str, Any], column_mapping: dict[str, str]) -> str:
    """Render ``template`` by replacing each ``{name}`` whose ``column_mapping[name]`` is a
    column present in ``row`` with ``str(row[col])``. Unknown placeholders stay literal.

    Executes nothing: a regex substitution, deliberately NOT ``str.format`` (which reaches
    attributes/indices + the format mini-language) and NOT a template engine. Capped.
    """

    def _sub(match: re.Match[str]) -> str:
        col = column_mapping.get(match.group(1))
        if col is None or col not in row:
            return match.group(0)  # leave an unmapped/missing placeholder literal
        return str(row[col])

    rendered = _PLACEHOLDER_RE.sub(_sub, template)
    return rendered[:_MAX_RENDERED_CHARS]


def _load_rows(spec: RunSpec, token: str | Literal[False]) -> list[dict[str, Any]]:
    try:
        # Lazy optional-extra import (forge convention: `Any` so pyright skips the unresolved
        # module); the [hf] extra ships `datasets`.
        datasets_mod: Any = __import__("datasets")
    except ImportError as exc:
        msg = "the [hf] extra is required to load datasets: pip install 'strata-forge[hf]'"
        raise RunError(msg) from exc
    dataset = datasets_mod.load_dataset(
        spec.dataset_id,
        name=spec.dataset_config,
        split=spec.split,
        revision=spec.dataset_commit_sha,
        token=token,
        streaming=False,
    )
    columns = set(dataset.column_names or [])
    missing = {c for c in spec.column_mapping.values() if c not in columns}
    if missing:
        msg = f"column_mapping references columns not in the split: {sorted(missing)}"
        raise RunError(msg)
    # Slice DURING iteration (not after): dataset size/content is attacker-controlled, so
    # row_limit must bound materialization — a huge split mustn't OOM the VM before the cap.
    limit = spec.hyperparams.row_limit
    source = itertools.islice(dataset, limit) if limit is not None else dataset
    return [dict(r) for r in source]


def _build_requests(
    spec: RunSpec, rows: list[dict[str, Any]]
) -> tuple[list[list[UserMessage]], list[str]]:
    prompts: list[list[UserMessage]] = []
    custom_ids: list[str] = []
    for i, row in enumerate(rows):
        rendered = render_template(spec.template, row, spec.column_mapping)
        prompts.append([UserMessage(content=rendered)])  # one-message conversation
        custom_ids.append(f"row-{i}")  # positional reconciliation key
    return prompts, custom_ids


async def _run_batches(
    spec: RunSpec,
    client: LLMClient,
    prompts: list[list[UserMessage]],
    custom_ids: list[str],
    writer: JsonlProgressWriter | None,
    is_alive: Callable[[], Awaitable[bool]] | None = None,
    gpu: GpuSampler | None = None,
    *,
    redactor: Redactor,
) -> list[dict[str, Any]]:
    """Run the prompts in progress-chunked batches; reconcile results positionally.

    ``is_alive`` is checked between chunks. The model server is verified once before the batch
    starts and then never again, but it can die at any point after that — an OOM on a long prompt,
    a CUDA fault. Every row from then on fails against a socket nobody is listening on, and the
    client RETRIES each one, so a dead server turns into a long expensive silence instead of an
    error: the rest of the run is spent timing out one row at a time, and the failure that
    eventually surfaces describes a connection, not the crash that caused it.

    A failed row's ``error`` is redacted before it is stored: the column is written into the
    results and pushed to the Hub, and an exception's text can quote whatever the failing call
    held. ``redactor`` is required so a caller cannot forget the run's token: a shapes-only
    ``Redactor()`` misses its encoded forms.
    """
    hp = spec.hyperparams
    runner = BatchInferenceRunner(client, concurrency=hp.concurrency, on_error="collect")
    total = len(prompts)
    out: list[dict[str, Any]] = []
    ok = failed = 0
    # Throughput is measured from the first chunk, not from process start: everything before this
    # point is dataset loading and engine warmup, and folding minutes of that into the denominator
    # would report a rows/s the run never actually ran at.
    started = time.monotonic()
    # Bounded, and a ring rather than a list for two reasons. `statistics.median` copies and
    # sorts everything it is given, so an unbounded history makes the per-chunk cost quadratic
    # in the row count -- and `progress_chunk` can be as low as 1 with no `row_limit`, which
    # would put a full sort per row on the event loop. A median over the most recent rows is
    # also the more useful number: it tracks how the run is behaving NOW.
    latencies_ms: deque[float] = deque(maxlen=_LATENCY_WINDOW)
    output_tokens = 0
    for start in range(0, total, hp.progress_chunk):
        # Between chunks, not between rows: the check costs a remote status probe, and a chunk is
        # the granularity the run already reports at. It bounds the waste at one chunk rather than
        # the whole remaining batch.
        if start and is_alive is not None and not await is_alive():
            msg = (
                f"the model server died after {len(out)} of {total} rows "
                "(its log is beside the results on the VM)"
            )
            raise RunError(msg)
        chunk = prompts[start : start + hp.progress_chunk]
        ids = custom_ids[start : start + hp.progress_chunk]
        results = await runner.run(
            chunk,
            temperature=hp.temperature,
            max_tokens=hp.max_tokens,
            top_p=hp.top_p,
            provider_extras=_sampling_extras(hp),
        )
        for cid, result in zip(ids, results, strict=True):
            if result.succeeded and result.response is not None:
                response = result.response
                out.append(
                    {
                        "custom_id": cid,
                        # The final answer only: the model server's reasoning parser keeps the
                        # reasoning out of `text`, and it lands in its own column.
                        "output": response.text,
                        "reasoning": response.reasoning,
                        "error": None,
                        "finish_reason": response.finish_reason,
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    }
                )
                ok += 1
                # Both already measured per request by the client — a true per-row latency and a
                # real token count, not an average reconstructed from the chunk's wall time.
                latencies_ms.append(result.response.latency_ms)
                output_tokens += result.response.usage.output_tokens
            else:
                error = redactor.redact(repr(result.error))
                out.append(
                    {
                        "custom_id": cid,
                        "output": None,
                        "reasoning": None,
                        "error": error,
                        "finish_reason": None,
                        "input_tokens": None,
                        "output_tokens": None,
                    }
                )
                failed += 1
        emit(
            writer,
            ProgressEvent(
                kind="step",
                stage="run",
                step=len(out),
                total_steps=total,
                metrics={
                    "succeeded": float(ok),
                    "failed": float(failed),
                    **_throughput(started, len(out), output_tokens, latencies_ms),
                    **(gpu.sample() if gpu is not None else {}),
                },
            ),
        )
    return out


def _sampling_extras(hp: Hyperparams) -> dict[ProviderName, dict[str, Any]] | None:
    """The sampling settings the chat-completions call has no parameter for.

    ``seed`` is an OpenAI-compatible request field. ``top_k`` is not an OpenAI parameter, so it
    rides in ``extra_body``, the transport's explicit channel for server-specific body fields,
    rather than depending on how the transport treats an unrecognised keyword. So does
    ``reasoning_effort``: the transport would otherwise decide per model id whether it supports
    the field, and an operator's model id is one it has never seen.
    """
    extras: dict[str, Any] = {}
    if hp.seed is not None:
        extras["seed"] = hp.seed
    body: dict[str, Any] = {}
    if hp.top_k is not None:
        body["top_k"] = hp.top_k
    if hp.reasoning_effort is not None:
        body["reasoning_effort"] = hp.reasoning_effort
    if body:
        extras["extra_body"] = body
    return {"openai_compat": extras} if extras else None


def _serve_args(hp: Hyperparams) -> tuple[str, ...]:
    """The model-server flags the runner adds beyond the model, its address and its size."""
    args: list[str] = list(_SERVE_LOAD_FORMAT)
    if hp.reasoning_parser is not None:
        args += ["--reasoning-parser", hp.reasoning_parser]
    if hp.gpu_memory_utilization is not None:
        args += ["--gpu-memory-utilization", str(hp.gpu_memory_utilization)]
    if hp.max_num_batched_tokens is not None:
        args += ["--max-num-batched-tokens", str(hp.max_num_batched_tokens)]
    return tuple(args)


def _today() -> str:
    """This machine's local date, the clock the model server would read."""
    return datetime.datetime.now().astimezone().date().isoformat()


def _harmony_date(hp: Hyperparams) -> str | None:
    """The date a gpt-oss run's system message states: the requested one, else today's.

    Today's on this machine's clock, which is the one the model server would read, but taken once
    so that every row of the run states the same date. ``None`` for a model without harmony.
    """
    if hp.reasoning_parser != _HARMONY_REASONING_PARSER:
        return None
    return hp.harmony_date or _today()


def _serve_flags(run: str) -> list[str]:
    """The flags of the model server's command line, from ``--host`` on.

    What precedes them is the interpreter and the local snapshot path, which describe this
    machine rather than the run; the manifest already records the model and its commit. A command
    line without ``--host`` records no flags rather than all of it.
    """
    argv = shlex.split(run)
    return argv[argv.index("--host") :] if "--host" in argv else []


def _throughput(
    started: float, rows_done: int, output_tokens: int, latencies_ms: deque[float]
) -> dict[str, float]:
    """Rate and latency counters for the rows finished so far.

    Rates are cumulative rather than per-chunk: a chunk is small enough that its own wall time is
    mostly noise from whichever row happened to be slowest, and a gauge that swings on that reads
    as broken. ``latency_p50_ms`` is a real median over the most recent rows — the tail is
    what makes a mean useless here, and the median is what survives it.
    """
    elapsed = time.monotonic() - started
    metrics: dict[str, float] = {}
    if elapsed > 0:
        metrics["rows_per_s"] = rows_done / elapsed
        metrics["tokens_per_s"] = output_tokens / elapsed
    if latencies_ms:
        metrics["latency_p50_ms"] = median(latencies_ms)
    return metrics


def _check_produced_output(rows: list[dict[str, Any]], ok: int) -> None:
    """Fail a run that reached the end without producing a single usable row.

    Row failures are COLLECTED rather than fatal on purpose: a handful of rows tripping a
    content filter must not throw away thousands of good generations. But that must not decide
    the run's VERDICT, which the control plane reads from this process's exit code. A run whose
    every row failed produced nothing, and calling it succeeded tells the user their results are
    ready when the file holds only errors — a staging run reported `succeeded` with exit 0 over
    2098 failed rows and zero generations, and nothing on the run said otherwise.

    A partial failure is deliberately still a success: it produced usable output, and the
    succeeded/failed counts on the run describe it. Only "nothing at all" is a failure, because
    only that has no reading under which the run did its job.

    The results file is written and pushed BEFORE this runs, so the failure keeps its evidence:
    the per-row `error` column is the record of what went wrong, for every row.
    """
    if ok:
        return
    if not rows:
        msg = "the run produced no rows: the dataset split was empty"
        raise RunError(msg)
    sample = next((str(row["error"]) for row in rows if row["error"]), "")
    msg = f"all {len(rows)} rows failed. First error: {sample[:_MAX_SAMPLE_ERROR_CHARS]}"
    raise RunError(msg)


def _local_results_dir(run_id: str | None) -> Path:
    """Where this run's parquet lands on the VM — see :func:`strata_forge.pipelines._common.results_dir`."""
    return results_dir(run_id, name=_RESULTS_DIR_NAME)


def _write_results(rows: list[dict[str, Any]], outdir: Path) -> Path:
    """Write the results as parquet (renders in the HF Datasets Viewer)."""
    datasets_mod: Any = __import__("datasets")  # already required by _load_rows ([hf] extra)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / _RESULTS_FILENAME
    datasets_mod.Dataset.from_list(rows).to_parquet(str(path))
    return path


def _require_safetensors(snapshot: Path, model_id: str) -> None:
    # The directory ROOT, as the model server reads it: weights in a subfolder are not what it
    # would load, so a repo whose only safetensors sit in one has none that this run can serve.
    if not any(snapshot.glob("*.safetensors")):
        msg = (
            f"the model {model_id} has no safetensors weights; only safetensors checkpoints are "
            "served, because a pickle checkpoint (*.bin, *.pt) can run code when it is loaded"
        )
        raise RunError(msg)


async def _fetch_model(
    spec: RunSpec, token: str | Literal[False], revision: str | None = None
) -> Path:
    """Download the model's servable files with the run's credential; return the local snapshot.

    The model server is pointed at this directory with the Hub switched off, so what it can load
    is exactly what this download admitted, and the token stays in this process. A pinned
    ``model_revision`` is the only revision downloaded.
    """
    hub = HFHubClient(token=token)  # explicit: never forge settings or the VM's own login
    try:
        snapshot = await hub.download_snapshot(
            spec.model_id,
            revision=revision or spec.model_revision,
            allow_patterns=SNAPSHOT_PATTERNS,
            ignore_patterns=SNAPSHOT_IGNORED,
        )
    except Exception as exc:
        msg = f"could not download the model {spec.model_id}: {exc}"
        if token is False:
            msg += (
                " (the run carries no Hugging Face token, so a gated or private model is"
                " unreadable)"
            )
        raise RunError(msg) from exc
    await asyncio.to_thread(_require_safetensors, snapshot, spec.model_id)
    return snapshot


async def _served_max_model_len(base_url: str, model_id: str) -> int | None:
    """The context length the model server actually admitted, as its ``/models`` route reports it.

    The server may clamp or derive the value it serves, so the requested ``max_model_len`` is not
    evidence of it. Best effort: an answer without the field records ``None``, never a failure.
    """
    import httpx  # already a core dep via litellm

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
            response.raise_for_status()
            payload: Any = response.json()  # pyright: ignore[reportAny] - untyped JSON from the server
    except httpx.HTTPError, ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    entries = cast("dict[str, Any]", payload).get("data")
    if not isinstance(entries, list):
        return None
    for raw in cast("list[Any]", entries):
        if not isinstance(raw, dict):
            continue
        entry = cast("dict[str, Any]", raw)
        if entry.get("id") == model_id:
            value = entry.get("max_model_len")
            return value if isinstance(value, int) and not isinstance(value, bool) else None
    return None


def _served_dtype(log_dir: Path) -> str | None:
    """The dtype the model server resolved, read from its own start-up configuration line.

    ``auto`` (or no ``dtype`` at all) leaves the choice to the server, which can differ from the
    checkpoint's own dtype; its log is the one place that states what it chose. ``None`` when no
    such line is found.
    """
    for name in _SERVE_LOGS:
        path = log_dir / name
        try:
            with path.open("rb") as handle:
                head = handle.read(_SERVE_LOG_HEAD_BYTES).decode("utf-8", errors="replace")
        except OSError:
            continue
        match = _SERVED_DTYPE_RE.search(head)
        if match:
            return match.group(1)
    return None


def _checkpoint_config(snapshot: Path) -> dict[str, Any]:
    """The checkpoint's ``config.json``, or an empty mapping when it is missing or unreadable."""
    try:
        config: Any = json.loads((snapshot / "config.json").read_text())  # pyright: ignore[reportAny] - untyped JSON
    except OSError, ValueError:
        return {}
    return cast("dict[str, Any]", config) if isinstance(config, dict) else {}


def _checkpoint_dtype(snapshot: Path) -> str | None:
    """The dtype the checkpoint's ``config.json`` declares (``torch_dtype`` or ``dtype``)."""
    typed = _checkpoint_config(snapshot)
    value: Any = typed.get("torch_dtype") or typed.get("dtype")
    return value if isinstance(value, str) else None


async def _read_model_config(
    spec: RunSpec, token: str | Literal[False]
) -> tuple[dict[str, Any], str | None]:
    """The model's ``config.json`` alone, and the commit it was read at.

    One small file, read before the weights so that a parser that does not fit the architecture
    fails within seconds rather than after a download of tens of gigabytes. The commit is what an
    unpinned run then downloads, so the weights come from the same revision the check read.
    """
    hub = HFHubClient(token=token)  # explicit: never forge settings or the VM's own login
    try:
        path = await hub.download_file(spec.model_id, "config.json", revision=spec.model_revision)
    except Exception as exc:
        msg = f"could not read the config of the model {spec.model_id}: {exc}"
        if token is False:
            msg += (
                " (the run carries no Hugging Face token, so a gated or private model is"
                " unreadable)"
            )
        raise RunError(msg) from exc
    config = await asyncio.to_thread(_checkpoint_config, path.parent)
    return config, spec.model_revision or snapshot_commit(path.parent)


def _effective_hyperparams(hp: Hyperparams, config: dict[str, Any]) -> Hyperparams:
    """The hyperparameters as served, with the reasoning parser settled against ``model_type``.

    A gpt-oss checkpoint is always served with ``openai_gptoss``: without it the model server
    returns the analysis channel inside every answer. So an unset parser defaults to it, and
    any other parser is refused. ``openai_gptoss`` on another architecture has no harmony output
    to split, and ``qwen3`` is for Qwen3 checkpoints only, so both are refused, as are the
    harmony-only settings on a model without harmony. The model id says nothing reliable about
    the architecture; the checkpoint's own ``model_type`` does.
    """
    model_type = config.get("model_type")
    if model_type == _HARMONY_MODEL_TYPE:
        if hp.reasoning_parser not in (None, _HARMONY_REASONING_PARSER):
            msg = (
                f"a gpt-oss model is served with reasoning_parser="
                f"{_HARMONY_REASONING_PARSER!r}, not {hp.reasoning_parser!r}"
            )
            raise RunError(msg)
        return hp.model_copy(update={"reasoning_parser": _HARMONY_REASONING_PARSER})
    if hp.reasoning_parser == _HARMONY_REASONING_PARSER:
        msg = (
            f"reasoning_parser={_HARMONY_REASONING_PARSER!r} is for gpt-oss models; this model "
            f"declares model_type={model_type!r}"
        )
        raise RunError(msg)
    if hp.reasoning_parser == "qwen3" and model_type not in _QWEN3_MODEL_TYPES:
        msg = (
            f"reasoning_parser='qwen3' is for Qwen3 models; this model declares "
            f"model_type={model_type!r}"
        )
        raise RunError(msg)
    for name in ("reasoning_effort", "harmony_date"):
        if getattr(hp, name) is not None:
            msg = f"{name} applies only to gpt-oss models; this model declares model_type={model_type!r}"
            raise RunError(msg)
    return hp


def _build_manifest(
    spec: RunSpec,
    rows: list[dict[str, Any]],
    model_commit: str | None,
    serving: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What a run needs to be reproduced and audited, with no row content and no secret.

    The template is recorded as a digest: the spec that holds its text is stored by the control
    plane, and a pushed manifest should not republish prompt text beside the outputs.
    """
    finish = Counter(str(r["finish_reason"]) for r in rows if r["error"] is None)
    ok = sum(1 for r in rows if r["error"] is None)
    return {
        "schema": MANIFEST_SCHEMA,
        "kind": _MANIFEST_KIND,
        "engine": manifest_engine(spec.engine_version, _RECORDED_DISTRIBUTIONS),
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
        },
        # What the model server actually ran with, next to what was asked for in `hyperparams`.
        "serving": serving
        or {
            "max_model_len": None,
            "dtype": None,
            "flags": None,
            "reasoning_parser": None,
            "harmony_date": None,
        },
        "template_sha256": hashlib.sha256(spec.template.encode()).hexdigest(),
        "column_mapping": spec.column_mapping,
        "hyperparams": spec.hyperparams.model_dump(),
        "counts": {
            "rows": len(rows),
            "succeeded": ok,
            "failed": len(rows) - ok,
            "finish_reasons": dict(sorted(finish.items())),
            "input_tokens": sum(int(r["input_tokens"] or 0) for r in rows),
            "output_tokens": sum(int(r["output_tokens"] or 0) for r in rows),
        },
    }


async def _ensure_private_output(hub: HFHubClient, out_repo: str) -> None:
    """Create the output repo private if it is missing, and refuse one that is public.

    ``create_repo(private=True, exist_ok=True)`` leaves an existing repo's visibility alone, and
    a run's outputs and manifest name the model and dataset it read, which may be private.
    """
    try:
        await hub.create_repo(out_repo, repo_type="dataset", private=True, exist_ok=True)
        private = await hub.is_private(out_repo, repo_type="dataset")
    except Exception as exc:
        msg = f"could not prepare the output repo {out_repo}: {exc}"
        raise RunError(msg) from exc
    if not private:
        msg = (
            f"the output repo {out_repo} is public; results are pushed only to a private repo. "
            "Make it private or choose another output repo"
        )
        raise RunError(msg)


async def _check_output_repo(spec: RunSpec, hf_token: str | None) -> None:
    """Refuse a public output repo before any GPU time is spent, not after the run.

    It creates the repo (private) when missing, so a run that then fails leaves an empty private
    repo behind: cheaper than discovering a public one after the generations are done.
    """
    if not (hf_token and spec.output_repo_id):
        return
    out_repo = validate_repo_id(spec.output_repo_id, "output repo")
    await _ensure_private_output(HFHubClient(token=hf_token), out_repo)


async def _push_results(spec: RunSpec, results_path: Path, hf_token: str) -> str:
    """Push the results and the manifest beside them in ONE Hub commit.

    One commit, because two can half-land: a results file next to an earlier run's manifest is a
    false provenance record, worse than none.
    """
    if not spec.output_repo_id:
        msg = "output_repo_id is required to push results"
        raise RunError(msg)
    out_repo = validate_repo_id(spec.output_repo_id, "output repo")
    hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
    # Re-checked at push time: visibility can change during a long run.
    await _ensure_private_output(hub, out_repo)
    await hub.upload_files(
        {
            _RESULTS_FILENAME: results_path,
            MANIFEST_FILENAME: results_path.parent / MANIFEST_FILENAME,
        },
        out_repo,
        repo_type="dataset",
        commit_message="batch inference results",
    )
    return out_repo


async def _execute(spec: RunSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None) -> str:
    if spec.hyperparams.engine != "vllm_chat":
        return await _execute_scoring(spec, secrets, writer)
    hf_token = secrets.hf_token_value()
    hub_token = secrets.hub_credential()
    # One sampler for the run: the phase sink folds its counters into every caption, and the
    # step events below reuse the same cached reading rather than shelling out twice.
    gpu = GpuSampler()
    phase = phase_sink(writer, hf_token, gpu=gpu)

    if hf_token and spec.output_repo_id:
        async with ticking_phase(phase, "Checking the output repo", stage="load_model"):
            await _check_output_repo(spec, hf_token)
    # The split download is unbounded by row_limit (that only slices during iteration), so it
    # runs BEFORE the first countable milestone and can take minutes on its own.
    # to_thread, not a direct call: `load_dataset` downloads and materialises the split, and a
    # blocked event loop cannot tick the caption that says it is still going.
    # The model's config before anything large: a reasoning parser that does not fit the
    # architecture is refused here, and an unset one is settled, in seconds.
    config, config_commit = await _read_model_config(spec, hub_token)
    hp = _effective_hyperparams(spec.hyperparams, config)
    async with ticking_phase(phase, "Loading the dataset", stage="load_model"):
        rows = await asyncio.to_thread(_load_rows, spec, hub_token)
    prompts, custom_ids = _build_requests(spec, rows)
    # `start` stays exactly here: it is the documented milestone that says inference is about
    # to happen, and the orchestrator reads it as such. Phases fill the silence around it.
    emit(writer, ProgressEvent(kind="start", stage="run", total_steps=len(prompts)))

    # Before the server starts, and in this process: the download is the one step that needs the
    # token, and the server that follows is third-party code that must not hold it.
    async with ticking_phase(phase, "Downloading the model", stage="load_model"):
        snapshot = await _fetch_model(spec, hub_token, config_commit)
    model_commit = spec.model_revision or snapshot_commit(snapshot)

    harmony_date = _harmony_date(hp)
    task = build_vllm_task(
        str(snapshot),
        served_model_name=spec.model_id,  # the client below keeps addressing it by its Hub id
        port=_SERVE_PORT,
        host=_SERVE_HOST,
        tensor_parallel_size=hp.tensor_parallel_size,
        max_model_len=hp.max_model_len,
        dtype=hp.dtype,
        setup="",  # vLLM is already installed by the server's setup step; don't reinstall
        # Serve with the interpreter this runner is executing under. LocalBackend runs the task
        # through a login shell, which re-sources the profile and drops this virtualenv from PATH:
        # a bare `vllm` is then "command not found" even though vLLM is installed right here.
        python_executable=sys.executable,
        extra_args=_serve_args(hp),
    )
    # Unbuffered: the served process writes through a pipe, so CPython would otherwise hold
    # its output in an 8 KiB block buffer — and a server that hangs before filling it leaves
    # the log file empty, which is precisely the case the file exists for.
    # A batch run must not depend on a CUDA build toolchain being present on someone else's box.
    # vLLM's default sampler is FlashInfer's, which JIT-COMPILES its kernels during warmup: it
    # shells out to ninja, and a GPU image carrying the driver and runtime but no build tools
    # fails with "No such file or directory: 'ninja'" — after loading the weights, compiling the
    # graph, capturing CUDA graphs and allocating the KV cache, so the run has already paid for
    # everything before it dies. The PyTorch-native sampler is marginally slower per token and
    # needs no compiler, which is the right default for a machine we do not provision.
    task = task.model_copy(
        update={
            "env": {
                # The server's whole environment: an allow-listed slice of this one, then the
                # task's own settings. The backend below does not inherit, so nothing else
                # reaches it.
                **model_server_environ(),
                **task.env,
                # Unbuffered: the served process writes through a pipe, so CPython would otherwise
                # hold its output in an 8 KiB block buffer — and a server that hangs before filling
                # it leaves the log file empty, which is precisely the case the file exists for.
                "PYTHONUNBUFFERED": "1",
                "VLLM_USE_FLASHINFER_SAMPLER": "0",
                # The weights are already local. Offline, the server cannot fetch a file the
                # download above refused, and has no reason to look for a credential.
                "HF_HUB_OFFLINE": "1",
                **({_HARMONY_DATE_ENV: harmony_date} if harmony_date else {}),
            }
        }
    )
    # serving_endpoint reports "Starting the model server" itself the moment it submits, so
    # announcing it here as well would show the same phrase twice for one step.
    async with serving_endpoint(
        # Tee the served process's streams into the run workdir. When the runner dies, the
        # buffers die with it; the files are what is left to explain why the server never came up.
        LocalBackend(log_dir=Path.cwd(), env_inherit=False),
        task,
        base_url=f"http://{_SERVE_HOST}:{_SERVE_PORT}/v1",
        wait_timeout_s=hp.wait_timeout_s,
        # serving.py reports prose through a plain one-arg hook and knows nothing about run
        # stages, which is right: bind the stage here rather than widening its public API.
        on_phase=lambda message: phase(message, stage="load_model"),
    ) as endpoint:
        # Read once the server is up, while its logs and routes still exist.
        serving_facts: dict[str, Any] = {
            "max_model_len": await _served_max_model_len(endpoint.base_url, spec.model_id),
            "dtype": {
                "requested": hp.dtype,
                "checkpoint": await asyncio.to_thread(_checkpoint_dtype, snapshot),
                "served": await asyncio.to_thread(_served_dtype, Path.cwd()),
            },
            "flags": _serve_flags(task.run),
            # The parser served with, which an unset one on a gpt-oss checkpoint defaults to.
            "reasoning_parser": hp.reasoning_parser,
            "harmony_date": harmony_date,
        }
        # The endpoint this runner just launched is on loopback and takes no credential, and
        # saying so EXPLICITLY is what keeps it deterministic: left unset, the client falls back
        # to whatever OPENAI_API_KEY the VM happens to carry, which would send the user's real
        # provider key to a local server that never asked for one.
        provider = OpenAICompatProvider(
            OpenAICompatConfig(
                base_url=endpoint.base_url, api_key=SecretStr(UNAUTHENTICATED_API_KEY)
            )
        )
        client = LLMClient(
            model=spec.model_id,
            provider="openai_compat",
            provider_clients={"openai_compat": provider},  # pins the LOCAL endpoint
        )
        # The first `step` only lands once a whole progress_chunk has completed, and that
        # chunk absorbs the client's cold start on top of its generations.
        async with ticking_phase(phase, "Generating responses", stage="run"):
            out = await _run_batches(
                spec,
                client,
                prompts,
                custom_ids,
                writer,
                endpoint.is_alive,
                gpu,
                redactor=run_redactor(hf_token),
            )

    # Results are ALWAYS written outside the per-run workdir, whether or not they are then pushed.
    # The orchestrator deletes that workdir on every terminal state, so writing there and pushing
    # from it means a failed upload — an expired token, a rate limit, a network blip — takes the
    # entire run's output with it. Hours of generation, gone at the last step, with nothing left to
    # retry from. Outside it, the parquet survives for the user to retrieve over SSH (or push by
    # hand) exactly as in the no-push case.
    #
    # Both branches sit between the last `step` (which reads 100%) and `end`, so a run that dies
    # here would otherwise look like it died complete, with no explanation.
    async with ticking_phase(phase, "Writing results", stage="push"):
        results_path = await asyncio.to_thread(_write_results, out, _local_results_dir(spec.run_id))
        # Beside the results, so the push and the on-VM copy carry it alike.
        await asyncio.to_thread(
            write_manifest,
            _build_manifest(spec, out, model_commit, serving_facts),
            Path(results_path).parent,
        )
    if hf_token and spec.output_repo_id:
        try:
            async with ticking_phase(phase, "Uploading results to the Hub", stage="push"):
                destination = await _push_results(spec, results_path, hf_token)
        except Exception as exc:
            # Name WHERE the results are. The run still fails — the user asked for them on the Hub
            # and they are not there — but a failure at the last step of a long run is the moment
            # it matters most that the output was not lost with it.
            msg = f"results are on the VM at {results_path}, but the push failed: {exc}"
            raise RunError(msg) from exc
    else:
        destination = str(results_path)

    ok = sum(1 for r in out if r["error"] is None)
    emit(
        writer,
        ProgressEvent(
            kind="end",
            stage="push",
            step=len(out),
            total_steps=len(out),
            metrics={"succeeded": float(ok), "failed": float(len(out) - ok)},
            message=destination,  # WHERE results landed: a repo id (pushed) or a VM path (local)
        ),
    )
    # AFTER `end`: the counts and the destination are true whichever way the verdict falls, and
    # for a run that failed this way they are the diagnosis — they say every row failed and where
    # the per-row errors can be read.
    _check_produced_output(out, ok)
    return destination


async def _dataset_commit(spec: RunSpec, token: str | Literal[False]) -> str:
    """The dataset commit a scoring run reads: its pin, or the default branch's head resolved now.

    Resolved once, before the load, and then loaded at that commit, so the manifest names the
    exact rows scored even when the spec carried no pin.
    """
    if spec.dataset_commit_sha:
        return spec.dataset_commit_sha
    try:
        hf_mod: Any = __import__("huggingface_hub")  # the [storage] extra
        info: Any = await asyncio.to_thread(
            hf_mod.HfApi().dataset_info, spec.dataset_id, token=token
        )
        sha = str(info.sha)
    except Exception as exc:
        msg = f"could not resolve the dataset {spec.dataset_id} to a commit: {exc}"
        raise RunError(msg) from exc
    require_commit_sha(sha, "the resolved dataset commit")
    return sha


async def _require_fresh_output(spec: RunSpec, hf_token: str) -> None:
    """Refuse an output repo that already holds files: one scoring run, one repo.

    Scores are read back by joining on the repo, so a repo holding an earlier run's commit would
    let a reader mix two runs' rows. A repo the preflight just created holds only the Hub's own
    ``.gitattributes``.
    """
    out_repo = validate_repo_id(cast("str", spec.output_repo_id), "output repo")
    try:
        files = await HFHubClient(token=hf_token).list_repo_files(out_repo, repo_type="dataset")
    except Exception as exc:
        msg = f"could not list the output repo {out_repo}: {exc}"
        raise RunError(msg) from exc
    held = sorted(name for name in files if name != ".gitattributes")
    if held:
        msg = (
            f"the output repo {out_repo} already holds {len(held)} file(s); a scoring run pushes "
            "to a repo of its own, so choose a new output repo"
        )
        raise RunError(msg)


async def _execute_scoring(
    spec: RunSpec, secrets: RunSecrets, writer: JsonlProgressWriter | None
) -> str:
    """Score every row with a scoring engine; no model server is started.

    The same frame as a chat run: the output repo is checked before any download, the model comes
    from the same filtered snapshot, and the results and manifest are written outside the workdir
    and pushed in one commit. What differs is the middle: rows over the model's length limit are
    dropped and counted before the first forward pass, and the rest run in length-sorted batches.
    The engine (a classification head, or a causal LM's label tokens) only measures, scores and
    describes rows; everything around it is shared.
    """
    hf_token = secrets.hf_token_value()
    hub_token = secrets.hub_credential()
    hp = spec.hyperparams
    gpu = GpuSampler()
    phase = phase_sink(writer, hf_token, gpu=gpu)
    pushing = bool(hf_token and spec.output_repo_id)
    # What the user asked for, kept for the manifest beside the commits actually read.
    requested = spec

    if pushing:
        async with ticking_phase(phase, "Checking the output repo", stage="load_model"):
            await _check_output_repo(spec, hf_token)
            await _require_fresh_output(spec, cast("str", hf_token))
    async with ticking_phase(phase, "Loading the dataset", stage="load_model"):
        # Loaded at the commit that is recorded: the requested pin, or the head resolved now.
        dataset_commit = await _dataset_commit(spec, hub_token)
        spec = spec.model_copy(update={"dataset_commit_sha": dataset_commit})
        rows = await asyncio.to_thread(_load_rows, spec, hub_token)
    if not rows:
        msg = "the run produced no rows: the dataset split was empty"
        raise RunError(msg)
    inputs = build_inputs(spec, rows)
    emit(writer, ProgressEvent(kind="start", stage="run", total_steps=len(rows)))

    async with ticking_phase(phase, "Downloading the model", stage="load_model"):
        snapshot = await _fetch_model(spec, hub_token)
    model_commit = spec.model_revision or snapshot_commit(snapshot)
    async with ticking_phase(phase, _LOADING_PHASE[hp.engine], stage="load_model"):
        scorer = await asyncio.to_thread(_load_scorer, snapshot, hp)
    async with ticking_phase(phase, "Measuring rows", stage="run"):
        lengths = await asyncio.to_thread(scorer.lengths, inputs)
    batches, overflow = plan_batches(
        lengths, max_length=scorer.max_length, batch_size=hp.scoring_batch_size
    )

    logits: dict[int, list[float]] = {}
    started = time.monotonic()
    async with ticking_phase(phase, "Scoring rows", stage="run"):
        for batch in batches:
            values = await asyncio.to_thread(scorer.score, inputs, batch)
            logits.update(zip(batch, values, strict=True))
            elapsed = time.monotonic() - started
            emit(
                writer,
                ProgressEvent(
                    kind="step",
                    stage="run",
                    step=len(logits) + len(overflow),
                    total_steps=len(rows),
                    metrics={
                        "succeeded": float(len(logits)),
                        "failed": 0.0,
                        "overflow": float(len(overflow)),
                        **({"rows_per_s": len(logits) / elapsed} if elapsed > 0 else {}),
                        **gpu.sample(),
                    },
                ),
            )

    out = scorer.result_rows(lengths, logits, inputs)
    async with ticking_phase(phase, "Writing results", stage="push"):
        results_path = await asyncio.to_thread(_write_results, out, _local_results_dir(spec.run_id))
        await asyncio.to_thread(
            write_manifest,
            scorer.manifest(
                requested,
                out,
                model_commit=model_commit,
                dataset_commit=dataset_commit,
                batches=batches,
            ),
            Path(results_path).parent,
        )
    if pushing:
        try:
            async with ticking_phase(phase, "Uploading results to the Hub", stage="push"):
                await _require_fresh_output(spec, cast("str", hf_token))
                destination = await _push_results(spec, results_path, cast("str", hf_token))
        except Exception as exc:
            msg = f"results are on the VM at {results_path}, but the push failed: {exc}"
            raise RunError(msg) from exc
    else:
        destination = str(results_path)

    counts = tally(out)
    emit(
        writer,
        ProgressEvent(
            kind="end",
            stage="push",
            step=len(out),
            total_steps=len(out),
            metrics={
                "succeeded": float(counts["scored"]),
                "failed": 0.0,
                "overflow": float(counts["overflow"]),
            },
            message=destination,
        ),
    )
    if not counts["scored"]:
        msg = (
            f"all {len(out)} rows exceed the model's limit of {scorer.max_length} tokens, so "
            "none was scored (nothing is truncated)"
        )
        raise RunError(msg)
    return destination


_LOADING_PHASE = {"hf_classify": "Loading the classifier", "hf_label_logits": "Loading the model"}


class _Scorer(Protocol):
    """What the scoring frame drives: measure rows, score a batch, describe the results."""

    max_length: int

    def lengths(self, inputs: ScoringInputs) -> list[int | None]: ...

    def score(self, inputs: ScoringInputs, batch: Sequence[int]) -> list[list[float]]: ...

    def result_rows(
        self,
        lengths: Sequence[int | None],
        values: Mapping[int, Sequence[float]],
        inputs: ScoringInputs,
    ) -> list[dict[str, Any]]: ...

    def manifest(
        self,
        spec: RunSpec,
        rows: Sequence[Mapping[str, Any]],
        *,
        model_commit: str | None,
        dataset_commit: str | None,
        batches: Sequence[Sequence[int]],
    ) -> dict[str, Any]: ...


def _load_scorer(snapshot: Path, hp: Hyperparams) -> _Scorer:
    """The run's scoring engine, loaded from the local snapshot."""
    if hp.engine == "hf_label_logits":
        return load_label_reader(
            snapshot,
            label_tokens=hp.label_tokens or [],
            prefill=hp.assistant_prefill,
            chat_template_kwargs=hp.chat_template_kwargs,
            max_length=hp.max_model_len,
            dtype=cast("str", hp.dtype),
        )
    classifier = load_classifier(
        snapshot,
        label_alias=cast("dict[str, str | None]", hp.label_alias),
        max_length=hp.max_model_len,
        dtype=cast("str", hp.dtype),
    )
    return ClassifierScorer(classifier)


async def main() -> int:
    """Entry point: returns a process exit code (0 ok, 1 failure). Never leaks the token."""
    return await runner_main(
        lambda writer, secrets: _execute(load_spec(writer=writer), secrets, writer)
    )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
