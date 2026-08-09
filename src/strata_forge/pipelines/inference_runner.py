"""VM-side batch-inference runner: STRATA_RUN_CONFIG (inert JSON) -> vLLM -> HF results.

Launched on the user's own VM by the control plane (``python -m
strata_forge.pipelines.inference_runner``). It reads an INERT run spec from the
``STRATA_RUN_CONFIG`` env var, loads a Hugging Face dataset split, renders one prompt
per row by simple ``{column}`` substitution (NO code execution), serves the model with
vLLM on the same VM, runs a concurrency-bounded batch through an openai-compatible
``LLMClient``, and then EITHER pushes ``{custom_id, output}`` results to the user's HF
dataset repo (when a write token + an output repo are supplied) OR keeps them as a parquet
file on the VM (outside the per-run workdir, for the user to retrieve over SSH). Live
progress is appended to the file named by ``FORGE_PROGRESS_PATH`` (the orchestrator tails it).

Security boundary (the VM is where untrusted-but-allow-listed config meets real
credentials + the network):

  - The config is parsed as DATA only: ``json`` + Pydantic ``extra="forbid"`` — never
    ``eval``/``pickle``/``yaml.unsafe_load``. ``template``/``column_mapping``/
    ``hyperparams`` are never executed.
  - Template rendering is a bounded, NON-executing ``{name}`` substitution (a regex, NOT
    ``str.format`` and NOT a template engine) — only placeholders that map to a real
    column are replaced; anything else stays literal.
  - The HF write token arrives in its OWN env var (``HF_WRITE_TOKEN``), never in
    ``STRATA_RUN_CONFIG``, is passed EXPLICITLY to the Hub/dataset clients (never the
    VM's ambient ``HF_TOKEN``), and is scrubbed from any surfaced error.
  - Progress + result rows carry no secret: events hold step counts + float metrics + a
    repo id; result rows are ``{custom_id, output, error}`` (model text only).
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from strata_forge.compute import LocalBackend
from strata_forge.compute.batch import BatchInferenceRunner
from strata_forge.compute.serving import build_vllm_task, serving_endpoint
from strata_forge.llm import LLMClient, UserMessage
from strata_forge.llm.providers.config import OpenAICompatConfig
from strata_forge.llm.providers.openai_compat import OpenAICompatProvider
from strata_forge.storage import HFHubClient
from strata_forge.training.progress import JsonlProgressWriter, ProgressEvent

__all__ = ["RunSpec", "main", "render_template"]

# vLLM serves on loopback only — the runner talks to it via localhost, and a multi-tenant
# / network-reachable VM must not expose the model server.
_SERVE_HOST = "127.0.0.1"
_SERVE_PORT = 8000
# A bare ``{name}`` placeholder only — no attribute/index access, no format mini-language.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
# An HF repo id: ``owner/name``, each segment alphanumeric-led, no traversal/scheme/space.
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
# A safe single path segment for the on-VM results dir name (no slash / traversal / shell chars).
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")
# Cap a rendered prompt so a pathological row can't blow up memory / the request.
_MAX_RENDERED_CHARS = 200_000
# Scrub token-shaped substrings from any surfaced error (defense in depth on top of
# replacing the known token value).
_TOKEN_RE = re.compile(r"(hf_[A-Za-z0-9]{8,}|Bearer\s+[A-Za-z0-9._\-]+)")
_RESULTS_FILENAME = "results.parquet"


class Hyperparams(BaseModel):
    """Inert tuning knobs (sampling + serving + bounds). Unknown keys are rejected."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = None
    concurrency: int = Field(default=8, ge=1, le=256)
    progress_chunk: int = Field(default=200, ge=1)
    row_limit: int | None = Field(default=None, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    max_model_len: int | None = Field(default=None, ge=1)
    dtype: str | None = None
    wait_timeout_s: float = Field(default=1800.0, gt=0)


class RunSpec(BaseModel):
    """The inert run spec the server delivers in ``STRATA_RUN_CONFIG`` (no secrets)."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_id: str
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


class RunError(Exception):
    """A runner failure whose message is safe to surface (already token-scrubbed)."""


def _validate_repo_id(repo_id: str, what: str) -> str:
    """Defensively re-validate an id even though the server allow-listed it — the VM is the
    trust boundary that actually fetches/pushes."""
    # fullmatch (not match): match's `$` accepts a trailing newline ("org/x\n").
    if ".." in repo_id or not _REPO_ID_RE.fullmatch(repo_id):
        msg = f"invalid {what} id"
        raise RunError(msg)
    return repo_id


def _sanitize(text: str, token: str | None) -> str:
    """Strip the write token + any token-shaped substring from a message before it's emitted."""
    if token:
        text = text.replace(token, "***")
    return _TOKEN_RE.sub("***", text)


def load_spec() -> RunSpec:
    raw = os.environ.get("STRATA_RUN_CONFIG")
    if not raw:
        msg = "STRATA_RUN_CONFIG is not set"
        raise RunError(msg)
    try:
        spec = RunSpec.model_validate_json(raw)  # parse + validate as DATA; no eval/yaml
    except ValueError as exc:
        msg = f"invalid STRATA_RUN_CONFIG: {exc}"
        raise RunError(msg) from exc
    _validate_repo_id(spec.model_id, "model")
    _validate_repo_id(spec.dataset_id, "dataset")
    if spec.output_repo_id is not None:
        _validate_repo_id(spec.output_repo_id, "output repo")
    return spec


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


def _load_rows(spec: RunSpec, hf_token: str | None) -> list[dict[str, Any]]:
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
        token=hf_token or None,
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


def _emit(writer: JsonlProgressWriter | None, event: ProgressEvent) -> None:
    if writer is not None:
        writer.emit(event)


async def _run_batches(
    spec: RunSpec,
    client: LLMClient,
    prompts: list[list[UserMessage]],
    custom_ids: list[str],
    writer: JsonlProgressWriter | None,
) -> list[dict[str, Any]]:
    """Run the prompts in progress-chunked batches; reconcile results positionally."""
    hp = spec.hyperparams
    runner = BatchInferenceRunner(client, concurrency=hp.concurrency, on_error="collect")
    total = len(prompts)
    out: list[dict[str, Any]] = []
    ok = failed = 0
    for start in range(0, total, hp.progress_chunk):
        chunk = prompts[start : start + hp.progress_chunk]
        ids = custom_ids[start : start + hp.progress_chunk]
        results = await runner.run(
            chunk, temperature=hp.temperature, max_tokens=hp.max_tokens, top_p=hp.top_p
        )
        for cid, result in zip(ids, results, strict=True):
            if result.succeeded and result.response is not None:
                out.append({"custom_id": cid, "output": result.response.text, "error": None})
                ok += 1
            else:
                out.append({"custom_id": cid, "output": None, "error": repr(result.error)})
                failed += 1
        _emit(
            writer,
            ProgressEvent(
                kind="step",
                step=len(out),
                total_steps=total,
                metrics={"succeeded": float(ok), "failed": float(failed)},
            ),
        )
    return out


def _local_results_dir(run_id: str | None) -> Path:
    """A stable, cleanup-surviving location for results when NOT pushing to the Hub: outside the
    per-run workdir the orchestrator deletes, named by the run id so the user can retrieve it over
    SSH. Falls back to the cwd when no usable run id was provided (degraded — may be cleaned — but
    never crashes; the run id is validated as a single safe path segment)."""
    if run_id and _SAFE_NAME_RE.fullmatch(run_id):
        return Path.home() / "strata-inference-results" / run_id
    return Path.cwd()


def _write_results(rows: list[dict[str, Any]], outdir: Path) -> Path:
    """Write the results as parquet (renders in the HF Datasets Viewer)."""
    datasets_mod: Any = __import__("datasets")  # already required by _load_rows ([hf] extra)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / _RESULTS_FILENAME
    datasets_mod.Dataset.from_list(rows).to_parquet(str(path))
    return path


async def _push_results(spec: RunSpec, results_path: Path, hf_token: str) -> str:
    if not spec.output_repo_id:
        msg = "output_repo_id is required to push results"
        raise RunError(msg)
    out_repo = _validate_repo_id(spec.output_repo_id, "output repo")
    hub = HFHubClient(token=hf_token)  # EXPLICIT write token, never the ambient HF_TOKEN
    await hub.create_repo(out_repo, repo_type="dataset", private=True, exist_ok=True)
    await hub.upload_file(
        results_path,
        _RESULTS_FILENAME,
        out_repo,
        repo_type="dataset",
        commit_message="batch inference results",
    )
    return out_repo


async def _execute(spec: RunSpec, hf_token: str | None, writer: JsonlProgressWriter | None) -> str:
    rows = _load_rows(spec, hf_token)
    prompts, custom_ids = _build_requests(spec, rows)
    _emit(writer, ProgressEvent(kind="start", total_steps=len(prompts)))

    hp = spec.hyperparams
    task = build_vllm_task(
        spec.model_id,
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
    )
    async with serving_endpoint(
        LocalBackend(),
        task,
        base_url=f"http://{_SERVE_HOST}:{_SERVE_PORT}/v1",
        wait_timeout_s=hp.wait_timeout_s,
    ) as endpoint:
        provider = OpenAICompatProvider(OpenAICompatConfig(base_url=endpoint.base_url))
        client = LLMClient(
            model=spec.model_id,
            provider="openai_compat",
            provider_clients={"openai_compat": provider},  # pins the LOCAL endpoint
        )
        out = await _run_batches(spec, client, prompts, custom_ids, writer)

    # Push to the Hub only when BOTH a write token and an output repo are present (the control
    # plane injects them together). Otherwise keep the results on the VM for the user to retrieve
    # over SSH — written OUTSIDE the per-run workdir so the orchestrator's cleanup leaves them intact.
    if hf_token and spec.output_repo_id:
        results_path = _write_results(out, Path.cwd())
        destination = await _push_results(spec, results_path, hf_token)
    else:
        results_path = _write_results(out, _local_results_dir(spec.run_id))
        destination = str(results_path)

    ok = sum(1 for r in out if r["error"] is None)
    _emit(
        writer,
        ProgressEvent(
            kind="end",
            step=len(out),
            total_steps=len(out),
            metrics={"succeeded": float(ok), "failed": float(len(out) - ok)},
            message=destination,  # WHERE results landed: a repo id (pushed) or a VM path (local)
        ),
    )
    return destination


def _progress_path() -> str | None:
    """Resolve the progress file: the spec's progress_path (read defensively, the spec may be
    invalid) else FORGE_PROGRESS_PATH."""
    raw = os.environ.get("STRATA_RUN_CONFIG") or "{}"
    try:
        from_spec = json.loads(raw).get("progress_path")
    except ValueError, AttributeError:
        from_spec = None
    return from_spec or os.environ.get("FORGE_PROGRESS_PATH")


async def main() -> int:
    """Entry point: returns a process exit code (0 ok, 1 failure). Never leaks the token."""
    hf_token = os.environ.get("HF_WRITE_TOKEN") or None
    path = _progress_path()
    writer = JsonlProgressWriter(path) if path else None
    try:
        spec = load_spec()
        await _execute(spec, hf_token, writer)
    except Exception as exc:  # top-level runner boundary: report + exit nonzero, never leak
        _emit(writer, ProgressEvent(kind="error", message=_sanitize(str(exc), hf_token)))
        return 1
    else:
        return 0
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(asyncio.run(main()))
