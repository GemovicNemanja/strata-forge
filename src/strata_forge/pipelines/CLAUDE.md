# Agent rules — strata_forge.pipelines

`strata_forge.pipelines` holds runnable entrypoints that are **launched on a compute target**, not
imported by user code. Each module reads an inert run spec from the environment, composes the
library's serving / batch / storage primitives into one job, and appends
:class:`strata_forge.training.progress.ProgressEvent` records to a file that the launching process
reads back on its own schedule. See
[ADR 0016](../../../docs/architecture/adr/0016-backend-read-file.md) for why the progress channel
is a pulled file rather than a pushed connection.

This module inverts the usual direction of the library, and almost every rule below follows from
that. Everywhere else, Forge code runs inside the caller's process and the caller owns the
environment. Here, the code runs alone on a machine, is handed everything through environment
variables, and reports back through a file and an exit code.

## Purpose

- ``inference_runner.py`` — batch inference over a Hugging Face dataset split against a model
  served locally by vLLM, with results written as parquet and optionally pushed to the Hub.
- :class:`RunSpec` + ``Hyperparams`` — the inert, ``extra="forbid"`` spec shape delivered as JSON
  in the ``STRATA_RUN_CONFIG`` env var.
- :func:`render_template` — a bounded, non-executing ``{name}`` substitution over a dataset row.
- :func:`main` — the process entrypoint; returns an exit code, never raises.

## Boundaries

- **Owns:** ``inference_runner.py`` and any future entrypoint module. One file per entrypoint;
  they do not import each other.
- **Imports from inside ``forge``:** :mod:`strata_forge.compute` (``LocalBackend``,
  ``BatchInferenceRunner``, ``build_vllm_task``, ``serving_endpoint``), :mod:`strata_forge.llm`
  (``LLMClient``, ``UserMessage``, the ``openai_compat`` provider + its config),
  :mod:`strata_forge.storage` (``HFHubClient``), and
  :mod:`strata_forge.training.progress` (``JsonlProgressWriter``, ``ProgressEvent``).
- **Does NOT import** :mod:`strata_forge.core`, :mod:`strata_forge.config`,
  :mod:`strata_forge.datasets`, :mod:`strata_forge.prompts`, :mod:`strata_forge.evals`,
  :mod:`strata_forge.agents`, :mod:`strata_forge.rag`, :mod:`strata_forge.tracing`.
- **This module is the top of the dependency graph.** It is the only place allowed to compose four
  sibling modules at once, and nothing in ``strata_forge`` may import *from* it. If another module
  needs something an entrypoint does, that logic belongs in the module, not here.
- **External deps:** Pydantic in the core install. ``datasets`` rides behind ``[hf]``,
  ``huggingface_hub`` behind ``[storage]`` (via :mod:`strata_forge.storage`), and vLLM must be
  installed on the target machine — the runner shells out to it and never imports it.

## Public API

``inference_runner.py`` declares ``__all__ = ["RunSpec", "main", "render_template"]``. The package
``__init__.py`` deliberately re-exports **nothing**: these modules are executed with ``python -m``,
so a package-level surface would imply an import-and-call usage that does not exist.

``Hyperparams``, ``RunError`` and ``load_spec`` are module-level and public-by-accident. Treat them
as internal; do not add them to ``__all__`` without deciding they are part of the contract.

Errors raised inside the runner are :class:`RunError` for failures whose message is safe to
surface, plus whatever the composed primitives raise. None of it escapes: :func:`main` catches
everything at the process boundary.

## Internal patterns

- **The run spec is inert data.** Parsed with ``json`` + Pydantic ``extra="forbid"`` — never
  ``eval``, ``pickle``, ``yaml.unsafe_load``, or anything else that can execute. ``template``,
  ``column_mapping`` and ``hyperparams`` are values. A new spec field is a new typed Pydantic
  field with bounds, not a free-form ``dict[str, Any]`` passthrough.
- **Template rendering executes nothing.** A regex over ``[A-Za-z0-9_]+`` placeholder names,
  deliberately not ``str.format`` (which reaches attributes and indices and carries a format
  mini-language) and not a template engine. Output is length-capped. Do not "improve" this into
  Jinja; :mod:`strata_forge.prompts` is where templating with a sandbox lives.
- **Ids are re-validated here.** ``owner/name`` shape, no traversal, and ``fullmatch`` rather than
  ``match`` so a trailing newline cannot slip through. Upstream validation is not a reason to skip
  it — this process is the one that actually fetches and pushes.
- **Credentials arrive in their own env var** (``HF_WRITE_TOKEN``), never in the spec, are passed
  **explicitly** to the client that needs them rather than relied on ambiently, and are scrubbed
  from any message that reaches the progress file. New credentials follow the same three rules.
- **Progress emission is best-effort and optional.** ``_emit`` no-ops when no writer is configured.
  A run with no progress path must still work; progress is observability, not control flow.
- **The entrypoint returns an exit code.** :func:`main` catches ``Exception`` at the top level,
  emits an ``error`` event, and returns ``1``. Only ``if __name__ == "__main__"`` calls
  ``sys.exit``. This is the one sanctioned broad ``except`` in the module — it is a process
  boundary, and it re-reports rather than swallowing.
- **Bound anything the spec can grow.** ``row_limit`` slices *during* iteration so an oversized
  split is never materialized; ``concurrency`` is capped; rendered prompts are capped. Any new
  input whose size the spec controls needs the same treatment.

## Test expectations

- Unit tests under ``tests/unit/pipelines/``, one file per entrypoint module.
- **No network, no GPU, no real vLLM.** The heavy externals (``datasets``, ``serving_endpoint``,
  the batch runner, the Hub client) are patched; pure functions are tested directly.
- **The security properties are the load-bearing tests.** Every entrypoint needs coverage for:
  a hostile template rendering literally, ``extra="forbid"`` rejecting unknown spec keys, id
  validation rejecting traversal / scheme / whitespace / trailing-newline forms, ``row_limit``
  stopping an unbounded iterator, and — most importantly — the token never appearing in the
  progress file on either the happy path or the error path.
- Assert on the emitted event stream, not just return values: kinds, ordering, and the final
  ``message`` naming the destination.

## Gotchas

- **Module-level imports must stay light.** ``import strata_forge.pipelines.inference_runner``
  has to succeed on a bare install with no extras, so that a spec can be built and validated
  anywhere. ``datasets`` is imported inside the function that loads rows; ``huggingface_hub``
  stays behind :mod:`strata_forge.storage`'s own lazy import; vLLM is never imported at all. A
  top-level ``import datasets`` breaks this and no test of the happy path will notice.
- **The progress file is append-only.** ``JsonlProgressWriter`` opens in ``"a"`` and flushes per
  line. Never truncate it, never rewrite it, never buffer a batch of events, and never emit
  anything but one complete JSON object per line — a reader is tailing it concurrently and a
  partial or rewritten file corrupts its view. Summarising many rows into one periodic event is
  the right way to control volume; batching writes is not.
- **Never assume the orchestrator's environment.** No config file, no working directory layout, no
  ``strata_forge.config`` lookup, no inbound network path, no assumption that anything is
  listening. If the runner needs a value, it arrives in an env var and has a defined behaviour
  when absent. The progress path in particular is resolved defensively — read straight from the
  raw env var so that an *invalid* spec can still report its own failure.
- **Results must survive cleanup.** An orchestrator calls ``Backend.cleanup``, which deletes the
  job's working directory. Anything the user is meant to collect afterwards goes outside it, in a
  location named by a validated single path segment.
- **``datasets`` is not :mod:`strata_forge.datasets`.** The runner consumes an arbitrary upstream
  Hugging Face split, not a Forge-owned ``Dataset``. Do not "fix" this by routing through
  :mod:`strata_forge.datasets`.
- **A failed row is not a failed run.** The batch runs with ``on_error="collect"`` so one bad row
  cannot abort a long job; per-row errors land in the results file and the ``failed`` counter.
  Do not switch it to ``"raise"``.
- **Docstrings here ship to PyPI.** Per root ``CLAUDE.md`` §9b, write them for an outside reader:
  describe what a library user controls — env vars, a spec, a progress file, an exit code — and
  name no private infrastructure, deployment topology, or internal service.

## When to update this file

- Adding a new entrypoint module.
- Adding or changing a field on a run spec, or the shape of the spec contract.
- Adding an env var an entrypoint reads, or a credential it accepts.
- Changing the progress event vocabulary or the file protocol.
- Changing the exit-code semantics.
- Changing which sibling modules an entrypoint composes.
