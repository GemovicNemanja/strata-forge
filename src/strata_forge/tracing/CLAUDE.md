# Agent rules — strata_forge.tracing

`strata_forge.tracing` owns Langfuse observability and explicit local attempt metadata. It wraps every
other module *from above* — no other `strata_forge.*` module imports
`strata_forge.tracing`. See
[ADR 0008](../../../docs/architecture/adr/0008-tracing-as-cross-cutting.md)
for the rationale.

## Purpose

Langfuse-based observability:

- **Auto-tracing for LLM calls** via LiteLLM's Langfuse callback,
  registered once at process startup by
  :func:`install_litellm_callback`. Every subsequent LiteLLM call —
  including those from third-party libraries that don't know about
  Forge — appears in Langfuse without `strata_forge.llm` knowing.
- **`@traced` decorator** for sync + async functions outside the LLM
  call path (eval graders, agent loops, RAG pipelines).
- **`traced_span()` async context manager** for nested observations
  inside a trace.
- **Score + metric helpers** for attaching grader feedback and
  arbitrary application metrics to traces or observations.
- **`correlation_id` integration**: structlog records carry the active
  Langfuse trace ID via the existing
  :data:`strata_forge.core.ids.correlation_id_var` ContextVar.

## Boundaries

- **Owns:** `client.py`, `litellm_callback.py`, `decorator.py`,
  `span.py`, `score.py`, `metrics.py`, `prompt_attempt_types.py`, `prompt_attempts.py`,
  `prompt_provider_values.py`, `prompt_provider_types.py`, `prompt_provider.py`,
  `prompt_stream_types.py`, `prompt_streams.py`.
- **Imports from inside `forge`:** `strata_forge.core` (logger,
  correlation_id), `strata_forge.config` (LangfuseConfig), `strata_forge.llm`
  (message/response types — for serializing trace payloads),
  `strata_forge.prompts` (typed local consumption and body-free prompt metadata).
- **Imports of `langfuse`** happen lazily inside the functions that
  need them. The package is behind the `[langfuse]` extra; importing
  `strata_forge.tracing` without the extra MUST work.
- **Reverse imports are banned.** `strata_forge.llm`, `strata_forge.prompts`, and
  every downstream module MUST NOT import from `strata_forge.tracing`. This
  is the cross-cutting design from ADR 0008.

## Public API

The module's `__init__.py` re-exports the supported surface:

- `get_client() -> Langfuse | None` — the cached singleton.
- `install_litellm_callback() -> None` — wire LiteLLM ↔ Langfuse.
- `@traced(name=None, tags=...)` — decorator for sync + async fns.
- `traced_span(name, **fields)` — async context manager.
- `score_trace`, `score_observation` — attach feedback to a trace.
- `record_numeric_metric`, `record_categorical_metric` — attach
  arbitrary application metrics.

Errors raised from this module are `ForgeError` subclasses; the
existing Langfuse helper paths are silent no-ops when unconfigured (never a crash).
The explicit local prompt-attempt recorder is independent of that configuration, as defined below.

## Explicit local byte-attempt recorder

`prepare_invocation` and `capture_once` are an explicit offline wrapper above prompts/core.
They never call get_client, read Langfuse credentials, install callbacks or export content.
Existing Langfuse functions retain their disabled no-op behavior. Prompts, llm and other
lower modules MUST NOT import this recorder or tracing. The one supplied application byte
delegate is a behavior protocol, not an auth/policy callback or a built-in provider client.

The sole retention mode is metadata_only. Working render/response containers use opaque
slots/private storage and safe repr; automatic serialization must not reveal bodies. Only
explicit body accessors return working content. Metadata snapshots contain qualified local
caller-bytes revalidation, original opaque byte hashes and explicit unknown provider/render
projection, never a provider-use/provenance or remote-access claim.

Python owns exactly one shielded delegate child per attempt, cancels it once and drains it
before slot reuse under repeated parent cancellation. No late retired body is inspected,
hashed or accepted as success. Settled ledger snapshots refuse while a call is active. A
declared baseline with no actual handoff uses not_observed, not guessed same/changed bytes.
No model/provider, remote loader, automatic retry, exporter or trace ingestion is enabled.

## Internal patterns

- **Lazy Langfuse import.** Every function that touches the Langfuse
  SDK imports it inside the function body. `strata_forge.tracing` imports
  cleanly without the extra installed.
- **`get_client()` is cached.** Once built, the singleton sticks for
  the lifetime of the process. Tests use `reset_client()` in conftest
  to clear it between cases.
- **No-op when unconfigured.** `LangfuseConfig.enabled` is False when
  either credential is missing. Every Langfuse-facing public function checks
  `get_client()` and short-circuits cleanly — no warnings, no
  crashes, no fallback prints. The diagnostic (`strata-forge doctor`) is
  where missing configuration surfaces.
- **Correlation IDs propagate.** When a trace starts, the trace ID is
  written into `strata_forge.core.ids.correlation_id_var` so subsequent log
  records carry it. The contextvar is restored on trace exit.
- **`install_litellm_callback()` is idempotent.** Calling it multiple
  times doesn't duplicate the callback. The check is by string
  identifier in `litellm.success_callback` and
  `litellm.failure_callback`.

## Test expectations

- Unit tests under `tests/unit/tracing/`, one file per source module.
- Coverage target: ≥ 90 % line.
- Mocked Langfuse client for every unit test — no live network.
- One `@pytest.mark.integration` test runs against the
  docker-compose Langfuse stack: starts a trace, runs an LLM call,
  retrieves the trace via the Langfuse API, verifies shape. Skipped
  unless `make stack-up` has the stack running and `LANGFUSE_*` env
  vars point at it.

## Gotchas

- **Forgetting to call `install_litellm_callback()`.** No traces
  surface. This is the most common reason a Forge user reports
  "tracing doesn't work." `strata-forge doctor` reports whether the
  callback is installed.
- **Eager import of `langfuse` at module top.** Breaks the
  no-extras-installed import contract. Every Langfuse-touching
  function lazy-imports inside its body.
- **Not no-op-ing when Langfuse isn't configured.** Production calls
  should never crash because Langfuse keys are missing. Every Langfuse-facing public
  function checks `get_client()` and exits early when it returns
  `None`.
- **Reverse imports.** A module under `strata_forge.llm` reaching for
  `strata_forge.tracing` violates ADR 0008. The dependency arrow points one
  way.

## When to update this file

- Adding a new public function to `__init__.py`.
- Adding a new file under `src/strata_forge/tracing/`.
- Changing the lazy-import pattern.
- Changing what counts as "configured" (currently `enabled` on
  `LangfuseConfig`).
- Adding a new dependency to the `[langfuse]` extra.

## Structured provider observation boundary

`prompt_provider_values.py` and `prompt_provider_types.py` own fixed allowlisted tagged
value projections and closed metadata, independent of the opaque-byte profile. They do not
invoke providers, inspect unselected values/config/credentials, install callbacks or alter
portable prompt canon. Provider wrappers live above llm through explicit injection only;
llm/prompts must never import tracing. Unsupported/overbudget observation is loss, not a
permission/spend decision or a reason to suppress/retry/replace an owning model call. Existing
Langfuse/default/no-op behavior and provider/cache/fallback/generation sources stay separate.

## Explicit native stream observation

`prompt_streams.py` is an explicit offline-opt-in wrapper above the existing ProviderClient
interface, with closed counts/timing metadata in `prompt_stream_types.py`. It selects no
provider/config/default and installs no telemetry/export hook. It observes exact native
async generators only; unsupported returns forward by identity with loss, not admitted facts.
Do not change the accepted nonstreaming wrapper's transparent stream identity behavior.

Native iterator/advance protocol views must preserve genuine outcomes and controls without
new tasks, shields, read-ahead, item inspection/serialization or automatic native close.
Working state is view-owned with weak issuance only, never rooted map values. Observation
may keep only bounded weak view references for fixed-slot extra-argument retirement clearing;
no user/native callback or process/terminal inference. Keep all lower llm/provider/prompts/
portable modules independent of tracing. The source binding qualifies caller-provided local
bytes; counts/first-item timing do not establish provider execution, wire, TTFT or model success.

Stream admission reserves the entry before constructing its metadata. Any allocation failure
removes a partially published record and accounts for that reserved hole exactly once. The
fixed 4 KiB header plus 4 KiB lifetime reservations inside 256 KiB admits an effective 63
entries under MAX_STREAMS=64; do not claim all 64 fit or silently widen budgets.
