# Changelog

All notable changes to `strata-forge` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Changes accumulate under **Unreleased** as they merge
to `dev`; cutting a release renames that heading to the version and its date (see `CLAUDE.md`
§9b for what forces a release and which bump it takes).

## [Unreleased]

### Added

- `strata_forge.core.redact`: `Redactor`, the one redaction implementation for every string a
  credential-holding process surfaces and every console a relay forwards. It removes each given
  value as written, line by line, in all four Unicode normal forms, base64-encoded at any
  alignment (standard and URL-safe), percent-encoded and JSON- or `repr`-escaped, plus
  `DEFAULT_PATTERNS` (Hugging Face, OpenAI/Anthropic `sk-`, GitHub, AWS access key ids, JWTs,
  URL userinfo, bearer credentials), plus PEM private-key blocks with or without their footer,
  which stay on with `patterns=()`. Each maximal redacted run becomes one fixed `***`, and runs
  separated only by short whitespace count as one, so the output encodes neither a secret's
  length nor its line count. A value shorter than 8 characters without its surrounding
  whitespace, or made only of `*`, is refused with `ValidationError`; a pydantic `SecretStr` is
  unwrapped. A redactor and its streams refuse to be pickled or copied.
- `Redactor.stream()` returns a `RedactingStream` for text read incrementally: `feed` holds back
  the last `max_len - 1` characters, `flush` releases them, and the concatenated output equals
  `redact` of the concatenated input whatever the piece boundaries, so a secret split across two
  console reads is still caught. `gap()` handles a read that dropped bytes, and a fresh stream
  resumed mid-transcript: the held tail and the first `max_len - 1` characters after the hole
  are masked. `docs/modules/compute.md` states the contract a console relay follows.
- `strata_forge.pipelines._common.run_redactor` builds a run's redactor from its write token.
- A nightly clean-install job (`.github/workflows/clean-install.yml`). It installs each extras
  string a run's machine installs (`[finetuning,storage]`, `[serving,storage,hf]`) plus `[all]`
  into empty Python 3.14 venvs from the public index, with no lockfile and no constraints, and
  `scripts/smoke_clean_install.py` then builds, offline and without a GPU, every enabled
  fine-tuning method x dataset format x adapter through the runner's own spec path and
  `build_trainer` (the TRL config, the peft config, the bitsandbytes config, and the runner's
  trainer keywords bound against the installed trainer's signature), the batch-inference spec,
  requests and results parquet, and the vLLM command line through the installed vLLM's own
  entrypoint parser, with each `VLLM_` variable the runner sets checked as one vLLM still reads.
  A breaking upstream release fails there the day it ships instead of on a user's machine. A
  QLoRA case warns while `[finetuning]` does not install `bitsandbytes` (QLoRA cannot load a model
  on such a machine) and fails once the extra names it but the install lacks it. It also runs on
  any branch push other than `main` that changes `pyproject.toml`, and a manual run takes an
  `override` of plain version pins (no URL, path or option) as a drill that must turn it red. The
  schedule, which runs from `main`, only dispatches the job on `dev` and reports that run's
  result, so the unlocked install never runs in the default branch's cache scope.

### Changed

- `sanitize` and every runner message (phase captions, error events, the stderr failure reason)
  go through `Redactor`, so they also catch the token's encoded forms and the wider set of
  credential shapes. A write token shorter than 8 characters fails the run before any work.

### Fixed

- CLI error messages printed bracketed text as rich markup, so an extra's name
  (`strata-forge[finetuning]`) vanished from the line and exception text containing `[/...]` could
  raise inside the error path. They are printed literally.

### Security

- The batch-inference runner's per-row `error` column, written into the results and pushed to
  the Hub, is redacted. It used to carry each failed row's exception text unscrubbed.
- `release.yml`'s build job and every `nightly.yml` job no longer restore the Actions cache. Every
  ref restores the default branch's cache scope, so an entry written there would otherwise shape
  the distributions the publish job uploads, or run beside the nightly's provider keys and write
  token.

- The `openai_compat` provider documents that `OpenAICompatConfig.base_url` is caller-trusted:
  the provider validates nothing about the URL and keeps no private-address block, because
  loopback is its normal target. Its transport follows redirects. A hop to another host or port
  drops the `Authorization` header, so the configured `api_key` stays on the configured host, but
  nothing else is dropped: a 307/308 re-sends the request body, an `extra_headers` credential
  follows every hop, and the final reply (or an error quoting it) reaches the caller. A caller
  that takes `base_url` from an untrusted party therefore compares the whole URL for exact
  equality with one it allows, never the host or a prefix, and passes `api_key` explicitly
  (`docs/modules/llm.md`, "OpenAI-compatible endpoints"). Unit tests run the real LiteLLM, OpenAI
  client and `httpx` stack, streaming and not, against loopback servers and pin both halves, so a
  dependency upgrade that changes either fails there.

## [0.3.0] - 2026-10-01

### Added

- Every runner spec (`RunSpec`, `FinetuneSpec`) carries `engine_version: str | None`, the engine
  the orchestrator validated the spec against. `load_config` refuses any other installed engine
  with `engine version mismatch`, read off the parsed JSON before the model validates any other
  field (so a stale machine is blamed even when the spec also carries a field it does not know).
  The version half must equal both the executing `strata_forge.__version__` and the version the
  installed distribution's metadata records; a `"<version>+<commit>"` value is accepted only when
  the installed distribution's PEP 610 `direct_url.json` records that exact full commit id (an
  index install or an editable checkout, which record none, cannot satisfy a commit claim, and a
  `+` followed by anything else is refused as malformed). A spec class that does not declare the
  field is a `TypeError`, so a runner cannot opt out. `strata_forge.pipelines.SPEC_VERSION`
  exports the value an orchestrator writes.
- `None` (no claim) is accepted as the transition for an orchestrator from before the handshake,
  and recorded: the run's first `phase` event and a stderr `warning:` line say the installed
  engine was not checked. Setting `FORGE_REQUIRE_ENGINE_VERSION=1` in the runner's environment
  refuses a missing claim like any other mismatch, so an orchestrator that stamps every spec can
  close the transition on its machines at once. The default flips to refusal in the first minor
  release after every orchestrator that launches this engine stamps `engine_version`.
- The version compare is string equality, never a PEP 440 normalisation (`0.3`, `v0.3.0` and
  `0.3.0.post0` are refused for `0.3.0`), and the mismatch message caps the echoed claim.
- The model registry carries the current Anthropic and OpenAI lineups: `claude-fable-5-1`,
  `claude-opus-5-5`, `claude-sonnet-5-5` (anthropic, bedrock and vertex routes) and
  `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna` (openai route), each copied from the vendor's own
  model, pricing and guide pages. A native `anthropic` / `openai` route refused every one of these
  ids with `unknown_model`.
  - The three Claude 5.x entries have `structured_output: false`: they reject the forced tool
    call that structured output on an Anthropic route relies on, so `complete_structured` now
    refuses them pre-flight with `capability_missing`. Tool calling works.
  - All three GPT-6 entries have `tool_calling: true` on their Responses API route (below).
- OpenAI's Responses API ([ADR 0018](docs/architecture/adr/0018-openai-routes-speak-the-responses-api.md)).
  A registry route carries `wire_api: chat_completions | responses` (`responses` only on `openai`
  and `azure`), copied onto `ModelRoute.wire_api`. Every OpenAI model's `openai` route, and the
  `azure` routes of `gpt-5.5`, `gpt-5.5-thinking` and `gpt-5.5-instant`, speak it through
  `litellm.aresponses`; `openai_compat` (OpenRouter, a self-hosted vLLM, Ollama), Anthropic,
  Vertex and Bedrock stay on Chat Completions. The call surface is unchanged.
  - Requests are stateless (`store: false`, `include: ["reasoning.encrypted_content"]`, never
    `previous_response_id`), and `provider_extras` that set `store`, `previous_response_id`,
    `conversation` or `background` raise `ValidationError`. A turn's output items (encrypted
    reasoning, text with its `phase`, function-call references) come back on
    `LLMResponse.provider_items` and the final `ResponseChunk`, ride on
    `AssistantMessage.provider_items` through both tool loops and `PendingToolCalls.messages`,
    and are replayed verbatim to the provider that produced them (a replayed message with the
    `status` its input shape requires). The `complete()` cache never stores them.
  - In `provider_extras`, an `include` adds to the encrypted-reasoning include, a `text` object
    merges into the computed one (a structured-output format survives unless it sets its own),
    and `reasoning_effort` / `verbosity` / `response_format` become `reasoning.effort` /
    `text.verbosity` / `text.format`.
  - A registered model that LiteLLM's model map lacks is registered with LiteLLM from the Forge
    registry (streaming flag, limits, prices) before a Responses call. LiteLLM fakes the stream
    of a model it cannot look up, so the GPT-6 and `gpt-5.5` streams otherwise depended on
    LiteLLM fetching its remote map at import.
  - `response.incomplete` reports `length` / `content_filter`, `response.failed` and `error`
    events raise a mapped `ProviderError` (`map_responses_error`, reading an `error` event's code
    and message at its top level or in its nested `error` object), and a stream that ends before
    its terminal event raises `ProviderServerError` instead of reporting a clean stop.
  - `AzureConfig.responses_api_version` (default `v1`, env `AZURE_OPENAI_RESPONSES_API_VERSION`)
    selects Azure's `/openai/v1/responses` endpoint; Chat Completions keeps `api_version`. An
    OpenAI organization (`OPENAI_ORG_ID`) is sent as the `OpenAI-Organization` header, which
    LiteLLM's Responses path does not set from `organization`.
  - `Tool` / `ToolDeclaration.to_openai_responses_schema()`, `to_openai_responses_tool_schema`
    and `ImageContent.to_openai_responses_format()` render the Responses wire shapes.
- `Capabilities.sampling_params`: whether a model accepts `temperature` / `top_p` at its default
  reasoning effort. The client refuses them pre-flight with `ValidationError` where the provider
  would reject them: on a Responses route unless the caller sets the reasoning effort to `none`
  (or the model defaults to it), and on any route of a model with `sampling_params: false` (the
  Claude 5.x entries).
- `scripts/smoke_responses.py`: a live two-leg tool-call smoke test of every registered OpenAI
  Responses route (reads `OPENAI_API_KEY` from the environment and never prints it).

### Changed

- The `gpt-5.5` family's `openai` routes speak the Responses API: `max_tokens` is sent as
  `max_output_tokens`, which bounds reasoning as well as visible output, `response_format` as
  `text.format`, and a `provider_extras` `reasoning_effort` as `reasoning.effort`. This makes tool
  calls work on `gpt-5.5` at its default effort (Chat Completions rejected them) and makes
  `gpt-5.5-pro` callable at all (it is not served on Chat Completions).
- `Usage.input_tokens` is reported net of cached reads and cache writes on every route. Providers
  include both in their input count while `compute_cost` prices each bucket separately, so cached
  tokens were billed twice; Chat Completions usage now also reports `cache_write_tokens` from
  LiteLLM's `cache_creation_input_tokens`. A `BudgetContext` token ceiling counts every bucket
  (uncached input, cache reads and writes, output), so the netting does not loosen it.
- The `unknown_model` message from `resolve()` states only the failure (`Unknown model: '<id>' is
  not available on ...`), because it can reach an application's end users; the remediation (the
  registry file, the `openai_compat` escape hatch) is logged as an `unknown_model` warning. The
  `reason` is unchanged.
- A chain entry pinned only to `openai_compat` is not checked against the registry's capability
  flags even when its id equals a registered name: the operator's endpoint serves its own model.
- Registry corrections from the vendor pages: `gpt-5.5` and `gpt-5.5-pro` have a 1,050,000-token
  context window and 128,000 max output tokens; `claude-haiku-4-5` has 64,000 max output tokens.
  The registry header states the sourcing rule and marks unchecked values with `# TBD verify`.
- The `litellm` floor is `>=1.83,<2`, the release the Responses path is tested against (releases
  before 1.66 have no `litellm.aresponses`).

### Removed

- The `azure` route of `gpt-5.5-pro`. Azure does not list the model for its Responses API, and the
  model takes no tools on Chat Completions, so the route failed every tool call; a registry test
  now refuses a Chat Completions route on any tool-capable OpenAI model.

## [0.2.0] - 2026-08-27

The first release this changelog records. Earlier history is in the git log.

[Unreleased]: https://github.com/GemovicNemanja/strata-forge/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/GemovicNemanja/strata-forge/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/GemovicNemanja/strata-forge/releases/tag/v0.2.0
