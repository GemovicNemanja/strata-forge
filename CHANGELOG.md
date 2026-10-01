# Changelog

All notable changes to `strata-forge` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/). Changes accumulate under **Unreleased** as they merge
to `dev`; cutting a release renames that heading to the version and its date (see `CLAUDE.md`
§9b for what forces a release and which bump it takes).

## [Unreleased]

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
    `previous_response_id`). A turn's output items (encrypted reasoning, text with its `phase`,
    function-call references) come back on `LLMResponse.provider_items` and the final
    `ResponseChunk`, ride on `AssistantMessage.provider_items` through both tool loops and
    `PendingToolCalls.messages`, and are replayed verbatim to the provider that produced them.
  - `response.incomplete` reports `length` / `content_filter`, `response.failed` and `error`
    events raise a mapped `ProviderError` (`map_responses_error`), and a stream that ends before
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
  LiteLLM's `cache_creation_input_tokens`.
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

[Unreleased]: https://github.com/GemovicNemanja/strata-forge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GemovicNemanja/strata-forge/releases/tag/v0.2.0
