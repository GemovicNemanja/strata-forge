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
  model and pricing pages. A native `anthropic` / `openai` route refused every one of these ids
  with `unknown_model`.
  - `gpt-6.1-sol` has `tool_calling: false`: it calls tools only through the Responses API,
    which the `openai` provider does not speak, so a call with tools fails pre-flight with
    `capability_missing`.
  - The three Claude 5.x entries have `structured_output: false`: they reject the forced tool
    call that structured output on an Anthropic route relies on. Tool calling works.
- `ProviderRoute.tool_call_reasoning_effort`: the `reasoning_effort` an `openai` / `azure` route
  sends on every call that carries tools. `gpt-6-luna` sets `none`, the only effort at which its
  Chat Completions endpoint accepts function calling; calls without tools keep the default
  effort, and a `provider_extras` value still overrides it. The field is refused on any other
  provider.

### Changed

- `resolve()` names the registry file (`strata_forge/llm/registry_data.yaml`) and the
  `openai_compat` escape hatch in the `unknown_model` message. The message still starts with
  `Unknown model: '<id>'` and the `reason` is unchanged.

## [0.2.0] - 2026-08-27

The first release this changelog records. Earlier history is in the git log.

[Unreleased]: https://github.com/GemovicNemanja/strata-forge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GemovicNemanja/strata-forge/releases/tag/v0.2.0
