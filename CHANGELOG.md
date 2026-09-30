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
  with `engine version mismatch` before any other field is read: the version half must equal the
  installed `strata_forge.__version__`, and a `"<version>+<commit>"` value is accepted only when
  the installed distribution's PEP 610 `direct_url.json` records that exact commit (an index
  install or an editable checkout, which record none, cannot satisfy a commit claim). `None`
  makes no claim and is accepted. `strata_forge.pipelines.SPEC_VERSION` exports the value an
  orchestrator writes.

## [0.2.0] - 2026-08-27

The release the handshake above compares against. Earlier history is in the git log.

[Unreleased]: https://github.com/GemovicNemanja/strata-forge/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/GemovicNemanja/strata-forge/releases/tag/v0.2.0
