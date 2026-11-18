# Agent rules — strata_forge.plugins

Read [`docs/modules/plugins.md`](../../../docs/modules/plugins.md) and
[ADR 0021](../../../docs/architecture/adr/0021-trusted-plugins.md) first. This module decides
whose code runs on a user's machine; every rule below is a security property.

- **Never an import string.** An entry point is found by NAME in the pinned distribution's own
  metadata (`registered_entry_point`), never by `importlib.import_module` on a value from a spec,
  a manifest or a request, and never across all installed distributions.
- **The allow-list is the orchestrator's, not the spec's.** `plugin_runner` reads it from
  `STRATA_ALLOWED_PLUGINS`, which an orchestrator sets from its own configuration. Never add a
  spec field, hyperparameter or request field that widens it.
- **Exact pins only.** A build is distribution + version + source + full commit, and an artifact
  hash: a git build's `files_sha256`, an `hf` wheel's `wheel_sha256`. No ranges, no branches, no short ids, no local versions. An index install
  or an editable checkout never satisfies a pin.
- **The runner installs, without credentials.** `environment.py` builds each pin its own venv,
  never the engine's, with `install_environ` (no credential-shaped name, a fresh `HOME`, no pip or
  git configuration). An `hf` wheel is hash-checked before pip sees it. Never pass a token into
  the install, and never reuse an environment without re-verifying it.
- **The control-plane half is pure.** `manifest.py`, `pins.py` (apart from `installed_mismatch`
  and `registered_entry_point`, which read local metadata on the target) and `preflight.py` do no
  I/O and import nothing from a plugin. A control plane must be able to call them on a manifest
  it read as bytes.
- **Never run plugin code next to the token.** The runner runs with `-I`, refuses a token from
  the environment, and starts the host only through `sandbox.sandbox_command` with the
  isolation `choose_sandbox` picked; never weaken that order or fall back to `none` without the
  operator's `allow_unisolated`. Never import anything from a plugin environment into the runner.
- **Never touch the plugin's output in place.** `child.run_plugin_process` must have confirmed
  the group gone; then `output.stage_output` copies it and everything after reads the copy.
- **The plugin never holds a credential.** The runner downloads every input and pushes every
  output; the plugin runs in `host.py`'s child process with `plugin_environ` (an allow-list with
  every credential-shaped name dropped, the Hub offline on an empty cache). Do not pass the
  token, the secrets file path or the spec into the child, and do not let a plugin fetch its own
  inputs: a model it needs is a declared `model` input.
- **Configuration is closed.** Flat typed scalars declared in the manifest; unknown keys are
  refused. No lists, objects or code in configuration.
- **Step scorers get steps, not files; the platform scores and records.** The host feeds `Step`
  batches and writes the scores; `check_scores` refuses anything but one result per step inside
  the declared `score_kind` (never clip a probability) or a declared `Unscored` reason. Never
  re-split steps or join a structured context. The runner re-checks the written scores against
  the input before the push; never trust the host's file. Never put an aggregation in
  `scores.parquet` or pick one by default: reductions are opt-in and go to the diagnostic file.
- **Plugin-reported is labelled.** `PluginContext.report` keys every metric `plugin.<name>`; GPU
  telemetry is sampled by the runner, outside the plugin's process.
- Outputs are pushed only to a PRIVATE repository; an existing public one is refused.
