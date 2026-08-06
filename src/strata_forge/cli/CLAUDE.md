# Agent rules — strata_forge.cli

`strata_forge.cli` is the Typer-based command-line surface for the
library. Every Forge module that has a useful operator
workflow surfaces a command here: ``doctor``, ``chat``,
``prompts``, ``datasets``, ``eval``, ``experiments``,
``compute``, ``train``, ``serve``.

## Purpose

- Thin Typer wrappers that defer to the corresponding Forge
  modules — the CLI never re-implements business logic.
- Common operator workflows reachable in one shell line: chat
  against a model, list/inspect prompt and dataset registries,
  run a quick eval, submit a compute job, launch SFT/DPO,
  print a serving task.
- Persistent local state for compute and experiments under
  ``~/.forge/jobs`` and ``~/.forge/experiments`` so subsequent
  commands can pick up where prior ones left off.

## Boundaries

- **Owns:** ``main.py``, ``helpers.py``, ``doctor.py``,
  ``chat.py``, ``prompts.py``, ``datasets.py``, ``eval_cmd.py``,
  ``experiments.py``, ``compute.py``, ``train.py``, ``serve.py``.
- **Imports from inside ``forge``:** every other module —
  ``strata_forge.cli`` is the top of the dependency arrow.
- **Does NOT** carry business logic, persistence layers other
  than the small ``~/.forge`` state directory, or its own type
  definitions. Pydantic shapes belong to their owning modules.
- **External deps:** ``typer`` and ``rich`` (both in core install).

## Public API

The only public surface is :data:`app` (the Typer entry point
registered as the ``forge`` script). Subcommand groups are
private implementation detail — users invoke them via the CLI,
not as Python imports.

Helpers exposed for tests / cross-command reuse:

- :func:`run_async` — bridge to async functions.
- :func:`error_exit` — :class:`NoReturn` that prints a red error
  and raises ``typer.Exit``.
- :func:`prompt_store_from_settings` /
  :func:`dataset_store_from_settings` — pick Langfuse if
  configured, otherwise fall back to in-memory stores.

## Internal patterns

- **One file per top-level subcommand or subcommand group.**
  Each file exposes either a single Typer callable (``doctor``,
  ``chat``) or an ``app: typer.Typer`` instance attached via
  ``add_typer`` (everything else).
- **Async work goes through :func:`run_async`.** Every command
  body that touches I/O delegates to an ``async def _impl`` and
  runs it via ``asyncio.run``.
- **Heavy modules import lazily inside command bodies.** Don't
  import :mod:`strata_forge.training` or :mod:`strata_forge.compute.backends`
  at module load — the CLI must launch instantly even with the
  optional extras absent.
- **State files under ``~/.forge``.** ``~/.forge/jobs/<id>.json``
  holds compute job state (Job + backend + backend kwargs);
  ``~/.forge/experiments/<name>.md`` holds eval reports. Both
  paths are overridable via monkeypatching for tests.
- **Rich for output formatting.** Use :class:`Console` /
  :class:`Table` for human-facing output; structured JSON is
  available via specific subcommands when needed.

## Test expectations

- Unit tests under ``tests/unit/cli/``, one file per source
  module.
- Coverage target: ≥ 85 % line (lower than other modules because
  the interactive REPL loop and live cloud-backend constructors
  aren't exercised by unit tests).
- Use Typer's :class:`CliRunner` to invoke commands; never
  import the command function directly.
- Fake heavy backends (LLMClient, SFTRunner, etc.) via
  ``monkeypatch``; the CLI tests should never touch real
  providers or load real model weights.

## Gotchas

- **Don't add command-line flags that duplicate module-level
  Pydantic config.** Configs flow through ``strata_forge.config`` and
  env vars; flags are only for per-invocation knobs.
- **``error_exit`` is :class:`NoReturn`.** Don't write ``return``
  after calling it — pyright/ruff will flag the dead branch.
- **Typer's ``Argument`` and ``Option`` are function-calls in
  defaults.** The project's ``flake8-bugbear`` config whitelists
  these two via ``extend-immutable-calls`` so B008 doesn't fire
  on the documented Typer pattern.
- **``strata_forge.llm.registry.registry`` is a Registry instance, not a
  module.** When tests monkeypatch it, they target
  ``sys.modules["strata_forge.llm.registry"]``, not
  ``import strata_forge.llm.registry as m`` (the latter binds to the
  Registry instance via the package's ``from … import registry``
  re-export).
- **State files in ``~/.forge``** must be redirected in tests via
  ``monkeypatch.setattr`` on the module-level ``_STATE_DIR`` /
  ``_REPORT_DIR`` symbols; don't create writes under the real
  home directory during the suite.

## When to update this file

- Adding a new top-level subcommand or subcommand group.
- Adding a new shared helper to ``helpers.py``.
- Changing the on-disk state layout under ``~/.forge``.
- Adding a new heavy external dep used by a command.
