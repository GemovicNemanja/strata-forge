# Scripts

One-off and infrequently-run operational helpers — VCR cassette refresh, Langfuse prompt/dataset sync, registry validation, live provider smoke tests (`smoke_responses.py`, which spends real tokens on the user's own key), ad-hoc data wrangling.

- `run_eval_gate.py` — the canonical eval-regression gate the nightly `eval-gate` job runs.
- `smoke_clean_install.py` — builds every config a run builds, inside a fresh install of the engine; the nightly `clean-install.yml` job runs it with the interpreter of an empty venv holding only `strata-forge[<extras>]` (it refuses to run against the source tree). `--write-override PATH TEXT` validates the drill's override (plain version pins only) before the install.
- `prompt_reference.mts` — Node 24 reference consumer/renderer for authoritative Python prompt
  builds. `node --test scripts/prompt_reference.test.mts` verifies the portable fixture corpus;
  it does not compile authored definitions or publish a second compiler identity.
- `import_operational_prompts.py --output PATH` — imports only the hash-pinned authorized
  archived Strata operational prompt snapshot, verifies every unchanged baseline, and writes
  Python build/render conformance evidence for the Node consumer. It reads no research prompts
  and imports or executes no server source. See [prompt composition](../docs/modules/prompt-composition.md).

Each script is invokable via `uv run python scripts/<name>.py`. Where useful, scripts are also wired to `make` targets.
