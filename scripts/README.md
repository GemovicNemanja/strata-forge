# Scripts

One-off and infrequently-run operational helpers — VCR cassette refresh, Langfuse prompt/dataset sync, registry validation, live provider smoke tests (`smoke_responses.py`, which spends real tokens on the user's own key), ad-hoc data wrangling.

- `run_eval_gate.py` — the canonical eval-regression gate the nightly `eval-gate` job runs.
- `smoke_clean_install.py` — builds every config a run builds, inside a fresh install of the engine; the nightly `clean-install.yml` job runs it with the interpreter of an empty venv holding only `strata-forge[<extras>]` (it refuses to run against the source tree).

Each script is invokable via `uv run python scripts/<name>.py`. Where useful, scripts are also wired to `make` targets.
