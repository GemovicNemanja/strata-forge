# Scripts

One-off and infrequently-run operational helpers — VCR cassette refresh, Langfuse prompt/dataset sync, registry validation, live provider smoke tests (`smoke_responses.py`, which spends real tokens on the user's own key), ad-hoc data wrangling.

Each script is invokable via `uv run python scripts/<name>.py`. Where useful, scripts are also wired to `make` targets.
