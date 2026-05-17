# VCR cassette tests

Provider-touching integration coverage that replays previously-recorded HTTP
exchanges via [`pytest-recording`](https://pytest-recording.readthedocs.io/).
The goal: keep CI hermetic (no live keys) while still exercising the real
LiteLLM payloads each provider expects.

## Layout

```
tests/vcr/
├── conftest.py             # vcrpy filter config + scrubbing
├── test_providers.py       # the test matrix (provider × scenario)
└── cassettes/
    ├── openai/
    ├── anthropic/
    ├── vertex/
    ├── bedrock/
    ├── azure/
    └── openai_compat/
```

Cassettes live under `cassettes/<provider>/<scenario>.yaml`. Tests skip
gracefully when the matching cassette is missing.

## Scenarios

Per the Phase-1 plan, every provider gets a cassette per scenario when
recordings are feasible:

- `basic_completion` — a small non-streaming call returns a non-empty response.
- `streaming` — chunks accumulate into a coherent reply.
- `structured_output` — Pydantic schema dispatch returns a parsed instance.
- `multimodal` — image input round-trips for vision-capable models.
- `single_tool_call` — model emits one tool call; caller invokes it manually.
- `multi_turn_tool_loop` — `run_tool_loop` survives two tool iterations.
- `rate_limit_retry` — the `@retry` policy catches a 429 and succeeds on retry.
- `content_filter` — `ProviderContentFilterError` short-circuits the chain.

Cross-provider scenarios (in `tests/vcr/cassettes/_cross/`):

- `provider_fallthrough` — same logical model, one provider 429s, the next succeeds.
- `model_fallthrough` — first model exhausted, second model picks up.

## Recording new cassettes

You need live keys for every provider you want to cover. Set them in `.env`
or your shell, then:

```
RECORD=1 make vcr-record
```

`RECORD=1` flips `vcr_config.record_mode` from `"none"` to `"once"`. Every
header named in `_HEADERS_TO_SCRUB` (see `conftest.py`) is replaced with
`[REDACTED]` *before* the cassette is written. If you record without
scrubbing — visible API keys, AWS sigv4 signatures, GCP bearer tokens —
**treat them as leaked and rotate immediately**.

After recording, inspect the produced YAML for any field you don't
recognize (model-specific headers, request IDs that might still be
sensitive) before committing.

## Replay

```
make vcr-replay
```

This runs `tests/vcr/` with `record_mode="none"`. Tests fail loudly if the
request shape drifts from what the cassette captured — that's the early
warning that an upstream library (LiteLLM, the provider SDK) changed its
wire format and the LLM module needs an adjustment.

## Adding a new scenario

1. Add the test function to `test_providers.py`.
2. Decide which providers are eligible (some scenarios don't apply to
   every provider — e.g. multimodal needs vision support).
3. Record cassettes for those providers with `RECORD=1`.
4. Commit the resulting YAMLs and update this README's scenario list.
